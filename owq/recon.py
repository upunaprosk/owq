import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ_OWQ:
    def __init__(self, layer, n_out):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.n_out = n_out
        self.n_nonout = W.shape[1] - n_out
        self.owq = n_out > 0
        self.out_quantizer = None
        self.ids = None

    def add_batch_x01(self, inp):
        inp1 = inp.clone()
        if len(inp1.shape) == 2:
            inp1 = inp1.unsqueeze(0)  # Perhaps, we don't need this: Convert to 3D tensor [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = inp1.shape
        if batch_size != 2:
            # print(f'add_batch_x01: {batch_size} != 2')
            return

        inp1 = inp1.to(device=self.dev)
        tmp = inp1.shape[0]  # batch size
        X0 = inp1[0]  # Shape: [seq_len, hidden_dim]
        X1 = inp1[1]  # Shape: [seq_len, hidden_dim]
        X0 = X0.t()  # Shape: [hidden_dim, seq_len]
        X1 = X1.t()  # Shape: [hidden_dim, seq_len]

        if not hasattr(self, "H_x01"):
            try:
                self.H_x01 = torch.zeros((self.columns, self.columns), device=self.dev)
            except torch.OutOfMemoryError:
                print("Memory: OOM H allocate bypass")
                torch_empty_cache()
                self.H_x01 = torch.zeros((self.columns, self.columns), device=self.dev)

        else:
            self.H_x01 *= self.nsamples / (self.nsamples + tmp)

        samples_Hx01 = (self.nsamples + tmp) / 2
        # print(f"Input shape after processing: {inp1.shape}")
        # print(f"Accumulated samples: {self.nsamples}")
        delta = math.sqrt(2 / samples_Hx01) * (X0 - X1).float()
        try:
            self.H_x01 += delta.matmul(delta.t())
        except torch.OutOfMemoryError:
            print("Memory: OOM cpu bypass for process batch matmul")
            torch_empty_cache()
            device = self.H_x01.device
            CPU = torch.device("cpu")
            self.H_x01, delta = self.H_x01.to(device=CPU), delta.to(device=CPU)
            self.H_x01 += delta.matmul(delta.t())
            self.H_x01 = self.H_x01.to(device=device) # move back


    def bias_x01_sorting(self, percdamp=.01, frob_norm=None, dW_quant=None):
        if not hasattr(self, "H_x01"):
            return None

        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H # Hinv - upper Cholesky of inverse of H

        # Calculate debiasing changes for W matrix
        W = self.layer.weight.data.float()
        Jt = self.H_x01.matmul(W.transpose(0, 1))  # Jt - transposed gradient, its shape is the same as of W^t
        dW = Hinv.transpose(0, 1).matmul(Hinv.matmul(Jt)).transpose(0, 1) # potential debiasing correction: W = W - dW
        column_sensitivity = dW.pow(2).sum(dim=0) # L2 squared norm of dW columns
        hessian_diag_x01 = torch.diag(self.H_x01)
        if frob_norm is not None:
            hessian_diag_x01 *= frob_norm

        # agreement between dW and dW_quant
        if dW_quant is not None:
            dW2 = dW * dW_quant
            dW_product = dW2.sum(dim=0)
            dW_product2 = dW2.abs().sum(dim=0)
        else:
            dW_product = torch.zeros_like(column_sensitivity)
            dW_product2 = torch.zeros_like(column_sensitivity)

        # H_diag as in hessian_sorting
        hessian_diag = torch.diag(self.H)
        if frob_norm is not None:
            hessian_diag *= frob_norm

        # Sort with indices
        sorted_values, sorted_indices = torch.sort(column_sensitivity, descending=False)
        hessian_diag_x01 = hessian_diag_x01[sorted_indices]
        hessian_diag = hessian_diag[sorted_indices]
        dW_product = dW_product[sorted_indices]
        dW_product2 = dW_product2[sorted_indices]

        # del self.H_x01
        return torch.stack([sorted_values, sorted_indices, hessian_diag_x01, hessian_diag, dW_product, dW_product2], dim=0)


    def add_batch(self, inp, out):
        if len(inp.shape) == 3 and inp.shape[0] == 2:
            self.add_batch_x01(inp)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def hessian_sorting(self, actorder=False, frob_norm=None, custom=None, add_custom=False):
        if not custom:
            H = self.H

            if not self.owq:
                if actorder:
                    self.ids = torch.argsort(torch.diag(H), descending=True)
                return torch.tensor([])

            temp_mask = torch.full([self.columns], True, device=self.dev)

            H_diag = torch.diag(H)
            if frob_norm is not None:
                H_diag *= frob_norm
            descending_ids = torch.argsort(H_diag, descending=True)

            temp_mask[descending_ids[:self.n_out]] = False
            if actorder:
                ids = torch.cat([descending_ids[self.n_out:], descending_ids[:self.n_out]])
            else:
                ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], descending_ids[:self.n_out]])

            self.ids = ids
            return torch.sort(descending_ids[:self.n_out])[0].to(torch.int32)
        elif 'random' in custom:
            descending_ids = torch.randperm(self.columns, device=self.dev)

            if not self.owq:
                if actorder:
                    self.ids = descending_ids
                return torch.tensor([])

            temp_mask = torch.full([self.columns], True, device=self.dev)
            temp_mask[descending_ids[:self.n_out]] = False
            if actorder:
                ids = torch.cat([
                    descending_ids[self.n_out:],
                    descending_ids[:self.n_out]
                ])
            else:
                ids = torch.cat([
                    torch.arange(self.columns, device=self.dev)[temp_mask],
                    descending_ids[:self.n_out]
                ])

            self.ids = ids
            # print("Random outliers:", outliers.cpu().tolist())
            return descending_ids[:self.n_out].to(torch.int32)
        elif custom is not None and add_custom:
            # same as usual mode (w/o custom) but read custom columns table
            # and then take n_out columns from two sources: n_out/2 largest columns
            # from Hessian and n_out/2 from the custom columns
            H = self.H

            if not self.owq:
                if actorder:
                    self.ids = torch.argsort(torch.diag(H), descending=True)
                return torch.tensor([])

            n_out = self.n_out // 2 # n_out > 0 and even

            temp_mask = torch.full([self.columns], True, device=self.dev)

            H_diag = torch.diag(H)
            if frob_norm is not None:
                H_diag *= frob_norm
            descending_ids = torch.argsort(H_diag, descending=True)

            temp_mask[descending_ids[:n_out]] = False
            if actorder:
                ids = torch.cat([descending_ids[n_out:], descending_ids[:n_out]])
            else:
                ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], descending_ids[:n_out]])

            # n_out indices taken by sorted Hessian
            taken_list = descending_ids[:n_out].cpu().tolist()

            # get n_out from custom column list
            idx_list_full = []
            idx_list = []
            with open(custom, 'r') as f:
                for line in f:
                    line = line.strip()
                    _, idx_str, sens_str = line.split()[:3]
                    idx = int(idx_str)
                    idx_list_full.append(idx)
                    if idx in taken_list:
                        continue
                    idx_list.append(idx)
            print(f"Hessian n_out={taken_list}, bias n_out={idx_list[-n_out:]}")
            for x in idx_list_full[-n_out:]:
                if x in taken_list:
                    print(f"Column {x} is in both top lists")
            taken_list += idx_list[-n_out:] # take last n_out

            # compose ids again
            ids_cpu = [x for x in ids.cpu().tolist() if x not in taken_list]
            ids_cpu += taken_list

            self.ids = torch.tensor(ids_cpu, device=self.dev, dtype=torch.int32)
            return torch.sort(torch.tensor(taken_list, device=self.dev, dtype=torch.int32))[0]
        else:
            idx_list = []
            with open(custom, 'r') as f:
                for line in f:
                    line = line.strip()
                    _, idx_str, sens_str = line.split()
                    idx_list.append(int(idx_str))
            idx_list = idx_list[::-1]
            descending_ids = torch.tensor(idx_list, device=self.dev)

            if not self.owq:
                if actorder:
                    self.ids = descending_ids
                return torch.tensor([])

            temp_mask = torch.full([self.columns], True, device=self.dev)
            temp_mask[descending_ids[:self.n_out]] = False
            if actorder:
                ids = torch.cat([
                    descending_ids[self.n_out:],
                    descending_ids[:self.n_out]
                ])
            else:
                ids = torch.cat([
                    torch.arange(self.columns, device=self.dev)[temp_mask],
                    descending_ids[:self.n_out]
                ])

            self.ids = ids
            return torch.sort(descending_ids[:self.n_out])[0].to(torch.int32)

    def fasterquant(
            self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, debias_scale=0,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if actorder or self.owq:
            W = W[:, self.ids]
            self.H = self.H[self.ids][:, self.ids]

        self.quantizer.find_params(W[:, :self.n_nonout], weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        if hasattr(self, "H_x01"):
            Jt = self.H_x01.matmul(W.transpose(0, 1))  # Jt - transposed gradient, its shape is the same as of W^t
            dW = Hinv.transpose(0, 1).matmul(Hinv.matmul(Jt)).transpose(0, 1) # potential debiasing correction: W = W - dW

        CPU = torch.device("cpu")
        dw_abs_sum = torch.zeros((self.columns, ), device=CPU)
        dw_counts_minus = torch.zeros((self.columns, ), device=CPU)
        dw_counts_plus = torch.zeros((self.columns, ), device=CPU)
        dw_min = torch.zeros((self.columns, ), device=CPU)
        dw_max = torch.zeros((self.columns, ), device=CPU)


        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), (self.columns - self.n_out))], weight=True, num=40)

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()

                # limited debias correction
                if debias_scale is not None and hasattr(self, "H_x01"):
                    dw = dW[:, (i1 + i)]
                    dq = q - w
                    dw_a = dw.abs()
                    dq_a = dq.abs()
                    dw_s = dw.sign()
                    mask = (dw_a >= dq_a) # make only corrections that are large enough
                    dc = torch.minimum(dw_a, debias_scale * dq_a) # correction is capped by abs(debias_scale * dq)
                    # Note: What if abs(q[j]-w[j]) is small? How to get the step of quantization used in q[j]?
                    if mask.sum() > 0:
                        corr = dw_s[mask] * dc[mask]
                        w[mask] += corr # make correction in the "right" direction
                        # collect some statistics
                        q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                        dw_abs_sum[i1 + i] = float(dc[mask].sum())
                        se = (dq.sign() * dw_s).to(torch.int32) # agreement in direction
                        dw_counts_minus[i1 + i] = float((se[mask] < 0).sum())
                        dw_counts_plus[i1 + i] = float((se[mask] > 0).sum())
                        dw_min[i1 + i] = float(corr.min())
                        dw_max[i1 + i] = float(corr.max())

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder or self.owq:
            Q[:, self.n_nonout:] = W[:, self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return torch.stack([dw_abs_sum, dw_counts_minus, dw_counts_plus, dw_min, dw_max], dim=0)

    def free(self):
        self.H = None
        self.Losses = None
        self.ids = None
        if hasattr(self, "H_x01"):
            self.H_x01 = None
        torch.cuda.empty_cache()
