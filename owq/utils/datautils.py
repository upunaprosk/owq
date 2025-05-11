import numpy as np
import random
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
            
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        
        return testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test",trust_remote_code=True)
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        
        return testenc

def get_c4(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=True
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    else:
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', trust_remote_code=True
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc

def get_crows_pairs(nsamples, seed, seqlen, tokenizer, train, model=None):
    if not train:
        raise ValueError

    print(f"get_crows_pairs: seqlen={seqlen}")
    random.seed(seed)
    split = "test" # there is no "train" split in crows_pairs
    dataset = load_dataset("nyu-mll/crows_pairs", split=split, trust_remote_code=True)
    if model is not None and 'opt-125m' in model.lower():
        nsamples = len(dataset) # take all dataset in case of opt-125m
    sampled_indices = random.sample(range(len(dataset)), nsamples)

    if tokenizer.pad_token is None:
        # Llama 3.2 1B has no tokenizer.pad_token
        tokenizer.pad_token = tokenizer.eos_token

    trainloader = []
    for idx in sampled_indices:
        x0 = dataset[idx]["sent_more"]
        x1 = dataset[idx]["sent_less"]
        repeats = 8
        x0 = " ".join([x0] * repeats)
        x1 = " ".join([x1] * repeats)
        enc_x0 = tokenizer(x0, padding='max_length', max_length=seqlen, return_tensors="pt")
        enc_x1 = tokenizer(x1, padding='max_length', max_length=seqlen, return_tensors="pt")
        inp = torch.cat((enc_x0.input_ids, enc_x1.input_ids), dim=0)
        trainloader.append((inp,))

    print(f"get_crows_pairs: nsamples={len(trainloader)}")
    return trainloader

def get_crows_stories(nsamples, seed, seqlen, tokenizer, train, model=None):
    if not train:
        raise ValueError

    print(f"get_crows_stories: seqlen={seqlen}")
    random.seed(seed)
    split = "train" # there is no "test" split in crows-pairs-stories
    dataset = load_dataset("iproskurina/crows-pairs-stories", split=split, trust_remote_code=True)
    if model is not None and 'opt-125m' in model.lower():
        nsamples = len(dataset) # take all dataset in case of opt-125m
    sampled_indices = random.sample(range(len(dataset)), nsamples)

    if tokenizer.pad_token is None:
        # Llama 3.2 1B has no tokenizer.pad_token
        tokenizer.pad_token = tokenizer.eos_token

    trainloader = []
    for idx in sampled_indices:
        x0 = dataset[idx]["sent_more_story"]
        x1 = dataset[idx]["story_less"]
        repeats = 8
        x0 = " ".join([x0] * repeats)
        x1 = " ".join([x1] * repeats)
        enc_x0 = tokenizer(x0, padding='max_length', max_length=seqlen, return_tensors="pt")
        enc_x1 = tokenizer(x1, padding='max_length', max_length=seqlen, return_tensors="pt")
        inp = torch.cat((enc_x0.input_ids, enc_x1.input_ids), dim=0)
        trainloader.append((inp,))

    print(f"get_crows_stories: nsamples={len(trainloader)}")
    return trainloader

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', train=True
):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if isinstance(tokenizer, LlamaTokenizer) and 'ptb' in name:
        tokenizer.tokens_trie.data = {}
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, train)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer, train)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, train)
    elif 'crows_pairs' in name:
        return get_crows_pairs(nsamples, seed, seqlen, tokenizer, train, model)
    elif 'crows_stories' in name:
        return get_crows_stories(nsamples, seed, seqlen, tokenizer, train, model)
    else: # custom dataset
        print(f"Custom dataset load from {name}")
        datas = torch.load(name)
        ids_shuffle = list(range(len(datas)))
        random.shuffle(ids_shuffle)
        return [tuple(datas[idx].unsqueeze(0)) for idx in ids_shuffle[:nsamples]]
