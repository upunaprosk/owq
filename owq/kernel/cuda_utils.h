#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <assert.h>

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline half2 pair2pack(half a, half b){
  return __halves2half2(a, b);
}
__device__ inline nv_bfloat162 pair2pack(nv_bfloat16 a, nv_bfloat16 b){
  return __halves2bfloat162(a, b);
}

template<typename T>
__device__ inline T int2T(int a);

template<>
__device__ inline half int2T<half>(int a){
  return __int2half_rn(a);
}
template<>
__device__ inline nv_bfloat16 int2T<nv_bfloat16>(int a){
  return __int2bfloat16_rn(a);
}

__device__ inline half2 TtoT2(half a){
  return __half2half2(a);
}
__device__ inline nv_bfloat162 TtoT2(nv_bfloat16 a){
  return __bfloat162bfloat162(a);
}

template<typename T>
__device__ inline T float2T(float a);

template<>
__device__ inline half float2T<half>(float a){
  return __float2half(a);
}

template<>
__device__ inline nv_bfloat16 float2T<nv_bfloat16>(float a){
  return __float2bfloat16(a);
}

__device__ inline float T2float(half a){
  return __half2float(a);
}
__device__ inline float T2float(nv_bfloat16 a){
  return __bfloat162float(a);
}

template<typename T>
__device__ inline T getzero();

template<>
__device__ inline half getzero<half>(){
  return __ushort_as_half((unsigned short)0x0000U);
}
template<>
__device__ inline nv_bfloat16 getzero<nv_bfloat16>(){
  return __ushort_as_bfloat16((unsigned short)0x0000U);
}

template<typename T>
__device__ inline T getone();

template<>
__device__ inline half getone<half>(){
  return __ushort_as_half((unsigned short)0x3C00U);
}
template<>
__device__ inline nv_bfloat16 getone<nv_bfloat16>(){
  return __ushort_as_bfloat16((unsigned short)0x3F80U);
}

template<typename T>
__device__ inline T hneg(T a){
  return __hneg(a);
}

template<typename T>
__device__ inline T hadd(T a, T b){
  return __hadd(a, b);
}

template<typename T>
__device__ inline T hsub(T a, T b){
  return __hsub(a, b);
}

template<typename T>
__device__ inline T hmul(T a, T b){
  return __hmul(a, b);
}

template<typename T>
__device__ inline T hdiv(T a, T b){
  return __hdiv(a, b);
}

/*
template<typename T>
__device__ inline T hfma(T a, T b, T c){
  return __hfma(a, b, c);
}

template<typename T>
__device__ inline T hfma2(T a, T b, T c){
  return __hfma2(a, b, c);
}
*/
__device__ inline half hfma(half a, half b, half c){
  return __hfma(a, b, c);
}

__device__ inline nv_bfloat16 hfma(nv_bfloat16 a, nv_bfloat16 b, nv_bfloat16 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);  // Use hardware instruction on Ampere+
#else
    // Emulate FMA using float conversions for pre-Ampere GPUs
    float af = __bfloat162float(a);
    float bf = __bfloat162float(b);
    float cf = __bfloat162float(c);

    nv_bfloat16 result = __float2bfloat16_rn(af * bf + cf);
    return result;
#endif
}

__device__ inline half2 hfma2(half2 a, half2 b, half2 c){
  return __hfma2(a, b, c);
}

__device__ inline nv_bfloat162 hfma2(nv_bfloat162 a, nv_bfloat162 b, nv_bfloat162 c) {
#if __CUDA_ARCH__ >= 800
    return __hfma2(a, b, c);  // Use hardware instruction on Ampere+
#else
    // Emulate FMA using float conversions for pre-Ampere GPUs
    float a_low = __bfloat162float(a.x);
    float b_low = __bfloat162float(b.x);
    float c_low = __bfloat162float(c.x);

    float a_high = __bfloat162float(a.y);
    float b_high = __bfloat162float(b.y);
    float c_high = __bfloat162float(c.y);

    nv_bfloat162 result;
    result.x = __float2bfloat16_rn(a_low * b_low + c_low);
    result.y = __float2bfloat16_rn(a_high * b_high + c_high);

    return result;
#endif
}
