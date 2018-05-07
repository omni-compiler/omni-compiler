#ifndef _ACC_GPU_REDUCTION
#define _ACC_GPU_REDUCTION

#include <limits.h>
#include <float.h>
#include <stdio.h>
#include "acc_gpu_atomic.hpp"

#define _ACC_REDUCTION_PLUS 0
#define _ACC_REDUCTION_MUL 1
#define _ACC_REDUCTION_MAX 2
#define _ACC_REDUCTION_MIN 3
#define _ACC_REDUCTION_BITAND 4
#define _ACC_REDUCTION_BITOR 5
#define _ACC_REDUCTION_BITXOR 6
#define _ACC_REDUCTION_LOGAND 7
#define _ACC_REDUCTION_LOGOR 8

__device__ static inline
void _ACC_gpu_init_reduction_var(int *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0; return;
  case _ACC_REDUCTION_MUL: *var = 1; return;
  case _ACC_REDUCTION_MAX: *var = INT_MIN; return;
  case _ACC_REDUCTION_MIN: *var = INT_MAX; return;
  case _ACC_REDUCTION_BITAND: *var = ~0; return;
  case _ACC_REDUCTION_BITOR: *var = 0; return;
  case _ACC_REDUCTION_BITXOR: *var = 0; return;
  case _ACC_REDUCTION_LOGAND: *var = 1; return;
  case _ACC_REDUCTION_LOGOR: *var = 0; return;
  }
}

__device__ static inline
void _ACC_gpu_init_reduction_var(float *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0.0f; return;
  case _ACC_REDUCTION_MUL: *var = 1.0f; return;
  case _ACC_REDUCTION_MAX: *var = -FLT_MAX; return;
  case _ACC_REDUCTION_MIN: *var = FLT_MAX; return;
  }
}

__device__ static inline
void _ACC_gpu_init_reduction_var(double *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0.0; return;
  case _ACC_REDUCTION_MUL: *var = 1.0; return;
  case _ACC_REDUCTION_MAX: *var = -DBL_MAX; return;
  case _ACC_REDUCTION_MIN: *var = DBL_MAX; return;
  }
}

template<typename T>
__device__ static inline
void _ACC_gpu_init_reduction_var_single(T *var, int kind){
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    _ACC_gpu_init_reduction_var(var, kind);
  }
}

__device__ static inline
int op(int a, int b, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: return a + b;
  case _ACC_REDUCTION_MUL: return a * b;
  case _ACC_REDUCTION_MAX: return (a > b)? a : b;
  case _ACC_REDUCTION_MIN: return (a < b)? a : b;
  case _ACC_REDUCTION_BITAND: return a & b;
  case _ACC_REDUCTION_BITOR: return a | b;
  case _ACC_REDUCTION_BITXOR: return a ^ b;
  case _ACC_REDUCTION_LOGAND: return a && b;
  case _ACC_REDUCTION_LOGOR: return a || b;
  default: return a;
  }
}

template<typename T>
__device__ static inline
T op(T a, T b, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: return a + b;
  case _ACC_REDUCTION_MUL: return a * b;
  case _ACC_REDUCTION_MAX: return (a > b)? a : b;
  case _ACC_REDUCTION_MIN: return (a < b)? a : b;
  default: return a;
  }
}


#if __CUDA_ARCH__ >= 300
#include <cuda.h>
#if CUDA_VERSION <= 6000
__device__ static inline
double __shfl_xor(double var, int laneMask, int width=warpSize)
{
  int hi, lo;
  asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var) );
  hi = __shfl_xor( hi, laneMask, width );
  lo = __shfl_xor( lo, laneMask, width );
  return __hiloint2double( hi, lo );
}
#endif

/**
   Reduce input value among lanes and store it with addtional value to output location of lane 0 using shuffle instruction
   *
   * @param[out] output   pointer to output
   * @param[in]  input    input value
   * @param[in]  kind     operator kind
   * @param[in]  addition additional value
   */
template<typename T>
__device__ static inline
void _ACC_reduce_lanes(T *output, T input, int kind, bool do_acc)
{
  T v = input;
  unsigned int laneId = threadIdx.x & (32 - 1);

  v = op(v, __shfl_xor(v, 16), kind);
  v = op(v, __shfl_xor(v, 8), kind);
  v = op(v, __shfl_xor(v, 4), kind);
  v = op(v, __shfl_xor(v, 2), kind);
  v = op(v, __shfl_xor(v, 1), kind);

  if(laneId == 0){
    *output = do_acc? op(*output, v, kind) : v;
  }
}

/**
   Reduce input value among threads and store it to target pointer of thread 0 using shuffle instruction
   *
   * @param[in,out] target   target pointer
   * @param[in]     input    input value
   * @param[in]     kind     operator kind
   * @param[in]     do_acc   do accumulate flag
   */
template<typename T>
__device__ static inline
void _ACC_reduce_threads(T *target, T input, int kind, bool do_acc){
  __shared__ T tmp[32];

  unsigned int warpId = threadIdx.x >> 5;

  _ACC_reduce_lanes(&tmp[warpId], input, kind, false);

  __syncthreads();

  if(warpId == 0){
    unsigned int nwarps = blockDim.x >> 5 ;
    T v;
    if(threadIdx.x < nwarps){
      v = tmp[threadIdx.x];
    }else{
      _ACC_gpu_init_reduction_var(&v, kind);
    }
    _ACC_reduce_lanes(target, v, kind, do_acc);
  }
  __syncthreads();
}

#else //__CUDA_ARCH__ < 300

template<typename T>
__device__ static inline
void warpReduce(volatile T sdata[64], int kind){
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 32], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 16], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 8],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 4],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 2],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 1],  kind);
}

template<typename T>
__device__ static inline
void warpReduce32(volatile T sdata[64], int kind){
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 16], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 8],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 4],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 2],  kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 1],  kind);
}

/**
   Reduce input value among threads and store it to target pointer of thread 0 using shared memory
   *
   * @param[in,out] target   target pointer
   * @param[in]     input    input value
   * @param[in]     kind     operator kind
   * @param[in]     do_acc   do accumulate flag
   */
template<typename T>
__device__ static inline
void _ACC_reduce_threads(T *target, T input, int kind, bool do_acc){
  __shared__ T tmp[64];

  if(blockDim.x >= 64){
  if(threadIdx.x < 64){
    tmp[threadIdx.x] = input;
  }
  __syncthreads();

  unsigned int div64_q = threadIdx.x >> 6;
  unsigned int div64_m = threadIdx.x % 64;
  unsigned int max_q = ((blockDim.x - 1) >> 6) + 1;
  for(unsigned int i = 1; i < max_q; i++){
    if(i == div64_q){
      tmp[div64_m] = op(tmp[div64_m], input, kind);
    }
    __syncthreads();
  }

  if(threadIdx.x < 32) warpReduce(tmp, kind);
  }else{
    tmp[threadIdx.x] = input;
    warpReduce32(tmp, kind);
  }

  if(threadIdx.x == 0){
    *target = do_acc? op(*target, tmp[0], kind) : tmp[0];
  }
  __syncthreads(); //sync for next reduction among threads
}
#endif

/**
   Atomic accumulate input value to target variable for int type
   *
   * @param[in,out] target target pointer
   * @param[in]     val    value
   * @param[in]     kind   operator kind
   */
__device__ static inline
void _ACC_atomic_accumulate(int* target, int val, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS:
    atomicAdd(target, val);break;
  case _ACC_REDUCTION_MAX:
    atomicMax(target, val);break;
  case _ACC_REDUCTION_MIN:
    atomicMin(target, val);break;
  case _ACC_REDUCTION_BITAND:
    atomicAnd(target, val);break;
  case _ACC_REDUCTION_BITOR:
    atomicOr(target, val);break;
  case _ACC_REDUCTION_BITXOR:
    atomicXor(target, val);break;
  case _ACC_REDUCTION_LOGOR:
    if(val) atomicOr(target, ~0);
    break;
  case _ACC_REDUCTION_LOGAND:
    if(! val) atomicAnd(target, 0);
    break;
  }
}

/**
   Atomic accumulate input value to target variable for float type
   *
   * @param[in,out] target target pointer
   * @param[in]     val    value
   * @param[in]     kind   operator kind
   */
__device__ static inline
void _ACC_atomic_accumulate(float *target, float val, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS:
    atomicAdd(target, val);break;
  case _ACC_REDUCTION_MAX:
    atomicMax(target, val);break;
  case _ACC_REDUCTION_MIN:
    atomicMin(target, val);break;
  }
}

/**
   Reduce buffer data using threads and accumulate it to output location
   *
   * @param[in,out] result         result address
   * @param[in]     kind           operator kind
   * @param[in]     buf            buffer
   * @param[in]     element_offset (element_offset * num_elements) is buffer offset
   * @param[in]     num_elements   number of elements
   */
template<typename T>
__device__ static inline
void _ACC_gpu_reduction_block(T* result, int kind, void* buf, size_t element_offset, int num_elements){
  T *data = (T*)((char*)buf + (element_offset * num_elements));

  T part_result;
  _ACC_gpu_init_reduction_var(&part_result, kind);

  for(int idx = threadIdx.x; idx < num_elements; idx += blockDim.x){
    part_result = op(part_result, data[idx], kind);
  }

  _ACC_reduce_threads(result, part_result, kind, true);

}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_block(T* result, int kind, void* tmp, size_t offsetElementSize){
  _ACC_gpu_reduction_block(result, kind, tmp, offsetElementSize, gridDim.x);
}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_singleblock(T* result, T resultInBlock, int kind){
  if(threadIdx.x == 0){
    *result = op(*result, resultInBlock, kind);
  }
}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_tmp(T resultInBlock, void *tmp, size_t offsetElementSize){
  if(threadIdx.x==0){
    void *tmpAddr =  (char*)tmp + (gridDim.x * offsetElementSize);
    ((T*)tmpAddr)[blockIdx.x] = resultInBlock;
  }
  __syncthreads();//is need?
}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_bt(T *resultInGrid, int kind, T resultInThread){
  T resultInBlock;
  _ACC_reduce_threads(&resultInBlock, resultInThread, kind, false);
  if(threadIdx.x == 0){
    _ACC_atomic_accumulate(resultInGrid, resultInBlock, kind);
  }
}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_b(T *resultInGrid, int kind, T resultInBlock){
  if(threadIdx.x == 0){
    _ACC_atomic_accumulate(resultInGrid, resultInBlock, kind);
  }
}

template<typename T>
__device__ static inline
void _ACC_gpu_reduction_t(T *resultInBlock, int kind, T resultInThread){
  _ACC_reduce_threads(resultInBlock, resultInThread, kind, true);
}

#endif //_ACC_GPU_REDUCTION
