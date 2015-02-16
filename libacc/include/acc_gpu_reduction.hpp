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

static __device__
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

static __device__
void _ACC_gpu_init_reduction_var(float *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0.0f; return;
  case _ACC_REDUCTION_MUL: *var = 1.0f; return;
  case _ACC_REDUCTION_MAX: *var = FLT_MIN; return;
  case _ACC_REDUCTION_MIN: *var = FLT_MAX; return;
  }
}
static __device__
void _ACC_gpu_init_reduction_var(double *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0.0; return;
  case _ACC_REDUCTION_MUL: *var = 1.0; return;
  case _ACC_REDUCTION_MAX: *var = DBL_MIN; return;
  case _ACC_REDUCTION_MIN: *var = DBL_MAX; return;
  }
}

template<typename T>
static __device__
void _ACC_gpu_init_reduction_var_single(T *var, int kind){
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    _ACC_gpu_init_reduction_var(var, kind);
  }
}

static __device__
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
static __device__
T op(T a, T b, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: return a + b;
  case _ACC_REDUCTION_MUL: return a * b;
  case _ACC_REDUCTION_MAX: return (a > b)? a : b;
  case _ACC_REDUCTION_MIN: return (a < b)? a : b;
  default: return a;
  }
}


template<typename T>
static __device__ void warpReduce(volatile T sdata[64], int kind){
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 32], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 16], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 8], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 4], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 2], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 1], kind);
}

template<typename T>
static __device__ void warpReduce32(volatile T sdata[64], int kind){
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 16], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 8], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 4], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 2], kind);
  sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x + 1], kind);
}

#if __CUDA_ARCH__ >= 300
#include <cuda.h>
#if CUDA_VERSION <= 6000
static __inline__ __device__
double __shfl_xor(double var, int laneMask, int width=warpSize)
{
  int hi, lo;
  asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var) );
  hi = __shfl_xor( hi, laneMask, width );
  lo = __shfl_xor( lo, laneMask, width );
  return __hiloint2double( hi, lo );
}
#endif
template<typename T>
static __device__
void reduceInBlock(T *resultInBlock, T resultInThread, int kind){
  __shared__ T tmp[32];

  unsigned int warpId = threadIdx.x >> 5;
  unsigned int lane = threadIdx.x & (32 - 1);

  T v = resultInThread;
  v = op(v, __shfl_xor(v, 16), kind);
  v = op(v, __shfl_xor(v, 8), kind);
  v = op(v, __shfl_xor(v, 4), kind);
  v = op(v, __shfl_xor(v, 2), kind);
  v = op(v, __shfl_xor(v, 1), kind);

  if(lane == 0){
    tmp[warpId] = v;
  }
  __syncthreads();
  
  if(threadIdx.x < 32){
    unsigned int nwarps = blockDim.x >> 5 ;
    T v;
    if(threadIdx.x < nwarps){
      v = tmp[threadIdx.x];
    }else{
      _ACC_gpu_init_reduction_var(&v, kind);
    }
    v = op(v, __shfl_xor(v, 16), kind);
    v = op(v, __shfl_xor(v, 8), kind);
    v = op(v, __shfl_xor(v, 4), kind);
    v = op(v, __shfl_xor(v, 2), kind);
    v = op(v, __shfl_xor(v, 1), kind);
    if(threadIdx.x==0){
      *resultInBlock = op(v, *resultInBlock, kind);
    }
  }
  __syncthreads();
}
#else
template<typename T>
static __device__
void reduceInBlock(T *resultInBlock, T resultInThread, int kind){
  __shared__ T tmp[64];

  if(blockDim.x >= 64){
  if(threadIdx.x < 64){
    tmp[threadIdx.x] = resultInThread;
  }
  __syncthreads();
  
  unsigned int div64_q = threadIdx.x >> 6;
  unsigned int div64_m = threadIdx.x % 64;
  unsigned int max_q = ((blockDim.x - 1) >> 6) + 1;
  for(unsigned int i = 1; i < max_q; i++){
    if(i == div64_q){
      tmp[div64_m] = op(tmp[div64_m], resultInThread, kind);
    }
    __syncthreads();
  }

  if(threadIdx.x < 32) warpReduce(tmp, kind);
  }else{
    tmp[threadIdx.x] = resultInThread;
    warpReduce32(tmp, kind);
  }

  if(threadIdx.x == 0){
    *resultInBlock = op(tmp[0], *resultInBlock, kind);
  }
  __syncthreads(); //sync for next reduction among threads
}
#endif

template<typename T>
__device__
static void reduceInGridDefault(T *resultInGrid, T resultInBlock, int kind, T *tmp, unsigned int *cnt){
  __shared__ bool isLastBlockDone;

  if(threadIdx.x == 0){
    tmp[blockIdx.x] = resultInBlock;
    __threadfence();
    
    //increase cnt after tmp is visible from all.
    unsigned int value = atomicInc(cnt, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  __syncthreads();

  if(isLastBlockDone){
    T part_result;
    _ACC_gpu_init_reduction_var(&part_result, kind);
    for(int idx = threadIdx.x; idx < gridDim.x; idx += blockDim.x){
      part_result = op(part_result, tmp[idx], kind);
    }
    T result;
    if(threadIdx.x == 0){
      _ACC_gpu_init_reduction_var(&result, kind);
    }
    reduceInBlock(&result, part_result, kind);
    if(threadIdx.x == 0){
      *resultInGrid = op(*resultInGrid, result, kind);
      *cnt = 0; //important!
    }
  }
}

__device__
static void _ACC_gpu_reduction_block(int* result, int kind, int resultInBlock){
  if(threadIdx.x == 0){
    switch(kind){
    case _ACC_REDUCTION_PLUS:
      atomicAdd(result, resultInBlock);break;
    case _ACC_REDUCTION_MAX:
      atomicMax(result, resultInBlock);break;
    case _ACC_REDUCTION_MIN:
      atomicMin(result, resultInBlock);break;
    case _ACC_REDUCTION_BITAND:
      atomicAnd(result, resultInBlock);break;
    case _ACC_REDUCTION_BITOR:
      atomicOr(result, resultInBlock);break;
    case _ACC_REDUCTION_BITXOR:
      atomicXor(result, resultInBlock);break;
    case _ACC_REDUCTION_LOGOR:
      if(resultInBlock) atomicOr(result, ~0);
      break;
    case _ACC_REDUCTION_LOGAND:
      if(! resultInBlock) atomicAnd(result, 0);
      break;
    }
  }
}

__device__
static void _ACC_gpu_reduction_block(float *resultInGrid, int kind, float resultInBlock){ 
  if(threadIdx.x == 0){
    switch(kind){
    case _ACC_REDUCTION_PLUS:
      atomicAdd(resultInGrid, resultInBlock);break;
    case _ACC_REDUCTION_MAX:
      atomicMax(resultInGrid, resultInBlock);break;
    case _ACC_REDUCTION_MIN:
      atomicMin(resultInGrid, resultInBlock);break;
    }
  }
}

__device__
static void reduceInGrid(double *resultInGrid, double resultInBlock, int kind, double *tmp, unsigned int *cnt){
  if(kind == _ACC_REDUCTION_MAX){
    if(threadIdx.x == 0) atomicMax(resultInGrid, resultInBlock);
    __syncthreads();
  }else if(kind == _ACC_REDUCTION_MIN){
    if(threadIdx.x == 0) atomicMin(resultInGrid, resultInBlock);
    __syncthreads();
  }else{
    reduceInGridDefault(resultInGrid, resultInBlock, kind, tmp, cnt);
    //    __syncthreads();
  }
}  


template<typename T>
__device__
static void _ACC_gpu_reduce_block_thread_x(T *result, T resultInThread, int kind){
  T resultInBlock;
  if(threadIdx.x == 0){
    _ACC_gpu_init_reduction_var(&resultInBlock, kind);
  }
  reduceInBlock(&resultInBlock, resultInThread, kind);
  reduceInGrid(result, resultInBlock, kind, NULL, NULL);
}

template<typename T>
__device__
static void _ACC_gpu_reduce_block_thread_x(T *result, T resultInThread, int kind, T *tmp, unsigned int *cnt){
  T resultInBlock;
  if(threadIdx.x == 0){ 
    _ACC_gpu_init_reduction_var(&resultInBlock, kind);
  }
  reduceInBlock(&resultInBlock, resultInThread, kind);
  reduceInGrid(result, resultInBlock, kind, tmp, cnt);
}

template<typename T>
__device__
static void _ACC_gpu_reduce_thread_x(T *result, T resultInThread, int kind){
  reduceInBlock(result, resultInThread, kind);
}

/////////////


//template<typename T>
static __device__
void _ACC_gpu_init_block_reduction(unsigned int *counter, void * volatile * tmp, size_t totalElementSize){
  // initial value of counter and *tmp is 0 or NULL.
  if(threadIdx.x == 0){
    if(*tmp == NULL){
      unsigned int value = atomicCAS(counter, 0, 1);
      //      printf("counter=%d(%d)\n",value,blockIdx.x);
      if(value == 0){
	//malloc
	*tmp = malloc(gridDim.x * totalElementSize);
	__threadfence();
	*counter = 0;
	//	printf("malloced (%d)\n",blockIdx.x);
      }else{
	//wait completion of malloc
	while(*tmp == NULL); //do nothing
      }
    }
  }
  __syncthreads();
}

template<typename T>
__device__
static void _ACC_gpu_reduction_thread(T *resultInBlock, T resultInThread, int kind){
  reduceInBlock(resultInBlock, resultInThread, kind);
}

__device__
static void _ACC_gpu_is_last_block(int *is_last, unsigned int *counter){
  __threadfence();
  if(threadIdx.x==0){
    unsigned int value = atomicInc(counter, gridDim.x - 1);
    *is_last = (value == (gridDim.x - 1));
  }
  __syncthreads();
}

template<typename T>
__device__
static void reduceInGridDefault_new(T *result, T *tmp, int kind, int numBlocks){
  T part_result;
  _ACC_gpu_init_reduction_var(&part_result, kind);
  for(int idx = threadIdx.x; idx < numBlocks; idx += blockDim.x){
    part_result = op(part_result, tmp[idx], kind);
  }
  T resultInGrid;
  if(threadIdx.x == 0){
    _ACC_gpu_init_reduction_var(&resultInGrid, kind);
  }
  reduceInBlock(&resultInGrid, part_result, kind);
  if(threadIdx.x == 0){
    *result = op(*result, resultInGrid, kind);
  }
}

template<typename T>
__device__
static void _ACC_gpu_reduction_block(T* result, int kind, void* tmp, size_t offsetElementSize, int numBlocks){
  void *tmpAddr = (char*)tmp + (numBlocks * offsetElementSize);
  reduceInGridDefault_new(result, (T*)tmpAddr, kind, numBlocks);
}

template<typename T>
__device__
static void _ACC_gpu_reduction_block(T* result, int kind, void* tmp, size_t offsetElementSize){
  _ACC_gpu_reduction_block(result, kind, tmp, offsetElementSize, gridDim.x);
}

template<typename T>
__device__
static void _ACC_gpu_reduction_singleblock(T* result, T resultInBlock, int kind){
  if(threadIdx.x == 0){
    *result = op(*result, resultInBlock, kind);
  }
}

//template<typename T>
__device__
static void _ACC_gpu_finalize_reduction(unsigned int *counter, void* tmp){
  *counter = 0;
}

template<typename T>
__device__
static void _ACC_gpu_reduction_tmp(T resultInBlock, void *tmp, size_t offsetElementSize){
  if(threadIdx.x==0){
    void *tmpAddr =  (char*)tmp + (gridDim.x * offsetElementSize);
    ((T*)tmpAddr)[blockIdx.x] = resultInBlock;
  }
  __syncthreads();//is need?
}

#endif //_ACC_GPU_REDUCTION
