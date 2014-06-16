#include <stdio.h>
#include <stdarg.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "acc_gpu_data_struct.h"

#define NUM_THREADS_OF_PACK_UNPACK 128

template<int isPack>
static __global__
void _ACC_gpu_pack_data_kernel(void *dst, void *src, int dim, int total_elmnts, size_t type_size, int* info){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int totalId = blockDim.x * gridDim.x;
  int *info_lower = info;
  int *info_length = info + dim;
  int *info_dim_acc = info_length + dim;
  
  for(int t = tid; t < total_elmnts; t += totalId){
    int offset_elmnts = 0;
    int tmp = t;
    for(int i=dim-1;i>=0;i--){
      int idx = tmp % info_length[i] + info_lower[i];
      tmp /= info_length[i];
      offset_elmnts += info_dim_acc[i] * idx;
    }

    //copy
    size_t offset = offset_elmnts * type_size;
    for(int i=0;i<type_size;i++){
      if(isPack){
	*((char*)dst + t * type_size + i) = *((char *)src + offset + i);
      }else{
	*((char*)dst + offset + i) = *((char *)src + t * type_size + i);
      }
    }
  }
}

void _ACC_gpu_pack_data(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int num_blocks = (total_elmnts - 1)/ (NUM_THREADS_OF_PACK_UNPACK) + 1;
  _ACC_gpu_pack_data_kernel<1><<<num_blocks,NUM_THREADS_OF_PACK_UNPACK>>>(dst, src, dim, total_elmnts, type_size, info);
}

void _ACC_gpu_unpack_data(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int num_blocks = (total_elmnts - 1)/ (NUM_THREADS_OF_PACK_UNPACK) + 1;
  _ACC_gpu_pack_data_kernel<0><<<num_blocks,NUM_THREADS_OF_PACK_UNPACK>>>(dst, src, dim, total_elmnts, type_size, info);
}

template<typename T, int isPack> static
void pack_data(T *dst, T *src, int dim, int total_elmnts, int* info){
  int *low = info;
  int *len = info + dim;
  int *acc = len + dim;

  for(int t = 0; t < total_elmnts; t++){
    int offset_elmnts = 0;
    int tmp = t;
    for(int i=dim-1;i>=0;i--){
      int idx = tmp % len[i] + low[i];
      tmp /= len[i];
      offset_elmnts += acc[i] * idx;
    }

    //copy
    if(isPack){
      dst[t] = src[offset_elmnts];
    }else{
      dst[offset_elmnts] = src[t];
    }
  }
}

template<int isPack> static
void pack_data(char *dst, char *src, int dim, int total_elmnts, int* info, size_t type_size){
  int *low = info;
  int *len = info + dim;
  int *acc = len + dim;
  for(int t = 0; t < total_elmnts; t++){
    int offset_elmnts = 0;
    int tmp = t;
    for(int i=dim-1;i>=0;i--){
      int idx = tmp % len[i] + low[i];
      tmp /= len[i];
      offset_elmnts += acc[i] * idx;
    }

    //copy
    for(int i=0;i<type_size;i++){
      if(isPack){
	*((char*)dst + t * type_size + i) = *((char *)src + offset_elmnts * type_size + i);
      }else{
	*((char*)dst + offset_elmnts * type_size + i) = *((char *)src + t * type_size + i);
      }
    }
  }
}

void _ACC_gpu_pack_data_host(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  switch(type_size){
  case 1:
    pack_data<char, 1>((char*)dst, (char*)src, dim, total_elmnts, info);
    return;
  case 2:
    pack_data<short, 1>((short*)dst, (short*)src, dim, total_elmnts, info);
    return;
  case 4:
    pack_data<int, 1>((int*)dst, (int*)src, dim, total_elmnts, info);
    return;
  case 8:
    pack_data<long long, 1>((long long*)dst, (long long *)src, dim, total_elmnts, info);
    return;
  default:
    pack_data<1>((char *)dst, (char *)src, dim, total_elmnts, info, type_size);
  }
}


void _ACC_gpu_unpack_data_host(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  switch(type_size){
  case 1:
    pack_data<char, 0>((char*)dst, (char*)src, dim, total_elmnts, info);
    return;
  case 2:
    pack_data<short, 0>((short*)dst, (short*)src, dim, total_elmnts, info);
    return;
  case 4:
    pack_data<int, 0>((int*)dst, (int*)src, dim, total_elmnts, info);
    return;
  case 8:
    pack_data<long long, 0>((long long*)dst, (long long *)src, dim, total_elmnts, info);
    return;
  default:
    pack_data<0>((char*)dst, (char*)src, dim, total_elmnts, info, type_size);
  }
}

template <typename T, int isPack>
__global__ static
void pack_vector_kernel(T * __restrict__ dst, const T * __restrict__ src, int count, int blocklength, long stride)
{
  long i_init = blockIdx.y * blockDim.y + threadIdx.y;
  long i_step = gridDim.y * blockDim.y;
  int j_init = blockIdx.x * blockDim.x + threadIdx.x;
  int j_step = gridDim.x * blockDim.x;

  for(int i = i_init; i < count; i += i_step)
    for(int j = j_init; j < blocklength; j += j_step)
      if(isPack){
	dst[i * blocklength + j] = src[i * stride + j];
      }else{
	dst[i * stride + j] = src[i * blocklength + j];
      }
}

void _ACC_gpu_pack_vector(void *dst, void *src, int count, int blocklength, int stride, size_t typesize, int asyncId)
{
  const int numThreads = 128; //must be 2^n
  cudaStream_t st = _ACC_gpu_get_stream(asyncId);

  int bx = 1, by;
  int tx = 1, ty;
  int tmp = blocklength;
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridDim(bx,by);
  dim3 blockDim(tx, ty);

  //printf("blocklen=%d, count=%d, grid(%d,%d), block(%d,%d)\n", blocklength_c, count, bx,by,tx,ty);
  switch(typesize){
  case 1:
    pack_vector_kernel<char, 1><<<gridDim, blockDim>>>((char *)dst, (char *)src, count, blocklength, stride);
    break;
  case 2:
    pack_vector_kernel<short, 1><<<gridDim, blockDim>>>((short *)dst, (short *)src, count, blocklength, stride);
    break;
  case 4:
    pack_vector_kernel<int, 1><<<gridDim, blockDim>>>((int *)dst, (int *)src, count, blocklength, stride);
    break;
  case 8:
    pack_vector_kernel<long long, 1><<<gridDim, blockDim>>>((long long *)dst, (long long *)src, count, blocklength, stride);
    break;
  default:
    pack_vector_kernel<char, 1><<<gridDim, blockDim>>>((char *)dst, (char *)src, count, blocklength * typesize, stride * typesize);
  }
}

void _ACC_gpu_unpack_vector(void *dst, void *src, int count, int blocklength, int stride, size_t typesize, int asyncId)
{
  const int numThreads = 128; //must be 2^n
  cudaStream_t st = _ACC_gpu_get_stream(asyncId);

  int bx = 1, by;
  int tx = 1, ty;
  int tmp = blocklength;
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridDim(bx,by);
  dim3 blockDim(tx, ty);

  //printf("blocklen=%d, count=%d, grid(%d,%d), block(%d,%d)\n", blocklength, count, bx,by,tx,ty);
  switch(typesize){
  case 1:
    pack_vector_kernel<char, 0><<<gridDim, blockDim>>>((char *)dst, (char *)src, count, blocklength, stride);
    break;
  case 2:
    pack_vector_kernel<short, 0><<<gridDim, blockDim>>>((short *)dst, (short *)src, count, blocklength, stride);
    break;
  case 4:
    pack_vector_kernel<int, 0><<<gridDim, blockDim>>>((int *)dst, (int *)src, count, blocklength, stride);
    break;
  case 8:
    pack_vector_kernel<long long, 0><<<gridDim, blockDim>>>((long long *)dst, (long long *)src, count, blocklength, stride);
    break;
  default:
    pack_vector_kernel<char, 0><<<gridDim, blockDim>>>((char *)dst, (char *)src, count, blocklength * typesize, stride * typesize);
  }
}

template<typename T, int isPack> static
void pack_vector(T *dst, T *src, int count, int blocklength, int stride)
{
  for(int i = 0; i < count; i++)
    for(int j = 0; j < blocklength; j++)
      if(isPack){
	dst[i * blocklength + j] = src[i * stride + j];
      }else{
	dst[i * stride + j] = src[i * blocklength + j];
      }
}

void _ACC_pack_vector(void *dst, void *src, int count, int blocklength, int stride, size_t typesize)
{
  switch(typesize){
  case 1:
    pack_vector<char, 1>((char *)dst, (char *)src, count, blocklength, stride);
    break;
  case 2:
    pack_vector<short, 1>((short *)dst, (short *)src, count, blocklength, stride);
    break;
  case 4:
    pack_vector<int, 1>((int *)dst, (int *)src, count, blocklength, stride);
    break;
  case 8:
    pack_vector<long long, 1>((long long *)dst, (long long *)src, count, blocklength, stride);
    break;
  default:
    pack_vector<char, 1>((char *)dst, (char *)src, count, blocklength * typesize, stride * typesize);
  }
}

void _ACC_unpack_vector(void *dst, void *src, int count, int blocklength, int stride, size_t typesize)
{
  switch(typesize){
  case 1:
    pack_vector<char, 0>((char *)dst, (char *)src, count, blocklength, stride);
    break;
  case 2:
    pack_vector<short, 0>((short *)dst, (short *)src, count, blocklength, stride);
    break;
  case 4:
    pack_vector<int, 0>((int *)dst, (int *)src, count, blocklength, stride);
    break;
  case 8:
    pack_vector<long long, 0>((long long *)dst, (long long *)src, count, blocklength, stride);
    break;
  default:
    pack_vector<char, 0>((char *)dst, (char *)src, count, blocklength * typesize, stride * typesize);
  }
}
