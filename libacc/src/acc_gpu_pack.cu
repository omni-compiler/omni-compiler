#include <stdio.h>
#include <stdarg.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "acc_gpu_data_struct.h"

#define NUM_THREADS_OF_PACK_UNPACK 128


__global__
void _ACC_gpu_pack_data_kernel(void *dst, void *src, int dim, int total_elmnts, size_t type_size, int* info){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int totalId = blockDim.x * gridDim.x;
  int t;
  int *info_lower = info;
  int *info_length = info + dim;
  int *info_dim_acc = info_length + dim;
  
  for(t = tid; t < total_elmnts; t += totalId){
    int offset_elmnts = 0;
    int tmp = t;
    // for row-major
    int i;
    for(i=dim-1;i>=0;i--){
      int idx = tmp % info_length[i] + info_lower[i];
      tmp /= info_length[i];
      offset_elmnts += info_dim_acc[i] * idx;
    }

    //copy
    //printf("kernel_offset: %d\n", offset_elmnts);
    size_t offset = offset_elmnts * type_size;
    for(i=0;i<type_size;i++){
      *((char*)dst + t * type_size + i) = *((char *)src + offset + i);
    }
  }
}

__global__
void _ACC_gpu_unpack_data_kernel(void *dst, void *src, int dim, int total_elmnts, size_t type_size, int* info){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int totalId = blockDim.x * gridDim.x;
  int t;
  int *info_lower = info;
  int *info_length = info + dim;
  int *info_dim_acc = info_length + dim;
  
  for(t = tid; t < total_elmnts; t += totalId){
    int offset_elmnts = 0;
    int tmp = t;
    // for row-major
    int i;
    for(i=dim-1;i>=0;i--){
      int idx = tmp % info_length[i] + info_lower[i];
      tmp /= info_length[i];
      offset_elmnts += info_dim_acc[i] * idx;
    }

    //copy
    //printf("kernel_offset: %d\n", offset_elmnts);
    for(i=0;i<type_size;i++){
      *((char*)dst + offset_elmnts * type_size + i) = *((char *)src + t * type_size + i);
    }
  }
}

void _ACC_gpu_pack_data(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int num_blocks = (total_elmnts - 1)/ (NUM_THREADS_OF_PACK_UNPACK) + 1;
  _ACC_gpu_pack_data_kernel<<<num_blocks,NUM_THREADS_OF_PACK_UNPACK>>>(dst, src, dim, total_elmnts, type_size, info);
}

void _ACC_gpu_unpack_data(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int num_blocks = (total_elmnts - 1)/ (NUM_THREADS_OF_PACK_UNPACK) + 1;
  _ACC_gpu_unpack_data_kernel<<<num_blocks,NUM_THREADS_OF_PACK_UNPACK>>>(dst, src, dim, total_elmnts, type_size, info);
}

template<typename T>
void _ACC_gpu_pack_data_host__(T *dst, T *src, int dim, int total_elmnts, int* info){
  int *low = info;
  int *len = info + dim;
  int *acc = len + dim;

  if(dim == 2){
    /*
    int i,j;
    int i_upper = low[0] + len[0];
    int j_upper = low[1] + len[1];
    int acc0 = acc[0];
    for(i = low[0]; i < i_upper; i++){
      for(j = low[1]; j < j_upper; j++){
	*dst++ = src[i * acc0 + j];
      }
    }
    */
    //    return;
    src += low[0]*acc[0] + low[1];
    int stride = acc[0] - acc[1] * len[1];
    int len0 = len[0]; int  len1 = len[1];
    for(int i = 0; i < len0; i++){
      for(int j = 0; j < len1; j++){
	*dst++ = *src++;
	//      memcpy(dst, src, sizeof(T)*len1);
	//      dst+=len1;src+=len1;
      }
      src+=stride;
    }

  }else{
    for(int t = 0; t < total_elmnts; t++){
      int offset_elmnts = 0;
      int tmp = t;
      // for row-major
      int i;
      for(i=dim-1;i>=0;i--){
	int idx = tmp % len[i] + low[i];
	tmp /= len[i];
	offset_elmnts += acc[i] * idx;
      }

      //copy
      //dst[offset_elmnts] = src[t];
      dst[t] = src[offset_elmnts];
    }
  }
}

void _ACC_gpu_pack_data_host(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int *info_lower = info;
  int *info_length = info + dim;
  int *info_dim_acc = info_length + dim;
  int t;

  switch(type_size){
  case 1:
    _ACC_gpu_pack_data_host__<char>((char*)dst, (char*)src, dim, total_elmnts, info);
    return;
  case 4:
    _ACC_gpu_pack_data_host__<int>((int*)dst, (int*)src, dim, total_elmnts, info);
    return;
  case 8:
    _ACC_gpu_pack_data_host__<long long>((long long*)dst, (long long *)src, dim, total_elmnts, info);
    return;
  default:
    {
      for(t = 0; t < total_elmnts; t++){
	int offset_elmnts = 0;
	int tmp = t;
	// for row-major
	int i;
	for(i=dim-1;i>=0;i--){
	  int idx = tmp % info_length[i] + info_lower[i];
	  tmp /= info_length[i];
	  offset_elmnts += info_dim_acc[i] * idx;
	}

	//copy
	for(i=0;i<type_size;i++){
	  *((char*)dst + t * type_size + i) = *((char *)src + offset_elmnts * type_size + i);
	}
      }      
    }
  }
}


template<typename T>
void _ACC_gpu_unpack_data_host__(T *dst, T *src, int dim, int total_elmnts, int* info){
  int *low = info;
  int *len = info + dim;
  int *acc = len + dim;

  if(dim == 2){
    /*
    int i,j;
    int i_upper = low[0] + len[0];
    int j_upper = low[1] + len[1];
    int acc0 = acc[0];
    for(i = low[0]; i < i_upper; i++){
      for(j = low[1]; j < j_upper; j++){
	dst[i * acc0 + j] = *src++;
      }
    }
    */
    //    return;
    dst += low[0]*acc[0] + low[1];
    int stride = acc[0] - acc[1] * len[1];
    int len0 = len[0], len1 = len[1];
    for(int i = 0; i < len0; i++){
      for(int j = 0; j < len1; j++){
      	*dst++ = *src++;
      }
      dst+=stride;
    }

  }else{
    for(int t = 0; t < total_elmnts; t++){
      int offset_elmnts = 0;
      int tmp = t;
      // for row-major
      int i;
      for(i=dim-1;i>=0;i--){
	int idx = tmp % len[i] + low[i];
	tmp /= len[i];
	offset_elmnts += acc[i] * idx;
      }

      //copy
      dst[offset_elmnts] = src[t];
    }
  }
}

void _ACC_gpu_unpack_data_host(void *dst, void *src, int dim, int total_elmnts, int type_size, int* info){
  int *info_lower = info;
  int *info_length = info + dim;
  int *info_dim_acc = info_length + dim;
  int t;

  switch(type_size){
  case 1:
    _ACC_gpu_unpack_data_host__<char>((char*)dst, (char*)src, dim, total_elmnts, info);
    return;
  case 4:
    _ACC_gpu_unpack_data_host__<int>((int*)dst, (int*)src, dim, total_elmnts, info);
    return;
  case 8:
    _ACC_gpu_unpack_data_host__<long long>((long long*)dst, (long long *)src, dim, total_elmnts, info);
    return;
  default:
    {
      for(t = 0; t < total_elmnts; t++){
	int offset_elmnts = 0;
	int tmp = t;
	// for row-major
	int i;
	for(i=dim-1;i>=0;i--){
	  int idx = tmp % info_length[i] + info_lower[i];
	  tmp /= info_length[i];
	  offset_elmnts += info_dim_acc[i] * idx;
	}

	//copy
	for(i=0;i<type_size;i++){
	  *((char*)dst + offset_elmnts * type_size + i) = *((char *)src + t * type_size + i);
	}
      }
    }
  }
}
