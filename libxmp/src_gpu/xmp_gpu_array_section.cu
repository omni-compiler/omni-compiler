/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_gpu_internal.h"

#define _XMP_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _XMP_M_FLOORi(a_, b_) ((a_) / (b_))
#define _XMP_M_COUNTi(a_, b_) ((b_) - (a_) + 1)
#define _XMP_M_COUNT_TRIPLETi(l_, u_, s_) (_XMP_M_FLOORi(((u_) - (l_)), s_) + 1)

static void _XMP_gpu_config_block(unsigned long long total_elmts, int *block_x, int *block_y, int *block_z) {
  unsigned long long num_threads = 16 * 16; // DEFAULT value

  if (num_threads > _XMP_gpu_max_thread) {
    _XMP_fatal("too many threads are requested for GPU");
  }

  if (num_threads >= total_elmts) {
    *block_x = 1;
    *block_y = 1;
    *block_z = 1;
    return;
  }

  total_elmts = _XMP_M_CEILi(total_elmts, num_threads);

  if (total_elmts > _XMP_gpu_max_block_dim_x) {
    *block_x = _XMP_gpu_max_block_dim_x;

    total_elmts = _XMP_M_CEILi(total_elmts, _XMP_gpu_max_block_dim_x);
    if (total_elmts > _XMP_gpu_max_block_dim_y) {
      *block_y = _XMP_gpu_max_block_dim_y;

      total_elmts = _XMP_M_CEILi(total_elmts, _XMP_gpu_max_block_dim_y);
      if (total_elmts > _XMP_gpu_max_block_dim_z) {
        _XMP_fatal("data is too big for GPU");
      } else {
        *block_z = total_elmts;
      }
    } else {
      *block_y = total_elmts;
      *block_z = 1;
    }
  } else {
    *block_x = total_elmts;
    *block_y = 1;
    *block_z = 1;
  }
}

__device__ static void _XMP_gpu_calc_thread_id(unsigned long long *index) {
  *index = threadIdx.x +
          (threadIdx.y * blockDim.x) +
          (threadIdx.z * blockDim.x * blockDim.y) +
         ((blockIdx.x +
          (blockIdx.y * gridDim.x) +
          (blockIdx.z * gridDim.x * gridDim.y)) * (blockDim.x * blockDim.y * blockDim.z));
}

__global__ static void _XMP_gpu_pack_array_kernel(char *gpu_buffer, char *array_addr, size_t type_size, unsigned long long total_elmts,
                                                  int array_dim, int *lower, int *upper, int *stride, unsigned long long *dim_acc) {
  unsigned long long tid;
  _XMP_gpu_calc_thread_id(&tid);

  if (tid < total_elmts) {
    __syncthreads();
  }
}

void _XMP_gpu_pack_array(void *host_buffer, void *array_addr, size_t type_size, size_t alloc_size,
                         int array_dim, int *lower, int *upper, int *stride, unsigned long long *dim_acc) {
  // config block parameters
  unsigned long long total_elmts = alloc_size / type_size;
  int block_x, block_y, block_z;
  _XMP_gpu_config_block(total_elmts, &block_x, &block_y, &block_z);

  // alloc GPU buffer
  void *gpu_buffer;
  _XMP_gpu_alloc(&gpu_buffer, alloc_size);

  // init shadow data on GPU
  int *gpu_lower, *gpu_upper, *gpu_stride;
  size_t lus_size = sizeof(int) * array_dim;
  _XMP_gpu_alloc((void **)&gpu_lower, lus_size);
  _XMP_gpu_alloc((void **)&gpu_upper, lus_size);
  _XMP_gpu_alloc((void **)&gpu_stride, lus_size);
  cudaMemcpy(gpu_lower, lower, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_upper, upper, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_stride, stride, lus_size, cudaMemcpyHostToDevice);

  unsigned long long *gpu_dim_acc;
  size_t da_size = sizeof(unsigned long long) * array_dim;
  _XMP_gpu_alloc((void **)&gpu_dim_acc, da_size);
  cudaMemcpy(gpu_dim_acc, dim_acc, da_size, cudaMemcpyHostToDevice);

  // pack shadow on GPU
  _XMP_gpu_pack_array_kernel<<<dim3(block_x, block_y, block_z), dim3(16, 16, 1)>>>((char *)gpu_buffer, (char *)array_addr, type_size, total_elmts,
                                                                                   array_dim, lower, upper, stride, dim_acc);
  cudaThreadSynchronize();

  // copy shadow buffer to host
  cudaMemcpy(host_buffer, gpu_buffer, alloc_size, cudaMemcpyDeviceToHost);

  // free GPU buffer
  _XMP_gpu_free(gpu_buffer);
}
