/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_gpu_internal.h"

#define _XMP_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _XMP_M_FLOORi(a_, b_) ((a_) / (b_))
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

__global__ static void _XMP_gpu_pack_array_kernel(_XMP_gpu_array_t *desc, char *shadow_buffer, char *array_addr,
                                                  size_t type_size, unsigned long long total_elmts, int array_dim,
                                                  int *lower, int *upper, int *stride) {
  unsigned long long tid;
  _XMP_gpu_calc_thread_id(&tid);

  if (tid < total_elmts) {
    // calc array addr
    unsigned long long temp = tid;
    for (int i = 0; i < array_dim; i++) {
      int lowerI = lower[i];
      int upperI = upper[i];
      int strideI = stride[i];
      int countI = _XMP_M_COUNT_TRIPLETi(lowerI, upperI, strideI);

      // calc index
      unsigned long long indexI = lowerI + ((temp % countI) * strideI);
      temp /= countI;

      // move array addr
      array_addr += indexI * type_size * (desc[i].acc);
    }

    // calc shadow buffer
    shadow_buffer += tid * type_size;

    // memory copy
    for (int i = 0; i < type_size; i++) {
      shadow_buffer[i] = array_addr[i];
    }
  }
}

__global__ static void _XMP_gpu_unpack_array_kernel(_XMP_gpu_array_t *desc, char *array_addr, char *shadow_buffer,
                                                    size_t type_size, unsigned long long total_elmts, int array_dim,
                                                    int *lower, int *upper, int *stride) {
  unsigned long long tid;
  _XMP_gpu_calc_thread_id(&tid);

  if (tid < total_elmts) {
    // calc array addr
    unsigned long long temp = tid;
    for (int i = 0; i < array_dim; i++) {
      int lowerI = lower[i];
      int upperI = upper[i];
      int strideI = stride[i];
      int countI = _XMP_M_COUNT_TRIPLETi(lowerI, upperI, strideI);

      // calc index
      unsigned long long indexI = lowerI + ((temp % countI) * strideI);
      temp /= countI;

      // move array addr
      array_addr += indexI * type_size * (desc[i].acc);
    }

    // calc shadow buffer
    shadow_buffer += tid * type_size;

    // memory copy
    for (int i = 0; i < type_size; i++) {
      array_addr[i] = shadow_buffer[i];
    }
  }
}

extern "C"
void _XMP_gpu_pack_array(_XMP_gpu_array_t *device_desc, void *host_shadow_buffer, void *gpu_array_addr,
                         size_t type_size, size_t alloc_size, int array_dim,
                         int *lower, int *upper, int *stride) {
  // config block parameters
  unsigned long long total_elmts = alloc_size / type_size;
  int block_x, block_y, block_z;
  _XMP_gpu_config_block(total_elmts, &block_x, &block_y, &block_z);

  // alloc GPU shadow buffer
  void *gpu_shadow_buffer;
  _XMP_gpu_alloc(&gpu_shadow_buffer, alloc_size);

  // init shadow data on GPU
  int *gpu_lower, *gpu_upper, *gpu_stride;
  size_t lus_size = sizeof(int) * array_dim;
  _XMP_gpu_alloc((void **)&gpu_lower, lus_size);
  _XMP_gpu_alloc((void **)&gpu_upper, lus_size);
  _XMP_gpu_alloc((void **)&gpu_stride, lus_size);
  cudaMemcpy(gpu_lower, lower, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_upper, upper, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_stride, stride, lus_size, cudaMemcpyHostToDevice);

  // pack shadow on GPU
  _XMP_gpu_pack_array_kernel<<<dim3(block_x, block_y, block_z), dim3(16, 16, 1)>>>(device_desc, (char *)gpu_shadow_buffer, (char *)gpu_array_addr,
                                                                                   type_size, total_elmts, array_dim,
                                                                                   gpu_lower, gpu_upper, gpu_stride);
  cudaThreadSynchronize();

  // copy shadow buffer to host
  cudaMemcpy(host_shadow_buffer, gpu_shadow_buffer, alloc_size, cudaMemcpyDeviceToHost);

  // free GPU buffers
  _XMP_gpu_free(gpu_shadow_buffer);
  _XMP_gpu_free(gpu_lower);
  _XMP_gpu_free(gpu_upper);
  _XMP_gpu_free(gpu_stride);
}

extern "C"
void _XMP_gpu_unpack_array(_XMP_gpu_array_t *device_desc, void *gpu_array_addr, void *host_shadow_buffer,
                           size_t type_size, size_t alloc_size, int array_dim,
                           int *lower, int *upper, int *stride) {
  // config block parameters
  unsigned long long total_elmts = alloc_size / type_size;
  int block_x, block_y, block_z;
  _XMP_gpu_config_block(total_elmts, &block_x, &block_y, &block_z);

  // alloc GPU shadow buffer
  void *gpu_shadow_buffer;
  _XMP_gpu_alloc(&gpu_shadow_buffer, alloc_size);

  // init shadow data on GPU
  int *gpu_lower, *gpu_upper, *gpu_stride;
  size_t lus_size = sizeof(int) * array_dim;
  _XMP_gpu_alloc((void **)&gpu_lower, lus_size);
  _XMP_gpu_alloc((void **)&gpu_upper, lus_size);
  _XMP_gpu_alloc((void **)&gpu_stride, lus_size);
  cudaMemcpy(gpu_lower, lower, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_upper, upper, lus_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_stride, stride, lus_size, cudaMemcpyHostToDevice);

  // copy shadow buffer to device
  cudaMemcpy(gpu_shadow_buffer, host_shadow_buffer, alloc_size, cudaMemcpyHostToDevice);

  // pack shadow on GPU
  _XMP_gpu_unpack_array_kernel<<<dim3(block_x, block_y, block_z), dim3(16, 16, 1)>>>(device_desc, (char *)gpu_array_addr, (char *)gpu_shadow_buffer,
                                                                                     type_size, total_elmts, array_dim,
                                                                                     gpu_lower, gpu_upper, gpu_stride);
  cudaThreadSynchronize();

  // free GPU buffers
  _XMP_gpu_free(gpu_shadow_buffer);
  _XMP_gpu_free(gpu_lower);
  _XMP_gpu_free(gpu_upper);
  _XMP_gpu_free(gpu_stride);
}
