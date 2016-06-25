#ifndef _ACC_GPU_INTERNAL
#define _ACC_GPU_INTERNAL

#include "acc_gpu_constant.h"
#include "acc_gpu_data_struct.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

  //acc_gpu_util.cu
  void _ACC_gpu_malloc(void **addr, size_t size);
  void _ACC_gpu_calloc(void **addr, size_t size);
  void _ACC_gpu_copy(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_gpu_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int id);
  bool _ACC_gpu_is_pagelocked(void *p);
  void _ACC_gpu_register_memory(void *host_addr, size_t size);
  void _ACC_gpu_unregister_memory(void *host_addr);
  void *_ACC_alloc_pinned(size_t size);
  void _ACC_free_pinned(void *p);
  void _ACC_gpu_fatal(cudaError_t error);
  cudaStream_t _ACC_gpu_get_stream(int id);



  //acc_gpu_pack.cu
  void _ACC_gpu_pack_data(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info, int asyncId);
  void _ACC_gpu_unpack_data(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info, int asyncId);
  void _ACC_gpu_pack_data_host(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info);
  void _ACC_gpu_unpack_data_host(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info);
  void _ACC_gpu_pack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize, int asyncId);
  void _ACC_gpu_unpack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize, int asyncId);
  void _ACC_pack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize);
  void _ACC_unpack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize);

#ifdef __cplusplus
}
#endif

#endif //_ACC_GPU_INTERNAL
