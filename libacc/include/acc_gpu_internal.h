#ifndef _ACC_GPU_INTERNAL
#define _ACC_GPU_INTERNAL

#include "acc_gpu_constant.h"
#include "acc_gpu_data_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

  //acc_gpu_runtime.c
  void _ACC_gpu_init(void);
  void _ACC_gpu_init_api(void);
  void _ACC_gpu_finalize(void);
  void _ACC_gpu_set_device_num(int num);
  int _ACC_gpu_get_device_num();
  void *_ACC_gpu_get_current_stream_map();
  void* _ACC_gpu_get_current_mpool();
  void _ACC_gpu_init_current_device_if_not_inited();

  //acc_gpu_util.cu
  void _ACC_gpu_alloc(void **addr, size_t size);
  void _ACC_gpu_malloc(void **addr, size_t size);
  void _ACC_gpu_calloc(void **addr, size_t size);
  void _ACC_gpu_free(void *addr);
  void _ACC_gpu_copy(void *host_addr, void *device_addr, size_t size, int direction);
  //  void _ACC_gpu_copy_async_all(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_gpu_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int id);
  bool _ACC_gpu_is_pagelocked(void *p);
  void _ACC_gpu_register_memory(void *host_addr, size_t size);
  void _ACC_gpu_unregister_memory(void *host_addr);
  void *_ACC_alloc_pinned(size_t size);
  void _ACC_free_pinned(void *p);

  //acc_gpu_stream.cu
  void* _ACC_gpu_init_stream_map(int table_size);
  void _ACC_gpu_finalize_stream_map(void*);
  void _ACC_gpu_set_stream_map(void*);
  void _ACC_gpu_wait(int id);
  void _ACC_gpu_wait_all();
  int _ACC_gpu_test(int id);
  int _ACC_gpu_test_all();
  void _ACC_gpu_mpool_get(void **ptr);
  void _ACC_gpu_mpool_get_async(void **ptr, int id);
  void _ACC_gpu_get_block_count(unsigned **count);
  void _ACC_gpu_get_block_count_async(unsigned **count, int id);

  //acc_gpu_mpool.cu
  void* _ACC_gpu_mpool_init();
  void _ACC_gpu_mpool_finalize(void *);
  void _ACC_gpu_mpool_set(void *);
  void _ACC_gpu_mpool_alloc_block(void **);
  void _ACC_gpu_mpool_free_block(void *);
  void _ACC_gpu_mpool_alloc(void **ptr, long long size, void *mpool, long long *pos);
  void _ACC_gpu_mpool_free(void *ptr, void *mpool);

  //acc_gpu_pack.cu
  void _ACC_gpu_pack_data(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info, int asyncId);
  void _ACC_gpu_unpack_data(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info, int asyncId);
  void _ACC_gpu_pack_data_host(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info);
  void _ACC_gpu_unpack_data_host(void *dst, void *src, int dim, unsigned long long total_elmnts, int type_size, unsigned long long* info);
  void _ACC_gpu_pack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize, int asyncId);
  void _ACC_gpu_unpack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize, int asyncId);
  void _ACC_pack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize);
  void _ACC_unpack_vector(void *dst, void *src, unsigned long long count, unsigned long long blocklength, unsigned long long stride, size_t typesize);

  //temporal funcdef
  int _ACC_gpu_get_num_devices();

#ifdef __CUDACC__
  //acc_gpu_util.cu
  void _ACC_gpu_fatal(cudaError_t error);
  //acc_gpu_stream.cu
  cudaStream_t _ACC_gpu_get_stream(int id);
#endif
  
#ifdef __cplusplus
}
#endif

#endif //_ACC_GPU_INTERNAL
