#ifndef _ACC_GPU_INTERNAL
#define _ACC_GPU_INTERNAL

#include "acc_gpu_constant.h"
#include "acc_gpu_data_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

  //acc_gpu_runtime.c
  void _ACC_gpu_init(void);
  void _ACC_gpu_finalize(void);
  void _ACC_gpu_set_device_num(int num);
  int _ACC_gpu_get_device_num();

  //acc_gpu_data.c
  //  void _ACC_gpu_init_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t size);
  void _ACC_gpu_init_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t offset, size_t size);
  void _ACC_gpu_pinit_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t offset, size_t size);
  void _ACC_gpu_finalize_data(_ACC_gpu_data_t *desc);
  //  void _ACC_gpu_copy_data(_ACC_gpu_data_t *desc, int direction);
  void _ACC_gpu_copy_data(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction);
  void _ACC_gpu_copy_data_async_default(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction);
  void _ACC_gpu_pcopy_data(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction);
  void _ACC_gpu_find_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t offset, size_t size);

  //acc_gpu_util.cu
  void _ACC_gpu_alloc(void **addr, size_t size);
  void _ACC_gpu_malloc(void **addr, size_t size);
  void _ACC_gpu_calloc(void **addr, size_t size);
  void _ACC_gpu_free(void *addr);
  void _ACC_gpu_copy(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_gpu_copy_async_all(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_gpu_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int id);

  //acc_gpu_data_table.c
  void _ACC_gpu_init_data_table();
  void _ACC_gpu_finalize_data_table();
  void _ACC_gpu_add_data(_ACC_gpu_data_t *host_desc);
  _Bool _ACC_gpu_remove_data(_ACC_gpu_data_t *host_desc);
  void _ACC_gpu_get_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t size);
  void _ACC_gpu_get_data_sub(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t offset, size_t size);

  //acc_gpu_stream.cu
  void _ACC_gpu_init_stream_map(int table_size);
  void _ACC_gpu_finalize_stream_map();
  void _ACC_gpu_wait(int id);
  void _ACC_gpu_wait_all();
  int _ACC_gpu_test(int id);
  int _ACC_gpu_test_all();


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
