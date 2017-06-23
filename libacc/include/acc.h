#define ACC_ASYNC_SYNC (-1)
#define ACC_ASYNC_NOVAL (-2)

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

  void _ACC_init(int argc, char** argv);
  void _ACC_finalize(void);

  void _ACC_init_data		(void **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]);  
  void _ACC_pinit_data		(void **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]);
  void _ACC_find_data		(void **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]);
  void _ACC_devptr_init_data	(void **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]);
  void _ACC_finalize_data(void *desc, int type);
  void _ACC_pcopy_data(void *desc, int direction, int asyncId);
  void _ACC_copy_data(void *desc, int direction, int asyncId);
  void _ACC_copy_subdata(void *desc, int direction, int asyncId, unsigned long long lower[], unsigned long long length[]);

  void _ACC_mpool_get(void **ptr);
  void _ACC_mpool_get_async(void **ptr, int async_num);
  void _ACC_mpool_alloc(void **ptr, long long size, void *mpool, long long *pos);
  void _ACC_mpool_free(void *ptr, void *mpool);

  void _ACC_copy(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int async);

  void _ACC_program_init(void **desc, char * kernel_src_filename, int num_kernels, char ** kernel_names);
  void _ACC_launch(void *program, int kernel_num, int *_ACC_conf, int async_num, int num_args, unsigned long long/*instead of size_t*/ *arg_sizes, void **args);
  void _ACC_program_finalize(void *program);

  int _ACC_adjust_num_gangs(long long num_gangs, int limit);

  // will be renamed
  void _ACC_gpu_wait(int id);
  void _ACC_gpu_wait_all(void);


  //will be removed
  void _ACC_gpu_get_block_count(unsigned **count);
  void _ACC_gpu_get_block_count_async(unsigned **count, int async_num);

#ifdef __CUDACC__
  cudaStream_t _ACC_gpu_get_stream(int id);
#endif

#ifdef __cplusplus
}
#endif
