#ifndef _ACC_INTERNAL
#define _ACC_INTERNAL

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct _ACC_array_type _ACC_array_t;
typedef struct _ACC_memory_type _ACC_memory_t;
typedef struct _ACC_data_type _ACC_data_t;
typedef struct _ACC_queue_map_type _ACC_queue_map_t;
typedef struct _ACC_queue_type _ACC_queue_t;
typedef struct _ACC_mpool_type _ACC_mpool_t;

//debug flag
//#define DEBUG

#define ACC_ASYNC_SYNC (-1)
#define ACC_ASYNC_NOVAL (-2)
#define ACC_ASYNC_NULL (-3)

//device type
#ifndef OMNI_TARGET_CPU_CRAY
typedef enum acc_device_t{
  acc_device_none,
  acc_device_default,
  acc_device_host,
  acc_device_not_host,
  acc_device_nvidia,
}acc_device_t;
#else
typedef long acc_device_t;
#define acc_device_none 0
#define acc_device_default 1
#define acc_device_host -1
#define acc_device_not_host -2
#define acc_device_nvidia 2
#endif

extern int _ACC_num_gangs_limit;

#ifdef __cplusplus
extern "C" {
#endif

  //acc_runtime.c
  void _ACC_init(int argc, char** argv);
  void _ACC_finalize(void);

  void *_ACC_alloc(size_t size);
  void _ACC_free(void *p);
  void _ACC_fatal(const char *msg);
  void _ACC_unexpected_error(void);

  //openacc.c
  int acc_get_num_devices( acc_device_t );
  void acc_set_device_type( acc_device_t );
  acc_device_t acc_get_device_type( void );
  void acc_set_device_num( int, acc_device_t );
  int acc_get_device_num( acc_device_t );
  int acc_async_test( int );
  int acc_async_test_all();
  void acc_async_wait( int );
  void acc_async_wait_all();
  void acc_init( acc_device_t );
  void acc_shutdown( acc_device_t );
  int acc_on_device( acc_device_t );
  void* acc_malloc( size_t );
  void acc_free( void* );
  void acc_map_data( void*, void*, size_t );
  void acc_unmap_data( void* );

  //acc_data.c
  void _ACC_init_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]);
  void _ACC_pinit_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]);
  void _ACC_find_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]);
  void _ACC_finalize_data(_ACC_data_t *desc, int type);
  void _ACC_pcopy_data(_ACC_data_t *desc, int direction, int asyncId);
  void _ACC_copy_data(_ACC_data_t *desc, int direction, int asyncId);
  void _ACC_copy_subdata(_ACC_data_t *desc, int direction, int asyncId, unsigned long long lower[], unsigned long long length[]);
  void _ACC_gpu_map_data(void *host_addr, void* device_addr, size_t size);
  void _ACC_gpu_unmap_data(void *host_addr);

  //acc_memory.c
  _ACC_memory_t* _ACC_memory_alloc(void *host_addr, size_t size, void *device_addr);
  void _ACC_memory_free(_ACC_memory_t* memory);
  void _ACC_memory_copy(_ACC_memory_t *data, ptrdiff_t offset, size_t size, int direction, int asyncId);
  void _ACC_memory_copy_sub(_ACC_memory_t* memory, ptrdiff_t memory_offset, int direction, int asyncId, size_t type_size, int dim, unsigned long long lowers[], unsigned long long lengths[], unsigned long long distance[]);
  void _ACC_memory_copy_vector(_ACC_memory_t *data, size_t memory_offset, int direction, int asyncId, size_t type_size, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride);

  void _ACC_memory_increment_refcount(_ACC_memory_t *memory);
  void _ACC_memory_decrement_refcount(_ACC_memory_t *memory);
  unsigned int _ACC_memory_get_refcount(_ACC_memory_t* memory);

  void* _ACC_memory_get_host_addr(_ACC_memory_t* memory);
  size_t _ACC_memory_get_size(_ACC_memory_t* memory);
  ptrdiff_t _ACC_memory_get_host_offset(_ACC_memory_t* data, void *host_addr);
  void* _ACC_memory_get_device_addr(_ACC_memory_t* data, ptrdiff_t offset);


  //acc_memory_table.c
  void _ACC_gpu_init_data_table();
  void _ACC_gpu_finalize_data_table();
  void _ACC_memory_table_add(void *host_addr, size_t size, _ACC_memory_t* memory);
  _ACC_memory_t* _ACC_memory_table_remove(void *addr, size_t size);
  _ACC_memory_t* _ACC_memory_table_find(void *addr, size_t size);

  //acc_queue_*.c
  _ACC_queue_t* _ACC_queue_create(int async_num);
  void _ACC_queue_destroy(_ACC_queue_t *queue);
  void _ACC_queue_wait(_ACC_queue_t *queue);
  int _ACC_queue_test(_ACC_queue_t *queue);
  void* _ACC_queue_get_mpool(_ACC_queue_t *queue);
  unsigned* _ACC_queue_get_block_count(_ACC_queue_t *queue);

  //acc_queue_map.c
  _ACC_queue_map_t* _ACC_gpu_init_stream_map(int hashtable_size);
  void _ACC_gpu_finalize_stream_map(_ACC_queue_map_t*);
  void _ACC_gpu_wait(int async_num);
  void _ACC_gpu_wait_all();
  int _ACC_gpu_test(int async_num);
  int _ACC_gpu_test_all();
  void _ACC_mpool_get(void **ptr);
  void _ACC_mpool_get_async(void **ptr, int async_num);
  void _ACC_gpu_get_block_count(unsigned **count);
  void _ACC_gpu_get_block_count_async(unsigned **count, int async_num);
  _ACC_queue_t* _ACC_queue_map_get_queue(int async_num);
  void _ACC_queue_map_set_queue(int async_num, _ACC_queue_t* queue);

  //acc_platform?
  void _ACC_platform_set_device_num(int device_num);
  bool _ACC_platform_allocate_device(int dev_num);
  void _ACC_platform_init_device(int device_num);
  int _ACC_platform_get_num_devices();
  void _ACC_platform_init();
  void _ACC_platform_finalize();

  //acc.c
  void _ACC_init(int argc, char** argv);
  void _ACC_finalize(void);
  void _ACC_init_type(acc_device_t device_type);
  void _ACC_init_api(void);
  void _ACC_finalize_type(acc_device_t device_type);
  void _ACC_set_device_num(int num);//num is 0-based
  int _ACC_get_device_num(); //return value is 0-based
  int _ACC_normalize_device_num(int device_num);
  acc_device_t _ACC_normalize_device_type(acc_device_t device_t);
  void _ACC_init_current_device_if_not_inited();
  _ACC_queue_map_t* _ACC_get_queue_map();
  _ACC_mpool_t* _ACC_get_mpool();
  void acc_init_(int *argc, char** argv); //for Fortran
  void acc_finalize_(); //for Fortran

  //acc_mpool_*.c
  _ACC_mpool_t* _ACC_mpool_create();
  void _ACC_mpool_destroy(_ACC_mpool_t *);
  void _ACC_mpool_alloc_block(void **);
  void _ACC_mpool_free_block(void *);
  void _ACC_mpool_alloc(void **ptr, long long size, void *mpool, long long *pos);
  void _ACC_mpool_free(void *ptr, void *mpool);

  void _ACC_gpu_alloc(void **addr, size_t size);
  void _ACC_gpu_free(void *addr);

  void _ACC_copy(void *host_addr, void *device_addr, size_t size, int direction);
  void _ACC_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int async);

#ifdef __cplusplus
}
#endif


#ifdef DEBUG
//#define _ACC_DEBUG(...) {printf("%s(l.%d):", __func__, __LINE__); printf(__VA_ARGS__);}
#define _ACC_DEBUG(...) printf("%s(%d)[%s]: ", __FILE__, __LINE__, __func__); printf(__VA_ARGS__);
#else
#define _ACC_DEBUG(...)
#endif


#endif //_ACC_INTERNAL
