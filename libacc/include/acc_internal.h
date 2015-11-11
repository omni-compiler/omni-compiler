#ifndef _ACC_INTERNAL
#define _ACC_INTERNAL

#include <stdlib.h>
#include <stddef.h>

//#include "acc_data_struct.h"
typedef struct _ACC_array_type _ACC_array_t;
typedef struct _ACC_memory_type _ACC_memory_t;
typedef struct _ACC_data_type _ACC_data_t;

//debug flag
//#define DEBUG

#define ACC_ASYNC_SYNC (-1)
#define ACC_ASYNC_NOVAL (-2)

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
