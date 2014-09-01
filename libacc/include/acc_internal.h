#ifndef _ACC_INTERNAL
#define _ACC_INTERNAL

#include <stdlib.h>


//debug flag
//#define DEBUG

#define ACC_ASYNC_SYNC (-1)
#define ACC_ASYNC_NOVAL (-2)

//device type
typedef enum acc_device_t{
  acc_device_none,
  acc_device_default,
  acc_device_host,
  acc_device_not_host,
  acc_device_nvidia,
}acc_device_t;


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
