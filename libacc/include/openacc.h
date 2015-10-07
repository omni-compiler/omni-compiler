// last modified 13/03/29

#ifndef _OPENACC_HEADER
#define _OPENACC_HEADER

#include <stddef.h>

#define _OPENACC 201111
#define ACC_ASYNC_SYNC (-1)
#define ACC_ASYNC_NOVAL (-2)

//device type
#ifndef _XMP_CRAY
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


//runtime library routines
#ifdef __cplusplus
extern "C" {
#endif

  /* returns the number of accelerator devices of the given type */
  int acc_get_num_devices( acc_device_t );

  /* sets the accelerator device type to execute an accelerator region */
  void acc_set_device_type( acc_device_t );

  /* returns the accelerator device type to execute the next accelerator region */
  acc_device_t acc_get_device_type( void );

  /* sets which accelerator device to use */
  void acc_set_device_num( int, acc_device_t );

  /* returns the device number of the given type that be used to execute the next accelerator region */
  int acc_get_device_num( acc_device_t );

  /* returns nonzero if all asynchronous activities with the given expression have completed, otherwise returns zero */
  int acc_async_test( int );

  /* returns nonzero if all asynchronous activities have completed, otherwise returns zero */
  int acc_async_test_all();

  /* waits until all asynchronous activities with the given expression have completed.*/
  void acc_async_wait( int );

  /* waits until all asynchronous activities have completed */
  void acc_async_wait_all();

  /* initializes the runtime for the given device type */
  void acc_init( acc_device_t );

  /* shutdowns the connection to the given accelerator device, and free up any runtime resources */
  void acc_shutdown( acc_device_t );

  /* returns nonzero if the program is running on the given device, otherwise returns zero */
  int acc_on_device( acc_device_t );

  /* returns the address of memory allocated on the accelerator device */
  void* acc_malloc( size_t );

  /* frees memory allocated by acc_malloc */
  void acc_free( void* );

  /* maps previously allocated device data to the specified host data */
  void acc_map_data( void*, void*, size_t );

  /* unmaps device data from the specified host data*/
  void acc_unmap_data( void* );
  
#ifdef __cplusplus
}
#endif

#endif //_OPENACC_HEADER
