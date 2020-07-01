#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "acc_internal.h"


/*device_num == 0 means that default device num will be used*/
/*we have to specify number from 1 to num_of_device to use particular device*/

//general
const acc_device_t default_device = acc_device_nvidia;
const acc_device_t default_not_host_device = acc_device_nvidia;
static acc_device_t current_device = acc_device_none; /* current_device = {none, host, nvidia} */

//for host
const int host_default_device_num = 1;
static int host_device_num = 1; //host_default_device_num;

//for nvidia gpu
//const int gpu_default_device_num = 1;
//static int gpu_device_num = 1; //gpu_default_device_num;

int acc_get_num_devices( acc_device_t acc_device )
{
  switch(acc_device){
  case acc_device_none:
    return 0;

  case acc_device_default:
    return acc_get_num_devices(default_device);

  case acc_device_host:
    return 1; //use 1 thread

  case acc_device_not_host:
    return acc_get_num_devices( default_not_host_device );

  case acc_device_nvidia:
    return _ACC_platform_get_num_devices( acc_device );

  default:
    _ACC_fatal("unknown device type");
  }
  return 0;
}

void acc_set_device_type( acc_device_t acc_device )
{
  _ACC_DEBUG("acc_set_device_type(type:%d)\n", acc_device)
  switch(acc_device){
  case acc_device_none:
    //    _ACC_fatal("acc_set_device_type : acc_device_none is not allowed");
    current_device = acc_device_none;
    break;

  case acc_device_default:
    acc_set_device_type( default_device );
    return;

  case acc_device_host:
    current_device = acc_device_host;
    return;

  case acc_device_not_host:
    acc_set_device_type( default_not_host_device );
    return;

  case acc_device_nvidia:
    current_device = acc_device_nvidia;
    _ACC_platform_set_device_type(acc_device);
    return;

  default:
    _ACC_fatal("acc_set_device_type : unknown device type");
  }
}

acc_device_t acc_get_device_type( void )
{
  return current_device;
}

void acc_set_device_num( int num, acc_device_t acc_device )
{
  _ACC_DEBUG("num=%d, type:%d\n",num, (int)acc_device)
  acc_set_device_type(acc_device);

  if( num < 0 ){
    _ACC_fatal("acc_set_device_num : negative integer is not allowed for 'device_num'");
  }

  if( num > acc_get_num_devices(acc_device) ){
    _ACC_fatal("acc_set_device_num : 'device_num' is greater than number of devices");
  }


  switch(acc_device){
  case acc_device_none:
    _ACC_fatal("acc_set_device_num : invalid device type 'acc_device_none'");

  case acc_device_default:
    acc_set_device_num( num, default_device );
    return;

  case acc_device_host:
    host_device_num = num;
    return;

  case acc_device_not_host:
    acc_set_device_num( num, default_not_host_device );
    return;

  case acc_device_nvidia:
    _ACC_set_device_num(num-1);
    return;

  default:
    _ACC_fatal("acc_set_device_num : unknown device type");
  }
}

int acc_get_device_num( acc_device_t acc_device)
{
  switch(acc_device){
  case acc_device_none:
    _ACC_fatal("acc_get_device_num : invalid device type 'acc_device_none'");

  case acc_device_default:
    return acc_get_device_num( default_device );

  case acc_device_host:
    return host_device_num;

  case acc_device_not_host:
    return acc_get_device_num( default_not_host_device );

  case acc_device_nvidia:
    //return gpu_device_num;
    return _ACC_get_device_num() + 1;

  default:
    _ACC_fatal("acc_get_device_num : unknown device type");
  }
  return 0;
}

int acc_async_test( int id )
{
  switch(current_device){
    //  case acc_device_none:
    //  case acc_device_host:
  case acc_device_nvidia:
    return _ACC_gpu_test(id);    
  default:
    _ACC_fatal("acc_async_test : invalid device type");
    return 0;
  }
}

int acc_async_test_all()
{
  switch(current_device){
    //  case acc_device_none:
    //  case acc_device_host:
  case acc_device_nvidia:
    return _ACC_gpu_test_all();
  default:
    _ACC_fatal("acc_async_test_all : invalid device type");
    return 0;
  }
}

void acc_async_wait( int id )
{  
  switch(current_device){
    //  case acc_device_none:
    //  case acc_device_host:
  case acc_device_nvidia:
    _ACC_gpu_wait(id);
    return;
  default:
    _ACC_fatal("acc_async_wait : invalid device type");
  }
}

void acc_async_wait_all()
{
  switch(current_device){
    //  case acc_device_none:
    //  case acc_device_host:
  case acc_device_nvidia:
    _ACC_gpu_wait_all();
    return;
  default:
    _ACC_fatal("acc_async_wait_all : invalid device type");
  }
}


void acc_init( acc_device_t acc_device )
{
  switch(acc_device){
  case acc_device_none:
    _ACC_fatal("acc_init : invalid device type 'acc_device_none'");
    return;

  case acc_device_default:
    acc_init( default_device );
    return;

  case acc_device_host:
    return;

  case acc_device_not_host:
    acc_init( default_not_host_device );
    return;

  case acc_device_nvidia:
    _ACC_init_type(acc_device);
    return;

  default:
    _ACC_fatal("acc_init : unknows device type");
  }
}

void acc_shutdown( acc_device_t acc_device )
{
  _ACC_finalize_type(_ACC_normalize_device_type(acc_device));
}

//FIXME implement
//for device==host
int acc_on_device( acc_device_t acc_device )
{
  switch(acc_device){
    case acc_device_none:
    _ACC_fatal("acc_shutdown : invalid device type 'acc_device_none'");
    return 0;

  case acc_device_default:
    return acc_on_device( default_device );

  case acc_device_host:
    return 1;

  case acc_device_not_host:
    return 0;

  case acc_device_nvidia:
    return 0;

  default:
    _ACC_fatal("acc_shutdown : unknows device type");
  }
  return 0;
}


void* acc_malloc( size_t size )
{
  void *addr;
  switch(current_device){
  case acc_device_none:
    _ACC_fatal("acc_malloc : device is not selected");
 
  case acc_device_host:
    return _ACC_alloc(size);

  case acc_device_nvidia:
    _ACC_gpu_alloc((void **)&addr, size);
    return addr;

  default:
    _ACC_fatal("acc_malloc : unknown device type");
    return NULL;
  }
}

void acc_free( void* ptr)
{
  switch(current_device){
  case acc_device_none:
    _ACC_fatal("acc_free : device is not selected");
    
  case acc_device_host:
    _ACC_free(ptr);
    return;

  case acc_device_nvidia:
    _ACC_gpu_free(ptr);
    return;

  default:
    _ACC_fatal("acc_free : unknown device type");
  }
}

void acc_map_data(void *host_p, void *dev_p, size_t size)
{
  switch(current_device){
  case acc_device_none:
    _ACC_fatal("acc_map_data : device is not selected");
    
  case acc_device_host:
    return;

  case acc_device_nvidia:
    _ACC_gpu_map_data(host_p, dev_p, size);
    return;

  default:
    _ACC_fatal("acc_map_data : unknown device type");
  }

}

void acc_unmap_data(void *host_p)
{
  switch(current_device){
  case acc_device_none:
    _ACC_fatal("acc_unmap_data : device is not selected");
    
  case acc_device_host:
    return;

  case acc_device_nvidia:
    _ACC_gpu_unmap_data(host_p);
    return;

  default:
    _ACC_fatal("acc_unmap_data : unknown device type");
  }
}
