#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"

static bool _ACC_runtime_working = false;

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


void _ACC_init(int argc, char** argv) {
  _ACC_DEBUG("begin\n")
  if (! _ACC_runtime_working) {
    _ACC_runtime_working = true;
  }
  
  //get environment variable
  char *acc_device_type_str;
  acc_device_t device_t = acc_device_default;
  acc_device_type_str = getenv("ACC_DEVICE_TYPE");
  if(acc_device_type_str == NULL){ // if not defined
    //acc_set_device_type(acc_device_default);
    device_t = acc_device_default;
  }else{
    if (strncmp(acc_device_type_str, "NVIDIA", 10) == 0){
      //acc_set_device_type(acc_device_nvidia);
      device_t = acc_device_nvidia;
    }else if(strncmp(acc_device_type_str, "HOST", 10) == 0){
      //acc_set_device_type(acc_device_host);
      device_t = acc_device_host;
    }else if(strncmp(acc_device_type_str, "NONE", 10) == 0){
      //acc_set_device_type(acc_device_none);
      device_t = acc_device_none;
    }else{
      _ACC_fatal("invalid device type is defined by ACC_DEVICE_TYPE");
    }
  }
  acc_init(device_t);

  char *acc_device_num;
  acc_device_num = getenv("ACC_DEVICE_NUM");
  if(acc_device_num == NULL){
    acc_set_device_num(0, device_t); //set default device
  }else{
    int dev_num = atoi(acc_device_num);
    acc_set_device_num(dev_num, device_t);
  }
  _ACC_DEBUG("end\n")
}

void _ACC_finalize(void) {
  _ACC_DEBUG("begin\n");
  switch(current_device){
  case acc_device_none:
  case acc_device_host:
    break;

  case acc_device_nvidia:
    _ACC_gpu_finalize();
    break;

  default:
    _ACC_fatal("unknown device type");
  }

  if (_ACC_runtime_working) {
    _ACC_runtime_working = false;
  }
  _ACC_DEBUG("end\n");
}

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
    return _ACC_gpu_get_num_devices();

  default:
    _ACC_fatal("unknown device type");
  }
  return 0;
}

void acc_set_device_type( acc_device_t acc_device )
{
  _ACC_DEBUG("acc_set_device(type:%d)\n",(int)acc_device)
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
    //gpu_device_num = num;
    _ACC_gpu_set_device_num(num);
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
    return _ACC_gpu_get_device_num();

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
    _ACC_gpu_init();
    return;

  default:
    _ACC_fatal("acc_init : unknows device type");
  }
}

void acc_shutdown( acc_device_t acc_device )
{
  switch(acc_device){
  case acc_device_none:
    _ACC_fatal("acc_shutdown : acc_device_none is unavailable");
    return;

  case acc_device_default:
    acc_shutdown( default_device );
    return;

  case acc_device_host:
    return;

  case acc_device_not_host:
    acc_shutdown( default_not_host_device );
    return;

  case acc_device_nvidia:
    _ACC_gpu_finalize();
    return;

  default:
    _ACC_fatal("acc_shutdown : unknows device type");
  }
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
