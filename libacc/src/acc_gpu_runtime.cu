#include <stdio.h>
#include <stdlib.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#define BUF_LEN 256

int _ACC_gpu_device_count;
int _ACC_gpu_max_thread;
int _ACC_gpu_max_block_dim_x;
int _ACC_gpu_max_block_dim_y;
int _ACC_gpu_max_block_dim_z;

static int current_device_num = 0;
static void init_device(int dev_num);

void _ACC_gpu_init(void) {
  cudaError_t cuda_err;
  int i;

  cuda_err = cudaGetDeviceCount(&_ACC_gpu_device_count);
  if(cuda_err == cudaErrorNoDevice){
    _ACC_fatal("no GPU device");
  }else if(cuda_err == cudaErrorInsufficientDriver){
    _ACC_fatal("installed CUDA driver is older than CUDA runtime library");
  }else if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }

  _ACC_DEBUG("Total number of GPUs = %d\n", _ACC_gpu_device_count)

  //init each GPUs
  for(i=0;i<_ACC_gpu_device_count;i++){
    init_device(i);
  }
}

void _ACC_gpu_finalize(void) {
  return;
}

int _ACC_gpu_get_num_devices()
{
  int count;
  cudaError_t error = cudaGetDeviceCount(&count);
  if(error != cudaSuccess){
    _ACC_gpu_fatal(error);
  }
  return count;
}

static void init_device(int dev_num){
  cudaError_t cuda_err;
  _ACC_DEBUG("initializing GPU %d\n",dev_num)

  if (cudaSetDevice(dev_num) != cudaSuccess) {
    _ACC_fatal("fail to set GPU device");
  }
  
  if(cudaDeviceReset() != cudaSuccess){
    _ACC_fatal("failed to reset GPU");
  }

  cudaDeviceProp dev_prop;
  cuda_err = cudaGetDeviceProperties(&dev_prop, dev_num);
  if(cuda_err != cudaSuccess){
    _ACC_fatal("fail to get GPU device properties");
  }
  _ACC_DEBUG("name : %s\n", dev_prop.name)
  _ACC_DEBUG("clock : %dKHz\n", dev_prop.clockRate)
  _ACC_DEBUG("cc : %d.%d\n",dev_prop.major, dev_prop.minor)
  _ACC_DEBUG("#sm : %d\n",dev_prop.multiProcessorCount)
  int *dummy;
  _ACC_gpu_alloc((void **)&dummy, sizeof(int));
  _ACC_gpu_free(dummy);
}

void _ACC_gpu_set_device_num(int num)
{
  _ACC_DEBUG("device_num(%d)\n",num)
  cudaError_t cuda_err;

  if(num < 0 || num > _ACC_gpu_device_count){
    _ACC_fatal("invalid device num in _ACC_gpu_set_device_num");
  }

  //finalize stream hashmap for previous device
  _ACC_gpu_finalize_stream_map();

  if(num == 0){ // 0 means default device num
    current_device_num = 0;
  }else{
    current_device_num = num - 1;
  }

  if (cudaSetDevice(current_device_num) != cudaSuccess) {
    _ACC_fatal("fail to set GPU device in _ACC_gpu_set_device_num");
  }
  
  cudaDeviceProp dev_prop;
  cuda_err = cudaGetDeviceProperties(&dev_prop, current_device_num);
  if(cuda_err != cudaSuccess){
    _ACC_fatal("fail to get GPU device properties in _ACC_gpu_set_device_num");
  }

  _ACC_gpu_max_thread = dev_prop.maxThreadsPerBlock;
  _ACC_gpu_max_block_dim_x = dev_prop.maxGridSize[0];
  _ACC_gpu_max_block_dim_y = dev_prop.maxGridSize[1];
  _ACC_gpu_max_block_dim_z = dev_prop.maxGridSize[2];

  //init stream hashmap
  _ACC_gpu_init_stream_map(64);
}

int _ACC_gpu_get_device_num(){
  return current_device_num + 1;
}
