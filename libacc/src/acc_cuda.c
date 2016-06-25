#include <stdio.h>
#include <stdlib.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "cuda_runtime.h"

int _ACC_platform_get_num_devices()
{
  int count;
  cudaError_t error = cudaGetDeviceCount(&count);
  if(error != cudaSuccess){
    _ACC_gpu_fatal(error);
  }
  return count;
}

bool _ACC_platform_allocate_device(int device_num)
{
  _ACC_DEBUG("alloc device (%d)\n", device_num)
  if (cudaSetDevice(device_num) != cudaSuccess) {
    return false;
  }

  int *dummy;
  if(cudaMalloc((void**)&dummy, sizeof(int)) != cudaSuccess){
    return false;
  }
  if(cudaFree(dummy) != cudaSuccess){
    return false;
  }

  return true;
}

void _ACC_platform_set_device_num(int device_num /*0-based*/)
{
  cudaError_t cuda_err = cudaSetDevice(device_num);
  if (cuda_err != cudaSuccess) {
    _ACC_fatal("fail to set GPU device in _ACC_gpu_set_device_num");
  }
}

void _ACC_platform_init_device(int device_num /*0-based*/)
{
  struct cudaDeviceProp dev_prop;
  cudaError_t cuda_err = cudaGetDeviceProperties(&dev_prop, device_num);
  if(cuda_err != cudaSuccess){
    _ACC_fatal("fail to get GPU device properties");
  }
  _ACC_DEBUG("name : %s\n", dev_prop.name)
  _ACC_DEBUG("clock : %dKHz\n", dev_prop.clockRate)
  _ACC_DEBUG("cc : %d.%d\n",dev_prop.major, dev_prop.minor)
  _ACC_DEBUG("#sm : %d\n",dev_prop.multiProcessorCount)

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

void _ACC_platform_init()
{
}
void _ACC_platform_finalize()
{
}
