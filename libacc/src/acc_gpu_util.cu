#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include <stdio.h>
#include <stdbool.h>

void _ACC_gpu_alloc(void **addr, size_t size)
{
  //printf("_ACC_gpu_alloc\n");
  _ACC_DEBUG("alloc addr=%p, size=%zd\n", addr, size)
  _ACC_gpu_init_current_device_if_not_inited();
  cudaError_t cuda_err = cudaMalloc(addr, size);
  if (cuda_err != cudaSuccess) {
    printf("failed to allocate data on GPU\n");
    _ACC_gpu_fatal(cuda_err);
    //_ACC_fatal("failed to allocate data on GPU");
  }
}

void _ACC_gpu_free(void *addr)
{
  //printf("_ACC_gpu_free\n");
  cudaError_t cuda_err = cudaFree(addr);
  if (cuda_err != cudaSuccess) {
    printf("failed to free data on GPU(%d)\n",(int)cuda_err);
    _ACC_gpu_fatal(cuda_err);
    //_ACC_fatal("failed to free data on GPU");
  }
}

void _ACC_gpu_malloc(void **addr, size_t size)
{
  _ACC_gpu_alloc(addr, size);
}

void _ACC_gpu_calloc(void **addr, size_t size)
{
  //printf("_ACC_gpu_calloc()\n");
  _ACC_gpu_alloc(addr, size);

  cudaError_t cuda_err = cudaMemset(*addr, 0, size);
  if(cuda_err != cudaSuccess){
    _ACC_fatal("failed to clear data on GPU");
  }
}

void _ACC_gpu_copy(void *host_addr, void *device_addr, size_t size, int direction){
  cudaError_t cuda_err;
  if(direction == _ACC_GPU_COPY_HOST_TO_DEVICE){
	_ACC_DEBUG("copy host(%p) to dev(%p), size(%zd)\n", host_addr, device_addr, size)
    cuda_err = cudaMemcpy(device_addr, host_addr, size, cudaMemcpyHostToDevice);
  }else if(direction == _ACC_GPU_COPY_DEVICE_TO_HOST){
	_ACC_DEBUG("copy dev(%p) to host(%p), size(%zd)\n", device_addr, host_addr, size)
    cuda_err = cudaMemcpy(host_addr, device_addr, size, cudaMemcpyDeviceToHost);
  }else{
    _ACC_fatal("invaild direction in 'gpu_copy'");
  }
  
  if(cuda_err != cudaSuccess){
    const char *err_str = cudaGetErrorString(cuda_err);
    _ACC_fatal( (char *)err_str );
  }
}

void _ACC_gpu_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int id){
  //printf("_ACC_gpu_copy_async\n");
  cudaError_t cuda_err;
  cudaStream_t stream = _ACC_gpu_get_stream(id);

  switch(direction){
  case _ACC_GPU_COPY_HOST_TO_DEVICE:
    cuda_err = cudaMemcpyAsync(device_addr, host_addr, size, cudaMemcpyHostToDevice, stream);
    break;
  case _ACC_GPU_COPY_DEVICE_TO_HOST:
    cuda_err = cudaMemcpyAsync(host_addr, device_addr, size, cudaMemcpyDeviceToHost, stream);
    break;
  default:
    _ACC_fatal("invaild direction in 'gpu_copy_async'");
  }
  
  if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }
}

void _ACC_gpu_register_memory(void *host_addr, size_t size){
  //printf("register_memory\n");
  cudaError_t cuda_err = cudaHostRegister(host_addr, size, cudaHostRegisterPortable);
  if( cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
    //return false;
  }else{
    //return true;
  }
}

void _ACC_gpu_unregister_memory(void *host_addr){
  //printf("unregister_memory\n");
  cudaError_t cuda_err = cudaHostUnregister(host_addr);
  if( cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
    //return false;
  }else{
    //return true;
  }
}


void _ACC_gpu_fatal(cudaError_t error)
{
  _ACC_fatal(cudaGetErrorString(error));
}

/*
int _ACC_gpu_get_num_devices()
{
  int count;
  cudaError_t error = cudaGetDeviceCount(&count);
  if(error != cudaSuccess){
    _ACC_gpu_fatal(error);
  }
  return count;
}
*/

bool _ACC_gpu_is_pagelocked(void *p)
{
  unsigned int flags;
  cudaHostGetFlags(&flags, p);
  cudaError_t error = cudaGetLastError();
  return (error == cudaSuccess);
}

void *_ACC_alloc_pinned(size_t size){
  void *addr;
  cudaError_t err = cudaMallocHost((void**)&addr, size);
  if(err != cudaSuccess){
    _ACC_gpu_fatal(err);
  }
  return addr;
}

void _ACC_free_pinned(void *p)
{
  cudaError_t err = cudaFreeHost(p);
  if(err != cudaSuccess){
    _ACC_gpu_fatal(err);
  }
}
