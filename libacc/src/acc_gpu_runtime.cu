#include <stdio.h>
#include <stdlib.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#define BUF_LEN 256

int _ACC_gpu_device_count;
//int _ACC_gpu_max_thread;
//int _ACC_gpu_max_block_dim_x;
//int _ACC_gpu_max_block_dim_y;
//int _ACC_gpu_max_block_dim_z;

static int current_device_num = 0;
static void init_device(int dev_num);
static void finalize_device(int dev_num);

typedef struct acc_context{
  char isInitialized;
  void *stream_map;
  void *mpool;
}acc_context;

acc_context *contexts;


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

  contexts = (acc_context*)_ACC_alloc(sizeof(acc_context) * _ACC_gpu_device_count);
  for(i = 0; i< _ACC_gpu_device_count; i++){
	contexts[i].isInitialized = 0;
	contexts[i].stream_map = NULL;
	contexts[i].mpool = NULL;
  }

  // //init each GPU
  // for(i=0;i<_ACC_gpu_device_count;i++){
  //   init_device(i);
  // }

  _ACC_gpu_set_device_num(0); //set device to default
}

void _ACC_gpu_finalize(void) {
  //finalize each GPU
  for(int i=0;i<_ACC_gpu_device_count;i++){
	if(contexts[i].isInitialized){
	  _ACC_gpu_set_device_num(i+1);
	  finalize_device(i);
	}
  }

  _ACC_free(contexts);
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

static void init_device(int dev_num){ //0-based
  cudaError_t cuda_err;
  _ACC_DEBUG("initializing GPU %d\n",dev_num)

  if (cudaSetDevice(dev_num) != cudaSuccess) {
    _ACC_fatal("fail to set GPU device");
  }
  int *dummy;
  _ACC_gpu_alloc((void **)&dummy, sizeof(int));
  _ACC_gpu_free(dummy);
  
  // if(cudaDeviceReset() != cudaSuccess){
  //   _ACC_fatal("failed to reset GPU");
  // }

  cudaDeviceProp dev_prop;
  cuda_err = cudaGetDeviceProperties(&dev_prop, dev_num);
  if(cuda_err != cudaSuccess){
    _ACC_fatal("fail to get GPU device properties");
  }
  _ACC_DEBUG("name : %s\n", dev_prop.name)
  _ACC_DEBUG("clock : %dKHz\n", dev_prop.clockRate)
  _ACC_DEBUG("cc : %d.%d\n",dev_prop.major, dev_prop.minor)
  _ACC_DEBUG("#sm : %d\n",dev_prop.multiProcessorCount)


  //init mpool
  contexts[dev_num].isInitialized = 1;
  contexts[dev_num].mpool = _ACC_gpu_mpool_init();
  //init stream hashmap
  contexts[dev_num].stream_map = _ACC_gpu_init_stream_map(16);
}

static void finalize_device(int dev_num){
  
  //finalize stream hashmap for previous device
  acc_context cont = contexts[dev_num];
  //printf("finalize_map(%d, %p)\n", dev_num, cont.stream_map);
  if(contexts[dev_num].isInitialized){
	_ACC_gpu_finalize_stream_map(cont.stream_map);
	_ACC_gpu_mpool_finalize(cont.mpool);
  }
  contexts[dev_num].isInitialized = 0;
}

void _ACC_gpu_set_device_num(int num)
{
  /* num is 1-origin */
  _ACC_DEBUG("device_num(%d)\n",num)
  cudaError_t cuda_err;

  if(num < 0 || num > _ACC_gpu_device_count){
    _ACC_fatal("invalid device num in _ACC_gpu_set_device_num");
  }

  if(num == 0){ // 0 means default device num
    current_device_num = 0;
  }else{
    current_device_num = num - 1;
  }

  if (cudaSetDevice(current_device_num) != cudaSuccess) {
    _ACC_fatal("fail to set GPU device in _ACC_gpu_set_device_num");
  }
  
  // acc_context cont = contexts[current_device_num];
  // _ACC_gpu_set_stream_map(cont.stream_map);
  // _ACC_gpu_mpool_set(cont.mpool);

}

int _ACC_gpu_get_device_num(){
  return current_device_num + 1;
}

void _ACC_gpu_init_device_if_not_inited(int num) //0-based
{
  if(! contexts[num].isInitialized){
	init_device(num);
  }
}

void* _ACC_gpu_get_current_stream_map()
{
  _ACC_DEBUG("get_current_stream_map\n")
  void *stream_map = contexts[current_device_num].stream_map;
  if(stream_map == NULL){
	_ACC_gpu_init_device_if_not_inited(current_device_num);
  }
  return contexts[current_device_num].stream_map;
}

void* _ACC_gpu_get_current_mpool()
{
  _ACC_DEBUG("get_current_mpool\n")
  void *mpool = contexts[current_device_num].mpool;
  if(mpool == NULL){
	_ACC_gpu_init_device_if_not_inited(current_device_num);
  }
  return contexts[current_device_num].mpool;
}
