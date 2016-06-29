#include <stdio.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "cuda_runtime.h"

void _ACC_gpu_fatal(cudaError_t error); //FIXME should be removed

struct _ACC_queue_type{
  int async_num;
  cudaStream_t stream;
  void *mpool;
  unsigned *block_count;
};

_ACC_queue_t* _ACC_queue_create(int async_num)
{
  _ACC_DEBUG("queue create\n")
  _ACC_queue_t *queue = (_ACC_queue_t *)_ACC_alloc(sizeof(_ACC_queue_t));
  if(async_num != ACC_ASYNC_SYNC){
    cudaError_t error = cudaStreamCreate(&(queue->stream));
    if(error != cudaSuccess){
      _ACC_gpu_fatal(error);
    }
  }else{
    queue->stream = NULL;
  }
  queue->async_num = async_num;
  _ACC_mpool_alloc_block(&queue->mpool);
  _ACC_gpu_calloc((void**)&queue->block_count, sizeof(unsigned));
  return queue;
}

void _ACC_queue_destroy(_ACC_queue_t* queue)
{
  if(queue == NULL) return;

  if(queue->async_num != ACC_ASYNC_SYNC){
    cudaError_t error = cudaStreamDestroy(queue->stream);
    if(error != cudaSuccess){
      _ACC_gpu_fatal(error);
    }
  }
  _ACC_mpool_free_block(queue->mpool);
  _ACC_gpu_free(queue->block_count);
  _ACC_free(queue);
}

void _ACC_queue_wait(_ACC_queue_t *queue)
{
  cudaError_t error = cudaStreamSynchronize(queue->stream);
  if(error != cudaSuccess){
    _ACC_gpu_fatal(error);
  }
}

int _ACC_queue_test(_ACC_queue_t *queue)
{
  cudaError_t error = cudaStreamQuery(queue->stream);
  if(error == cudaSuccess){
    return ~0;
  }else{
    return 0;
  }
}

void* _ACC_queue_get_mpool(_ACC_queue_t *queue)
{
  return queue->mpool;
}

unsigned* _ACC_queue_get_block_count(_ACC_queue_t *queue)
{
  return queue->block_count;
}

//XXX this function should be in other file
cudaStream_t _ACC_gpu_get_stream(int async_num)
{
  _ACC_queue_t *queue = _ACC_queue_map_get_queue(async_num);
  return queue->stream;
}
