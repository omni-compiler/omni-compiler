#include <stdio.h>
#include "acc_internal.h"
#include "acc_internal_cl.h"

struct _ACC_queue_type{
  cl_command_queue command_queue;
  cl_event last_event;
  void *mpool;
  unsigned *block_count;
};

_ACC_queue_t* _ACC_queue_create(int async_num)
{
  _ACC_DEBUG("queue create\n")
  _ACC_queue_t *queue = (_ACC_queue_t *)_ACC_alloc(sizeof(_ACC_queue_t));

  cl_int ret;
  queue->command_queue = clCreateCommandQueue(_ACC_cl_current_context, _ACC_cl_device_ids[_ACC_cl_device_num], 0 /*prop*/, &ret);
  CL_CHECK(ret);

  queue->last_event = NULL;

  _ACC_mpool_alloc_block(&queue->mpool);
  queue->block_count = NULL; //FIXME do something such as _ACC_gpu_calloc((void**)&queue->block_count, sizeof(unsigned));
  return queue;
}

void _ACC_queue_destroy(_ACC_queue_t* queue)
{
  if(queue == NULL) return;

  CL_CHECK(clFlush(queue->command_queue));
  CL_CHECK(clFinish(queue->command_queue));
  CL_CHECK(clReleaseCommandQueue(queue->command_queue));

  //FIXME do something such as _ACC_gpu_mpool_free_block(queue->mpool);
  //FIXME do something such as _ACC_gpu_free(queue->block_count);
  _ACC_free(queue);
}

void _ACC_queue_wait(_ACC_queue_t *queue)
{
  //XXX is clFlush need?
  CL_CHECK(clFlush(queue->command_queue));
  CL_CHECK(clFinish(queue->command_queue));
}

int _ACC_queue_test(_ACC_queue_t *queue)
{
  cl_int status;
  CL_CHECK(clGetEventInfo(queue->last_event,
			  CL_EVENT_COMMAND_EXECUTION_STATUS,
			  sizeof(cl_int),
			  &status,
			  NULL));
  
  if(status == CL_COMPLETE){
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

void _ACC_queue_set_last_event(_ACC_queue_t* queue, cl_event event)
{
  if(queue->last_event != NULL){
    clReleaseEvent(queue->last_event);
  }
  queue->last_event = event;
}

cl_command_queue _ACC_queue_get_command_queue(_ACC_queue_t *queue)
{
  return queue->command_queue;
}

cl_command_queue acc_get_opencl_queue(int async_num)
{
  _ACC_queue_t *queue = _ACC_queue_map_get_queue(async_num);
  return _ACC_queue_get_command_queue(queue);
}
void acc_set_opencl_queue(int async, cl_command_queue command_queue)
{
  _ACC_queue_t *queue = (_ACC_queue_t *)_ACC_alloc(sizeof(_ACC_queue_t));
  queue->command_queue;
  queue->last_event = NULL;
  queue->mpool = NULL;
  queue->block_count = NULL;

  _ACC_queue_map_set_queue(async, queue);
}
