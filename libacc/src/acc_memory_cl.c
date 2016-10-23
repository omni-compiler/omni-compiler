#include <stdio.h>
#include "acc_internal.h"
#include "acc_internal_cl.h"

/* temporal definitions */
static bool is_pagelocked(void *host_p)
{
  return false;
}
static void register_memory(void *host_p, size_t size)
{
}
static void unregister_memory(void *host_p)
{
}
static void pagelock(_ACC_memory_t *data)
{
  if(data->is_pagelocked == false && data->is_registered == false){
    register_memory(data->host_addr, data->size);
    data->is_registered = true;
  }
}
/* end of temporal definitions */

_ACC_memory_t* _ACC_memory_alloc(void *host_addr, size_t size, void *memory_object)
{
  _ACC_memory_t *memory = (_ACC_memory_t *)_ACC_alloc(sizeof(_ACC_memory_t));
  memory->host_addr = host_addr;
  if(memory_object != NULL){
    memory->memory_object = memory_object;
    memory->is_alloced = false;
  }else{
    //device memory alloc
    ////_ACC_gpu_alloc(&(memory->device_addr), size);
    cl_int ret;
    memory->memory_object = clCreateBuffer(_ACC_cl_current_context, CL_MEM_READ_WRITE, size, NULL, &ret);
    CL_CHECK(ret);
    
    memory->is_alloced = true;
  }
  memory->size = size;
  memory->ref_count = 0;

  //for memory attribute
  memory->is_pagelocked = is_pagelocked(memory->host_addr);
  memory->is_registered = false;

  return memory;
}

void _ACC_memory_free(_ACC_memory_t* memory)
{
  if(memory->is_alloced){
    //_ACC_gpu_free(memory->device_addr);
    CL_CHECK(clReleaseMemObject(memory->memory_object));
  }
  if(memory->is_registered){
    unregister_memory(memory->host_addr);
  }
  _ACC_free(memory);
}

void _ACC_cl_copy(void *host_addr, cl_mem memory_object, size_t mem_offset, size_t size, int direction, int asyncId)
{
  /* mem_offset is memory_object's offset and NOT host_addr's offset*/

  cl_bool is_blocking;
  if(asyncId == ACC_ASYNC_SYNC){
    is_blocking = CL_TRUE;
  }else{
    is_blocking = CL_FALSE;
  }

  _ACC_queue_t *queue = _ACC_queue_map_get_queue(asyncId);
  cl_command_queue command_queue = _ACC_queue_get_command_queue(queue);

  if(direction == _ACC_COPY_HOST_TO_DEVICE){
    CL_CHECK(clEnqueueWriteBuffer(command_queue, memory_object, is_blocking, mem_offset, size, host_addr, 0 /*num_wait_ev*/, NULL, NULL));
  }else if(direction == _ACC_COPY_DEVICE_TO_HOST){
    CL_CHECK(clEnqueueReadBuffer(command_queue, memory_object, is_blocking, mem_offset, size, host_addr, 0 /*num_wait_ev*/, NULL, NULL));
  }else{
    _ACC_FATAL("invalid direction\n");
  }
}

void _ACC_memory_copy(_ACC_memory_t *data, ptrdiff_t offset, size_t size, int direction, int asyncId)
{
  void *host_addr = ((char*)(data->host_addr) + offset);
  cl_mem memory_object = data->memory_object;

  if(asyncId != ACC_ASYNC_SYNC){
    pagelock(data);
  }

  _ACC_cl_copy(host_addr, memory_object, offset, size, direction, asyncId);
}

void _ACC_memory_copy_sub(_ACC_memory_t* memory, ptrdiff_t memory_offset, int direction, int isAsync,
			  size_t type_size, int dim, int pointer_dim_bit,
			  unsigned long long offsets[],
			  unsigned long long lowers[],
			  unsigned long long lengths[],
			  unsigned long long distance[])
{
  _ACC_fatal("_ACC_memory_copy_sub is unimplemented for OpenCL");
}
void _ACC_memory_copy_vector(_ACC_memory_t *data, size_t memory_offset, int direction, int asyncId, size_t type_size, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride)
{
  _ACC_fatal("_ACC_memory_copy_vector is unimplemented for OpenCL");
}


// refcount functions
void _ACC_memory_increment_refcount(_ACC_memory_t *data)
{
  ++data->ref_count;
}
void _ACC_memory_decrement_refcount(_ACC_memory_t *data)
{
  if(data->ref_count == 0){
    _ACC_fatal("ref_count is alreadly 0\n");
  }
  --data->ref_count;
}
unsigned int _ACC_memory_get_refcount(_ACC_memory_t* memory)
{
  return memory->ref_count;
}


void* _ACC_memory_get_host_addr(_ACC_memory_t* memory)
{
  return memory->host_addr;
}
size_t _ACC_memory_get_size(_ACC_memory_t* memory)
{
  return memory->size;
}
ptrdiff_t _ACC_memory_get_host_offset(_ACC_memory_t* data, void *host_addr)
{
  return (char*)data->host_addr - (char*)host_addr;
}
void* _ACC_memory_get_device_addr(_ACC_memory_t* data, ptrdiff_t offset)
{
  if(offset != 0){
    _ACC_fatal("sub memory object is unsupported now");
  }
  return data->memory_object;
}


void _ACC_memory_set_pointees(_ACC_memory_t* memory, int num_pointers, _ACC_memory_t** pointees, ptrdiff_t* pointee_offsets, void *device_pointee_pointers)
{
  _ACC_fatal("pointer data is not supported");
}

bool _ACC_memory_is_pointer(_ACC_memory_t* memory)
{
  return false;
}

_ACC_memory_t** _ACC_memory_get_pointers(_ACC_memory_t* memory)
{
  _ACC_fatal("pointer data is not supported");
  return NULL;
}

unsigned int _ACC_memory_get_num_pointees(_ACC_memory_t* memory)
{
  _ACC_fatal("pointer data is not supported");
  return 0;
}

