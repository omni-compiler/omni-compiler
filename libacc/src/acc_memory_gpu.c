#include <stdio.h>
#include <stdarg.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "acc_gpu_data_struct.h"

static void _ACC_gpu_pagelock(_ACC_memory_t *data);

void _ACC_memory_copy(_ACC_memory_t *data, ptrdiff_t offset, size_t size, int direction, int asyncId)
{
  void *host_addr = ((char*)(data->host_addr) + offset);
  void *device_addr = ((char*)(data->device_addr) + offset);

  switch(asyncId){
  case ACC_ASYNC_SYNC:
    _ACC_gpu_copy(host_addr, device_addr, size, direction);
    break;
  case ACC_ASYNC_NOVAL:
  default:
    {
      //pagelock if data is not pagelocked
      _ACC_gpu_pagelock(data);

      _ACC_gpu_copy_async(host_addr, device_addr, size, direction, asyncId);
    }
  }
}


void _ACC_memory_copy_vector(_ACC_memory_t *data, size_t memory_offset, int direction, int asyncId, size_t type_size, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride)
{
  //this function will be the below.
  //void _ACC_memory_copy_vector(_ACC_gpu_data_t *desc, int direction, int asyncId, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride);

  size_t buf_size = count * blocklength * type_size;
  size_t offset_size = offset * type_size;

  //alloc buffer
  void *host_buf = (void *)_ACC_alloc(buf_size);
  void *mpool;
  long long mpool_pos = 0;
  void *dev_buf;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);

  ////
  void *dev_data = (char*)data->device_addr + memory_offset + offset_size;
  void *host_data = (char*)data->host_addr + memory_offset + offset_size;

  if(direction == _ACC_GPU_COPY_HOST_TO_DEVICE){
    //host to device
    _ACC_pack_vector(host_buf, host_data, count, blocklength, stride, type_size);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_HOST_TO_DEVICE, asyncId);
    _ACC_gpu_unpack_vector(dev_data, dev_buf, count, blocklength, stride, type_size, asyncId);
    _ACC_gpu_wait(asyncId);
  }else if(direction == _ACC_GPU_COPY_DEVICE_TO_HOST){
    //device to host
    _ACC_gpu_pack_vector(dev_buf, dev_data, count, blocklength, stride, type_size, asyncId);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_DEVICE_TO_HOST, asyncId);
    _ACC_gpu_wait(asyncId);
    _ACC_unpack_vector(host_data, host_buf, count, blocklength, stride, type_size);
  }else{
    _ACC_fatal("bad direction");
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);

  _ACC_free(host_buf);
}

void _ACC_memory_copy_sub(_ACC_memory_t* memory, ptrdiff_t memory_offset, int direction, int isAsync, size_t type_size, int dim, unsigned long long lowers[], unsigned long long lengths[], unsigned long long distance[]){
  int i;
  void *dev_buf;
  void *host_buf = NULL;
  const char usePinnedHostBuffer = 1;
  void *host_addr = (char*)memory->host_addr - memory_offset;

  unsigned long long total_elmnts = 1;
  for(i=0;i<dim;i++){
    total_elmnts *= lengths[i];
  }

  size_t buf_size = total_elmnts * type_size;
  //alloc buffer
  if(usePinnedHostBuffer){
    host_buf = _ACC_alloc_pinned(buf_size);
  }else{
    host_buf = _ACC_alloc( buf_size);
  }

  void *mpool;
  long long mpool_pos = 0;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);
  //alloc and copy of trans_info
  unsigned long long *dev_trans_info;
  unsigned long long host_trans_info[dim * 3];
  for(int i = 0; i < dim; i++){
    host_trans_info[i + dim * 0] = lowers[i];
    host_trans_info[i + dim * 1] = lengths[i];
    host_trans_info[i + dim * 2] = distance[i];
  }
  size_t trans_info_size = dim * 3 * sizeof(unsigned long long);
  _ACC_gpu_mpool_alloc((void**)&dev_trans_info, trans_info_size, mpool, &mpool_pos);
  _ACC_gpu_copy(host_trans_info, dev_trans_info, trans_info_size, _ACC_GPU_COPY_HOST_TO_DEVICE);


  void *dev_data = _ACC_memory_get_device_addr(memory, memory_offset);
  if(direction == _ACC_GPU_COPY_HOST_TO_DEVICE){
    //host to device
    _ACC_gpu_pack_data_host(host_buf, host_addr, dim, total_elmnts, type_size, host_trans_info);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_HOST_TO_DEVICE, isAsync);
    _ACC_gpu_unpack_data(dev_data, dev_buf, dim, total_elmnts, type_size, dev_trans_info, isAsync);
    _ACC_gpu_wait(isAsync);
  }else if(direction == _ACC_GPU_COPY_DEVICE_TO_HOST){
    //device to host
    _ACC_gpu_pack_data(dev_buf, dev_data, dim, total_elmnts, type_size, dev_trans_info, isAsync);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_DEVICE_TO_HOST, isAsync);
    _ACC_gpu_wait(isAsync);
    _ACC_gpu_unpack_data_host(host_addr, host_buf, dim, total_elmnts, type_size, host_trans_info);
  }else{
    _ACC_fatal("bad direction");
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);
  _ACC_gpu_mpool_free(dev_trans_info, mpool);

  if(usePinnedHostBuffer){
    _ACC_free_pinned(host_buf);
  }else{
    _ACC_free(host_buf);
  }
}


void _ACC_memory_free(_ACC_memory_t* memory)
{
  if(memory->is_alloced){
    _ACC_gpu_free(memory->device_addr);
  }
  if(memory->is_registered){
    _ACC_gpu_unregister_memory(memory->host_addr);
  }
  _ACC_free(memory);
}

_ACC_memory_t* _ACC_memory_alloc(void *host_addr, size_t size, void *device_addr)
{
  _ACC_memory_t *memory = (_ACC_memory_t *)_ACC_alloc(sizeof(_ACC_memory_t));
  memory->host_addr = host_addr;
  if(device_addr != NULL){
    memory->device_addr = device_addr;
    memory->is_alloced = false;
  }else{
    //device memory alloc
    _ACC_gpu_alloc(&(memory->device_addr), size);
    memory->is_alloced = true;
  }
  memory->size = size;
  memory->ref_count = 0;

  //for CUDA memory attribute
  memory->is_pagelocked = _ACC_gpu_is_pagelocked(memory->host_addr);
  memory->is_registered = false;

  return memory;
}

void* _ACC_memory_get_device_addr(_ACC_memory_t* data, ptrdiff_t offset)
{
  return (char*)data->device_addr - offset;
}

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

static void _ACC_gpu_pagelock(_ACC_memory_t *data)
{
  if(data->is_pagelocked == false && data->is_registered == false){
    _ACC_gpu_register_memory(data->host_addr, data->size);
    data->is_registered = true;
  }
}

ptrdiff_t _ACC_memory_get_host_offset(_ACC_memory_t* data, void *host_addr)
{
  return (char*)data->host_addr - (char*)host_addr;
}

void* _ACC_memory_get_host_addr(_ACC_memory_t* memory)
{
  return memory->host_addr;
}
size_t _ACC_memory_get_size(_ACC_memory_t* memory)
{
  return memory->size;
}
unsigned int _ACC_memory_get_refcount(_ACC_memory_t* memory)
{
  return memory->ref_count;
}
