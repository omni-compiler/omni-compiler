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

  if(data->is_pointer){
    int p_offset = offset / sizeof(void*);
    for(int i = p_offset; i < data->num_pointers; i++){
      size_t pointee_size = data->pointees[i]->size;
      ptrdiff_t pointee_offset = 0; //data->memory_offsets[i] - ((char*)(data->pointees[i]->host_addr) - *((char**)(data->host_addr) + i));
      _ACC_memory_copy(data->pointees[i], pointee_offset, pointee_size, direction, asyncId);
    }
  }else{

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
}


void _ACC_memory_copy_vector(_ACC_memory_t *data, size_t memory_offset, int direction, int asyncId, size_t type_size, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride)
{
  size_t buf_size = count * blocklength * type_size;
  size_t offset_size = offset * type_size;

  //alloc buffer
  void *host_buf = (void *)_ACC_alloc(buf_size);
  void *mpool;
  long long mpool_pos = 0;
  void *dev_buf;
  _ACC_mpool_get(&mpool);
  _ACC_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);

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
  _ACC_mpool_free(dev_buf, mpool);

  _ACC_free(host_buf);
}

void _ACC_memory_copy_sub(_ACC_memory_t* memory, ptrdiff_t memory_offset, int direction, int isAsync,
			  size_t type_size, int dim, int pointer_dim_bit,
			  unsigned long long offsets[],
			  unsigned long long lowers[],
			  unsigned long long lengths[],
			  unsigned long long distance[])
{
  int i;
  void *dev_buf;
  void *host_buf = NULL;
  const char usePinnedHostBuffer = 1;
  void *host_addr = (char*)memory->host_addr - memory_offset;

  int num_top_array_dims = 0;
  //  printf("pointer_dim_bit=%d\n", pointer_dim_bit);
  for(int i = 0; i < dim; i++){
    if(i != 0 && pointer_dim_bit & (1 << i)){
      break;
    }else{
      num_top_array_dims++;
    }
  }
  //  printf("num_ldng_dims=%d\n", num_top_array_dims);

  if(num_top_array_dims != dim){
    if(num_top_array_dims != 1){
      _ACC_fatal("not supported pattern\n");
    }
    //    printf("offset=%lld, lower=%lld, length=%lld\n", offsets[0], lowers[0], lengths[0]);
    for(int i = lowers[0]-offsets[0]; i < lowers[0]-offsets[0]+lengths[0]; i++){
      _ACC_memory_copy_sub(memory->pointees[i],
			   memory->pointee_offsets[i],
			   direction,
			   isAsync,
			   type_size,
			   dim - num_top_array_dims,
			   pointer_dim_bit >> num_top_array_dims,
			   &offsets[num_top_array_dims],
			   &lowers[num_top_array_dims],
			   &lengths[num_top_array_dims],
			   &distance[num_top_array_dims]);
    }
    return;
  }
  //  printf("lower=%lld, length=%lld\n", lowers[0], lengths[0]);
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
  _ACC_mpool_get(&mpool);
  _ACC_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);
  //alloc and copy of trans_info
  unsigned long long *dev_trans_info;
  unsigned long long host_trans_info[dim * 3];
  for(int i = 0; i < dim; i++){
    host_trans_info[i + dim * 0] = lowers[i];
    host_trans_info[i + dim * 1] = lengths[i];
    host_trans_info[i + dim * 2] = distance[i];
  }
  size_t trans_info_size = dim * 3 * sizeof(unsigned long long);
  _ACC_mpool_alloc((void**)&dev_trans_info, trans_info_size, mpool, &mpool_pos);
  _ACC_gpu_copy(host_trans_info, dev_trans_info, trans_info_size, _ACC_GPU_COPY_HOST_TO_DEVICE);

  unsigned long long offset_acc = 0;
  for(i=0;i<dim;i++){
    offset_acc += offsets[i] * distance[i];
  }

  void *dev_data = _ACC_memory_get_device_addr(memory, offset_acc * type_size);
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
  _ACC_mpool_free(dev_buf, mpool);
  _ACC_mpool_free(dev_trans_info, mpool);

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

  memory->is_pointer = false;

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

void _ACC_memory_set_pointees(_ACC_memory_t* memory, int num_pointers, _ACC_memory_t** pointees, ptrdiff_t* pointee_offsets, void *device_pointee_pointers)
{
  memory->is_pointer = true;
  memory->num_pointers = (unsigned int)num_pointers;
  memory->pointees = pointees;
  memory->pointee_offsets = pointee_offsets;

  _ACC_gpu_copy(device_pointee_pointers, memory->device_addr, sizeof(void*) * num_pointers, _ACC_GPU_COPY_HOST_TO_DEVICE);
}

bool _ACC_memory_is_pointer(_ACC_memory_t* memory)
{
  return memory->is_pointer;
}

_ACC_memory_t** _ACC_memory_get_pointees(_ACC_memory_t* memory)
{
  return memory->pointees;
}

unsigned int _ACC_memory_get_num_pointees(_ACC_memory_t* memory)
{
  return memory->num_pointers;
}

