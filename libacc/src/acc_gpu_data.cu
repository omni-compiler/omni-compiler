#include <stdio.h>
#include <stdarg.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "acc_gpu_data_struct.h"

static void register_memory(void *host_addr, size_t size);
static void unregister_memory(void *host_addr);

void _ACC_gpu_init_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t offset, size_t size) {
  _ACC_gpu_data_t *host_data_d = NULL;

  // alloc desciptors
  host_data_d = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));

  // init host descriptor
  host_data_d->host_addr = addr;

  _ACC_gpu_alloc(&(host_data_d->device_addr), size);
  //host_data_d->host_array_desc = NULL;
  //host_data_d->device_array_desc = NULL;
  host_data_d->offset = offset;
  host_data_d->size = size;
  host_data_d->is_original = true;

  
  // about pagelock
  unsigned int flags;
  cudaHostGetFlags(&flags, addr);
  cudaError_t error = cudaGetLastError();
  if(error == cudaSuccess){
    //printf("memory is pagelocked\n");
    host_data_d->is_pagelocked = true;
  }else{
    //printf("memory is not pagelocked\n");
    host_data_d->is_pagelocked = false;
  }
  host_data_d->is_registered = false;
  
  
  // init params
  *host_data_desc = host_data_d;
  *device_addr = (void *)((char*)(host_data_d->device_addr) - offset);

  
  _ACC_gpu_add_data(host_data_d);
}

void _ACC_gpu2_init_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, ...){

  va_list args;
  int i;
  _ACC_gpu_data_t *host_data_d = NULL;

  // alloc desciptors
  host_data_d = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));

  // init host descriptor
  host_data_d->host_addr = addr;
  host_data_d->type_size = type_size;
  host_data_d->dim = dim;

  // set array info
  _ACC_gpu_array_t *array_info = (_ACC_gpu_array_t *)_ACC_alloc(dim * sizeof(_ACC_gpu_array_t));
  va_start(args, dim);
  for(i=0;i<dim;i++){
    array_info[i].dim_offset = va_arg(args, int);
    if(i != 0 && array_info[i].dim_offset != 0){
      _ACC_fatal("Non-zero lower is allowed only top dimension");
    }
    array_info[i].dim_elmnts = va_arg(args, int);
  }
  va_end(args);
  int accumulation = 1;
  for(i=dim-1; i >= 0; i--){
    array_info[i].dim_acc = accumulation;
    accumulation *= array_info[i].dim_elmnts;
  }
  host_data_d->array_info = array_info;
  size_t size = accumulation * type_size;
  size_t offset = array_info[0].dim_offset * array_info[0].dim_acc * type_size;

  //device memory alloc
  _ACC_gpu_alloc(&(host_data_d->device_addr), size);
  host_data_d->offset = offset;
  host_data_d->size = size;
  host_data_d->is_original = true;

  //printf("hostaddr=%p, size=%zu, offset=%zu\n", addr, size, offset);
  // about pagelock
  unsigned int flags;
  cudaHostGetFlags(&flags, addr);
  cudaError_t error = cudaGetLastError();
  if(error == cudaSuccess){
    //printf("memory is pagelocked\n");
    host_data_d->is_pagelocked = true;
  }else{
    //printf("memory is not pagelocked\n");
    host_data_d->is_pagelocked = false;
  }
  host_data_d->is_registered = false;

  // init params
  *host_data_desc = host_data_d;
  *device_addr = (void *)((char*)(host_data_d->device_addr) - offset);

  _ACC_gpu_add_data(host_data_d);
}


void _ACC_gpu_pinit_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t offset, size_t size) {
  _ACC_gpu_data_t *host_data_d = NULL;

  _ACC_gpu_data_t *present_host_data_desc;
  void *present_device_addr;
  unsigned char is_present = 0;
  _ACC_gpu_get_data_sub(&present_host_data_desc, &present_device_addr, host_addr, offset, size);
  if(present_host_data_desc != NULL){
    is_present = 1;
  }


  // alloc desciptor
  host_data_d = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));

  // init host descriptor
  host_data_d->host_addr = host_addr;
  host_data_d->offset = offset;
  host_data_d->size = size;
  if(is_present){
    host_data_d->device_addr = (void *)((char*)(present_device_addr) + offset); //is it correct? 
    host_data_d->is_original = false;
  }else{
    _ACC_gpu_alloc(&(host_data_d->device_addr), size);
    host_data_d->is_original = true;
    _ACC_gpu_add_data(host_data_d);
  }    


  // about pagelock
  if(is_present){
    host_data_d->is_pagelocked = present_host_data_desc->is_pagelocked;
    host_data_d->is_registered = present_host_data_desc->is_registered;
  }else{
    unsigned int flags;
    cudaHostGetFlags(&flags, host_addr);
    cudaError_t error = cudaGetLastError();
    if(error == cudaSuccess){
      host_data_d->is_pagelocked = true;
    }else{
      host_data_d->is_pagelocked = false;
    }
    host_data_d->is_registered = false;
  }

  
  
  // init params
  *host_data_desc = host_data_d;
  *device_addr = (void *)((char*)(host_data_d->device_addr) - offset);
}

void _ACC_gpu_finalize_data(_ACC_gpu_data_t *desc) {
  if(desc->is_original == true){
    if(desc->is_registered == true){
      unregister_memory(desc->host_addr);
    }

    if(_ACC_gpu_remove_data(desc) == false){
      _ACC_fatal("can't remove data from data table\n");
    }
    _ACC_gpu_free(desc->device_addr);
    //desc->device_addr = NULL;
  }

  _ACC_free(desc);
}

void _ACC_gpu_copy_data(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction){
  _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset), (void*)((char *)(desc->device_addr) + offset - desc->offset), size, direction);
}

void _ACC_gpu2_copy_data_using_pack(_ACC_gpu_data_t *desc, int direction, int isAsync, int *trans_info){
  int i;
  int dim = desc->dim;
  void *dev_buf;
  void *host_buf = NULL;
  int *info_length = trans_info + dim;
  const char useAsync = 0;

  int total_elmnts = 1;
  for(i=0;i<dim;i++){
    total_elmnts *= info_length[i];
  }

  size_t buf_size = total_elmnts * desc->type_size;
  //alloc buffer
  if(useAsync){
    if(host_buf == NULL){
      cudaMallocHost((void**)&host_buf, sizeof(double)*1024*1024);
    }
  }else{
    host_buf = (void *)_ACC_alloc( buf_size);
  }

  void *mpool;
  long long mpool_pos = 0;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);
  //alloc and copy of trans_info
  int *dev_trans_info;
  size_t trans_info_size = desc->dim * 3 * sizeof(int);
  _ACC_gpu_mpool_alloc((void**)&dev_trans_info, trans_info_size, mpool, &mpool_pos);
  _ACC_gpu_copy(trans_info, dev_trans_info, trans_info_size, 400);


  if(direction == 400){
    //host to device
    _ACC_gpu_pack_data_host(host_buf, desc->host_addr, desc->dim, total_elmnts, desc->type_size, trans_info);
    if(useAsync){
      cudaMemcpyAsync(dev_buf, host_buf, buf_size, cudaMemcpyHostToDevice);
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, 400);
    }
    
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_unpack_data(dev_data, dev_buf, desc->dim, total_elmnts, desc->type_size, dev_trans_info);

    cudaThreadSynchronize();
  }else{
    //device to host
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_pack_data(dev_buf, dev_data, desc->dim, total_elmnts, desc->type_size, dev_trans_info);
    if(useAsync){
      cudaMemcpyAsync(host_buf, dev_buf, buf_size, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, 401);
    }
    _ACC_gpu_unpack_data_host(desc->host_addr, host_buf, desc->dim, total_elmnts, desc->type_size, trans_info);
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);
  _ACC_gpu_mpool_free(dev_trans_info, mpool);

  if(! useAsync){
    _ACC_free(host_buf);
  }
}
  
void _ACC_gpu2_copy_subdata(_ACC_gpu_data_t *desc, int direction, int asyncId, ...){
  int dim = desc->dim;
  int *trans_info = (int *)_ACC_alloc(dim * 3 * sizeof(int));
  int *info_lower = trans_info;
  int *info_length = trans_info + dim;
  int *info_dim_acc = trans_info + dim*2;
  int i;

  va_list args;
  va_start(args, asyncId);
  for(i=0;i<dim;i++){
    info_lower[i] = va_arg(args, int);
    info_length[i] = va_arg(args, int);
    info_dim_acc[i] = desc->array_info[i].dim_acc;
  }
  va_end(args);

  char use_packing = 0;
  {
    for(i=0;i<dim;i++) if(info_length[i] != 1) break; //skip dims that len == 1
    if(i != dim){
      for(++i; i<dim; i++) if(info_lower[i] != 0 || info_length[i] != desc->array_info[i].dim_elmnts) break;
      if(i != dim){
	use_packing = 1;
      }
    }
  }

  if(use_packing){ //pack
    _ACC_gpu2_copy_data_using_pack(desc, direction, asyncId, trans_info);
  }else{
    int total_elmnts = 1;
    int offset_elmnts = 0;
    for(int i = 0; i<dim;i++){
      total_elmnts *= info_length[i];
      offset_elmnts += info_lower[i] * desc->array_info[i].dim_acc;
    }

    //printf("total_elmnts=%d, offset_el = %d\n", total_elmnts, offset_elmnts);
    size_t offset = offset_elmnts * desc->type_size;
    size_t size = total_elmnts * desc->type_size;
    _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset), (void*)((char *)(desc->device_addr) + offset - desc->offset), size, direction);
  }
}

void _ACC_gpu2_copy_data(_ACC_gpu_data_t *desc, int direction, int asyncId)
{
  switch(asyncId){
  case ACC_ASYNC_SYNC:
    _ACC_gpu_copy((void*)((char*)(desc->host_addr)), (void*)((char *)(desc->device_addr) - desc->offset), desc->size, direction);
    break;
  case ACC_ASYNC_NOVAL:
    _ACC_gpu_copy_async_all((void*)((char*)(desc->host_addr)), (void*)((char *)(desc->device_addr) - desc->offset), desc->size, direction);
    break;
  default:
    _ACC_gpu_copy_async((void*)((char*)(desc->host_addr)), (void*)((char *)(desc->device_addr) - desc->offset), desc->size, direction, asyncId);
  }
}

void _ACC_gpu_copy_data_async_all(_ACC_gpu_data_t *desc, int direction){
  //printf("_ACC_gpu_copy_data_async_all\n");

  //pagelock if data is not pagelocked
  if(desc->is_pagelocked == false && desc->is_registered == false){
    register_memory(desc->host_addr, desc->size);
    desc->is_registered = true;
  }

  _ACC_gpu_copy_async_all(desc->host_addr, desc->device_addr, desc->size, direction);
}


void _ACC_gpu_copy_data_async(_ACC_gpu_data_t *desc, int direction, int id){
  //printf("_ACC_gpu_copy_data_async\n");

  //pagelock if data is not pagelocked
  if(desc->is_pagelocked == false && desc->is_registered == false){
    register_memory(desc->host_addr, desc->size);
    desc->is_registered = true;
  }

  _ACC_gpu_copy_async(desc->host_addr, desc->device_addr, desc->size, direction, id);
}

void _ACC_gpu_copy_data_async_default(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction){
  //pagelock if data is not pagelocked
  if(desc->is_pagelocked == false && desc->is_registered == false){
    register_memory((void*)((char*)(desc->host_addr) + desc->offset), desc->size);
    desc->is_registered = true;
  }

  _ACC_gpu_copy_async_all((void*)((char*)(desc->host_addr) + offset), (void*)((char *)(desc->device_addr) + offset - desc->offset), size, direction);
}

void _ACC_gpu_find_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t offset, size_t size) {
  //printf("finding data addr=%p, offset=%zu, size=%zu\n", addr, offset, size);
  _ACC_gpu_get_data_sub(host_data_desc, device_addr, addr, offset, size);
  if(*host_data_desc==NULL){
    _ACC_fatal("data not found");
  }
}

void _ACC_gpu2_find_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, ...){
  int i;
  va_list args;

  // set array info
  _ACC_gpu_array_t *array_info = (_ACC_gpu_array_t *)_ACC_alloc(dim * sizeof(_ACC_gpu_array_t));
  va_start(args, dim);
  for(i=0;i<dim;i++){
    array_info[i].dim_offset = va_arg(args, int);
    if(i != 0 && array_info[i].dim_offset != 0){
      _ACC_fatal("Non-zero lower is allowed only top dimension");
    }
    array_info[i].dim_elmnts = va_arg(args, int);
  }
  va_end(args);
  
  int accumulation = 1;
  for(i=dim-1; i >= 0; i--){
    array_info[i].dim_acc = accumulation;
    accumulation *= array_info[i].dim_elmnts;
  }
  size_t size = accumulation * type_size;
  size_t offset = array_info[0].dim_offset * array_info[0].dim_acc * type_size;

  _ACC_gpu_get_data_sub(host_data_desc, device_addr, addr, offset, size);
  if(*host_data_desc==NULL){
    _ACC_fatal("gpu2 data not found");
  }
  (*host_data_desc)->array_info = array_info;
  (*host_data_desc)->type_size = type_size;
  (*host_data_desc)->dim = dim;
}

static void register_memory(void *host_addr, size_t size){
  printf("register_memory\n");
  cudaError_t cuda_err = cudaHostRegister(host_addr, size, cudaHostRegisterPortable);
  if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }
}

static void unregister_memory(void *host_addr){
  printf("unregister_memory\n");
  cudaError_t cuda_err = cudaHostUnregister(host_addr);
  if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }
}

void _ACC_gpu_pcopy_data(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction){
  if(desc->is_original == true){
    _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset), (void*)((char *)(desc->device_addr) + offset - desc->offset), size, direction);
  }
}


