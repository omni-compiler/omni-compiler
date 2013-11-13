#include <stdio.h>
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

/*
void _ACC_gpu_copy_data(_ACC_gpu_data_t *desc, int direction){
  _ACC_gpu_copy(desc->host_addr, desc->device_addr, desc->size, direction);
}
*/
void _ACC_gpu_copy_data(_ACC_gpu_data_t *desc, size_t offset, size_t size, int direction){
  _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset), (void*)((char *)(desc->device_addr) + offset - desc->offset), size, direction);
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
  _ACC_gpu_get_data_sub(host_data_desc, device_addr, addr, offset, size);
  if(*host_data_desc==NULL){
    _ACC_fatal("data not found");
  }
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
