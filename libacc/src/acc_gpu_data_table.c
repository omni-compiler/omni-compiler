#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include <stdbool.h>
#include <stdio.h>

typedef struct _ACC_gpu_data_list_type{
  void *host_addr;
  void *device_addr;
  size_t size;
  size_t offset;
  struct _ACC_gpu_data_list_type *next;
}_ACC_gpu_data_list_t;

_ACC_gpu_data_list_t *list_head = NULL;

void _ACC_gpu_init_data_table()
{
  _ACC_DEBUG("initialize\n")
  if(list_head != NULL){
    _ACC_gpu_finalize_data_table(); 
  }
}

void _ACC_gpu_finalize_data_table()
{
  _ACC_DEBUG("finalize\n")
  while(list_head != NULL){
    _ACC_gpu_data_list_t *next = list_head -> next;
    _ACC_free(list_head);
    list_head = next;
  }
}

void _ACC_gpu_add_data(_ACC_gpu_data_t *host_desc)
{
  _ACC_DEBUG("add\n")
  _ACC_gpu_data_list_t *new_data = (_ACC_gpu_data_list_t *)_ACC_alloc(sizeof(_ACC_gpu_data_list_t));
  new_data->host_addr = host_desc->host_addr;
  new_data->device_addr = host_desc->device_addr;
  new_data->size = host_desc->size;
  new_data->offset = host_desc->offset;

  _ACC_gpu_data_list_t *next = list_head;
  list_head = new_data;
  new_data->next = next;
}

_Bool _ACC_gpu_remove_data(_ACC_gpu_data_t *host_desc)
{
  _ACC_DEBUG("remove\n")
  _ACC_gpu_data_list_t *current_data;
  _ACC_gpu_data_list_t *prev_data = NULL;
  for(current_data = list_head; current_data != NULL; current_data = current_data->next){
    if(current_data->device_addr == host_desc->device_addr && current_data->size == host_desc->size){
      if(prev_data == NULL){ //current is head
	list_head = current_data->next;
      }else{
	prev_data->next = current_data->next;
      }
      _ACC_free(current_data);
      return true;
    }
    prev_data = current_data;
  }
  return false;
}

void _ACC_gpu_get_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t size)
{
  _ACC_DEBUG("get\n")
  _ACC_gpu_data_list_t *current_data;
  for(current_data = list_head; current_data != NULL; current_data = current_data->next){
    void *host_begin = current_data->host_addr;
    void *host_end = (char *)host_begin + current_data->size;
    if((host_begin <= host_addr) && ((char *)host_addr + size <= (char *)host_end)){
      //make copy of host_desc
      _ACC_gpu_data_t *host_data_copy = NULL;
      host_data_copy = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
      host_data_copy->host_addr = host_addr;
      host_data_copy->device_addr = (char *)(current_data->device_addr) + ((char *)host_addr - (char *)host_begin);
      host_data_copy->size = size;
      host_data_copy->is_original = false;
      *host_data_desc =  host_data_copy;
      *device_addr = host_data_copy->device_addr;
      return;
    }
  }
  _ACC_DEBUG("data not found\n")
  *host_data_desc = NULL;
  *device_addr = NULL;
}

void _ACC_gpu_get_data_sub(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t offset, size_t size)
{
  _ACC_DEBUG("get_sub\n")
  _ACC_gpu_data_list_t *current_data;
  for(current_data = list_head; current_data != NULL; current_data = current_data->next){
    void *host_begin = (void*)((char *)(current_data->host_addr) + current_data->offset);
    void *host_end = (void*)((char *)host_begin + current_data->size);

    void *begin = (void*)((char*)host_addr + offset);
    void *end = (void*)((char*)begin + size);
    if((host_begin <= begin) && (end <= host_end)){
      //make copy of host_desc
      _ACC_gpu_data_t *host_data_copy = NULL;
      host_data_copy = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
      host_data_copy->host_addr = host_addr;
      host_data_copy->device_addr = ((char*)current_data->device_addr) + ((char*)begin - (char*)host_begin);
      host_data_copy->size = size;
      host_data_copy->is_original = false;
      host_data_copy->offset = offset;
      *host_data_desc =  host_data_copy;
      *device_addr = (void*)((char*)host_data_copy->device_addr - offset);
      return;
    }
  }
  _ACC_DEBUG("data not found\n")
  *host_data_desc = NULL;
  *device_addr = NULL;
}

