#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include <stdbool.h>
#include <stdio.h>

_ACC_gpu_data_list_t *list_head = NULL;
static _ACC_gpu_data_list_t* find_data(void* begin, void* end);

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
  new_data->is_pagelocked = host_desc->is_pagelocked;
  new_data->is_registered = host_desc->is_registered;

  _ACC_gpu_data_list_t *next = list_head;
  list_head = new_data;
  new_data->next = next;
}

_Bool _ACC_gpu_remove_data(void *device_addr, size_t size)
{
  _ACC_DEBUG("remove\n")
  _ACC_gpu_data_list_t *current_data;
  _ACC_gpu_data_list_t *prev_data = NULL;
  for(current_data = list_head; current_data != NULL; current_data = current_data->next){
    if(current_data->device_addr == device_addr && current_data->size == size){
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

_ACC_gpu_data_list_t* _ACC_gpu_find_data(void *host_addr, size_t offset, size_t size)
{
  _ACC_DEBUG("get_sub\n")
  void *begin = (void*)((char*)host_addr + offset);
  void *end = (void*)((char*)begin + size);

  _ACC_gpu_data_list_t *data = find_data(begin, end);
  if(data == NULL){
    _ACC_DEBUG("data not found\n")
  }
  return data;
}

static _ACC_gpu_data_list_t* find_data(void* begin, void* end)
{
  for(_ACC_gpu_data_list_t *current_data = list_head; current_data != NULL; current_data = current_data->next){
    void *host_begin = (void*)((char *)(current_data->host_addr) + current_data->offset);
    void *host_end = (void*)((char *)host_begin + current_data->size);

    if((host_begin <= begin) && (end <= host_end)){
      return current_data;
    }
  }
  return NULL;
}
