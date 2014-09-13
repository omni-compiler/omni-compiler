//#include "acc_internal.h"
#include "../include/acc_internal.h"
//#include "acc_gpu_internal.h"
#include "../include/acc_gpu_internal.h"
#include <stdbool.h>
#include <stdio.h>

_ACC_memory_map_t* map = NULL;

_ACC_memory_map_t* _ACC_gpu_init_data_table()
{
  _ACC_DEBUG("initialize table\n")
  map = _ACC_memory_map_create();
  return map;
}

void _ACC_gpu_finalize_data_table(_ACC_memory_map_t* map)
{
  _ACC_DEBUG("finalize table\n")
  _ACC_memory_map_destroy(map);
}

void _ACC_gpu_add_data(_ACC_gpu_data_t *host_desc)
{
  _ACC_DEBUG("add");
  _ACC_memory_map_add(map, (char*)host_desc->host_addr + host_desc->offset, host_desc->size, host_desc);
}

_Bool _ACC_gpu_remove_data(_ACC_gpu_data_t *host_desc)
{
  _ACC_DEBUG("remove");
  void *result = _ACC_memory_map_remove(map, (char*)host_desc->host_addr + host_desc->offset, host_desc->size);
  return (result != NULL)? true : false;
}

void _ACC_gpu_get_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t size)
{
  _ACC_DEBUG("get\n")
  _ACC_gpu_data_t *host_desc = (_ACC_gpu_data_t*)_ACC_memory_map_find(map, host_addr, size);
  if(host_desc == NULL){
    _ACC_DEBUG("data not found\n")
    *host_data_desc = NULL;
    *device_addr = NULL;
    return;
  }

  //make copy of host_desc
  _ACC_gpu_data_t *host_data_copy = NULL;
  host_data_copy = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
  host_data_copy->host_addr = host_addr;
  host_data_copy->device_addr = (char *)(host_desc->device_addr) + ((char *)host_addr - (char *)host_desc->host_addr);
  host_data_copy->size = size;
  host_data_copy->is_original = false;
  *host_data_desc =  host_data_copy;
  *device_addr = host_data_copy->device_addr;
}

void _ACC_gpu_get_data_sub(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t offset, size_t size)
{
  _ACC_DEBUG("get_sub\n")
  _ACC_gpu_data_t *host_desc = (_ACC_gpu_data_t*)_ACC_memory_map_find(map, (char*)host_addr + offset, size);

  if(host_desc == NULL){
    _ACC_DEBUG("data not found\n")
      *host_data_desc = NULL;
    *device_addr = NULL;
    return;
  }

  void *begin = (void*)((char*)host_addr + offset);
  void *host_begin = (void*)((char *)(host_desc->host_addr) + host_desc->offset);

  //make copy of host_desc
  _ACC_gpu_data_t *host_data_copy = NULL;
  host_data_copy = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
  host_data_copy->host_addr = host_addr;
  host_data_copy->device_addr = (host_desc->device_addr) + (begin - host_begin);
  host_data_copy->size = size;
  host_data_copy->is_original = false;
  host_data_copy->offset = offset;
  *host_data_desc =  host_data_copy;
  *device_addr = (void*)((char*)host_data_copy->device_addr - offset);
  return;
}

void _ACC_gpu_set_data_table(_ACC_memory_map_t* m)
{
  map = m;
}
