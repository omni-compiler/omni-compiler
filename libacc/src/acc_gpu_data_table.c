//#include "acc_internal.h"
#include "../include/acc_internal.h"
//#include "acc_gpu_internal.h"
#include "../include/acc_gpu_internal.h"
#include <stdbool.h>
#include <stdio.h>

//_ACC_memory_map_t* map = NULL;
typedef struct _ACC_memory_t{
  void *host_addr;
  void *dev_addr;
  size_t size;
} _ACC_memory_t;

_ACC_memory_map_t* _ACC_gpu_init_data_table()
{
  _ACC_DEBUG("initialize table\n")
  _ACC_memory_map_t* map;
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
  _ACC_memory_map_t *map = (_ACC_memory_map_t*)_ACC_gpu_get_current_memory_map();
  _ACC_memory_t *mem = (_ACC_memory_t*)_ACC_alloc(sizeof(_ACC_memory_t));
  mem->host_addr = host_desc->host_addr + host_desc->offset;
  mem->dev_addr = host_desc->device_addr;
  mem->size = host_desc->size;
  _ACC_memory_map_add(map, (char*)host_desc->host_addr + host_desc->offset, host_desc->size, mem);
  _ACC_DEBUG("add_data, map(%p), hostp(%p), devp(%p), size(%zd)", map, (char*)host_desc->host_addr + host_desc->offset, host_desc->device_addr, host_desc->size);
}

_Bool _ACC_gpu_remove_data(_ACC_gpu_data_t *host_desc)
{
  _ACC_DEBUG("remove");
  _ACC_memory_map_t *map = (_ACC_memory_map_t*)_ACC_gpu_get_current_memory_map();
  void *result = _ACC_memory_map_remove(map, (char*)host_desc->host_addr + host_desc->offset, host_desc->size);
  return (result != NULL)? true : false;
}

void _ACC_gpu_get_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *host_addr, size_t size)
{
  _ACC_fatal("_ACC_gpu_get_data is deprecated\n");
  _ACC_DEBUG("get\n")
  _ACC_memory_map_t *map = (_ACC_memory_map_t*)_ACC_gpu_get_current_memory_map();
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
  _ACC_memory_map_t *map = (_ACC_memory_map_t*)_ACC_gpu_get_current_memory_map();
  _ACC_memory_t* mem = (_ACC_memory_t*)_ACC_memory_map_find(map, (char*)host_addr + offset, size);
  
  if(mem == NULL){
    _ACC_DEBUG("data not found\n")
      *host_data_desc = NULL;
    *device_addr = NULL;
    return;
  }

  _ACC_DEBUG("found host_desc= hostp(%p), devp(%p), size(%zd)\n", mem->host_addr, mem->dev_addr, mem->size);

  void *begin = (void*)((char*)host_addr + offset);
  void *host_begin = mem->host_addr;

  //make copy of host_desc
  _ACC_gpu_data_t *host_data_copy = NULL;
  host_data_copy = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
  host_data_copy->host_addr = host_addr;
  host_data_copy->device_addr = (mem->dev_addr) + (begin - host_begin);
  host_data_copy->size = size;
  host_data_copy->is_original = false;
  host_data_copy->offset = offset;
  *host_data_desc =  host_data_copy;
  *device_addr = (void*)((char*)host_data_copy->device_addr - offset);
  return;
}

void _ACC_gpu_set_data_table(_ACC_memory_map_t* m)
{
  //  map = m;
}
