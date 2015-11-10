#include <stdio.h>
#include "acc_internal.h"
#include "acc_data_struct.h"

#define INIT_DEFAULT 0
#define INIT_PRESENT 1
#define INIT_PRESENTOR 2

static void init_data(int mode, _ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]);
void _ACC_init_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_DEFAULT, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}
void _ACC_pinit_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENTOR, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}
void _ACC_find_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENT, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}

static void init_data(int mode, _ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  _ACC_data_t *host_data_d = NULL;

  // set array info
  _ACC_array_t *array_info = (_ACC_array_t *)_ACC_alloc(dim * sizeof(_ACC_array_t));
  for(int i=0;i<dim;i++){
    array_info[i].dim_offset = lower[i];//va_arg(args, int);
    if(i != 0 && array_info[i].dim_offset != 0){
      _ACC_fatal("Non-zero lower is allowed only top dimension");
    }
    array_info[i].dim_elmnts = length[i];//va_arg(args, int);
	//printf("[%llu:%llu]", array_info[i].dim_offset, array_info[i].dim_elmnts );
  }
  //printf("\n");
  unsigned long long accumulation = 1;
  for(int i=dim-1; i >= 0; i--){
    array_info[i].dim_acc = accumulation;
    accumulation *= array_info[i].dim_elmnts;
  }
  size_t size = accumulation * type_size;
  size_t offset = (dim > 0)? array_info[0].dim_offset * array_info[0].dim_acc * type_size : 0;

  _ACC_memory_t *present_data = NULL;
  //find host_data_d
  if(mode == INIT_PRESENT || mode == INIT_PRESENTOR){
    present_data = _ACC_memory_table_find((char*)addr + offset, size);
  }

  if(mode == INIT_PRESENT){
    if(present_data == NULL){
      _ACC_fatal("data not found");
    }
  }

  // alloc & init host descriptor
  host_data_d = (_ACC_data_t *)_ACC_alloc(sizeof(_ACC_data_t));
  host_data_d->host_addr = addr;
  host_data_d->offset = offset;
  host_data_d->size = size;
  host_data_d->type_size = type_size;
  host_data_d->dim = dim;
  host_data_d->array_info = array_info;

  if(present_data == NULL){
    _ACC_memory_t *memory = _ACC_memory_alloc((char*)addr + offset, size, NULL);
    if(memory == NULL){
      _ACC_fatal("failed to alloc memory");
    }
    _ACC_memory_table_add((char*)addr + offset, size, memory);
    host_data_d->memory = memory;
    host_data_d->memory_offset = offset;
  }else{
    host_data_d->memory = present_data;
    host_data_d->memory_offset = _ACC_memory_get_host_offset(present_data, addr);
  }

  _ACC_memory_increment_refcount(host_data_d->memory);

  //printf("hostaddr=%p, size=%zu, offset=%zu\n", addr, size, offset);

  // init params
  *host_data_desc = host_data_d;
  *device_addr = _ACC_memory_get_device_addr(host_data_d->memory, host_data_d->memory_offset);
}

void _ACC_finalize_data(_ACC_data_t *desc, int type) {
  //type 0:data, 1:enter data, 2:exit data

  if(type == 2){
    _ACC_memory_decrement_refcount(desc->memory);
  }

  if(type == 0 || type == 2){
    _ACC_memory_decrement_refcount(desc->memory);

    if(_ACC_memory_get_refcount(desc->memory) == 0){
      void *addr = _ACC_memory_get_host_addr(desc->memory);
      size_t size = _ACC_memory_get_size(desc->memory);
      if(_ACC_memory_table_remove(addr, size) == NULL){
	_ACC_fatal("can't remove data from data table\n");
      }
      _ACC_memory_free(desc->memory);
    }
  }

  _ACC_free(desc->array_info);
  _ACC_free(desc);
}

void _ACC_copy_data(_ACC_data_t *desc, int direction, int asyncId){
  ptrdiff_t offset = desc->memory_offset - desc->offset;
  _ACC_memory_copy(desc->memory, offset, desc->size, direction, asyncId);
}

void _ACC_pcopy_data(_ACC_data_t *desc, int direction, int asyncId){
  unsigned int refcount = _ACC_memory_get_refcount(desc->memory);
  if(refcount == 1){
    _ACC_copy_data(desc, direction, asyncId);
  }
}

void _ACC_copy_subdata(_ACC_data_t *desc, int direction, int asyncId, unsigned long long lowers[], unsigned long long lengths[]){
  int dim = desc->dim;
  _ACC_array_t *array_info = desc->array_info;

  unsigned long long count = 1;
  unsigned long long blocklength = 1;
  unsigned long long stride = 1;
  unsigned long long offset = 0;
  char isblockstride = 1;

  for(int i = 0 ; i < dim; i++){
    unsigned long long dim_elmnts = array_info[i].dim_elmnts;
    if(blocklength == 1 || (lowers[i] == 0 && lengths[i] == dim_elmnts)){
      blocklength *= lengths[i];
      stride *= dim_elmnts;
      offset = offset * dim_elmnts + lowers[i];
    }else if(count == 1){
      count = blocklength;
      blocklength = lengths[i];
      stride = dim_elmnts;
      offset = offset * dim_elmnts + lowers[i];
    }else{
      isblockstride = 0;
      break;
    }
  }
  
  if(!isblockstride){
    unsigned long long distances[8];
    for(int i = 0;i<desc->dim;i++){
      distances[i] = desc->array_info[i].dim_acc;
    }
    _ACC_memory_copy_sub(desc->memory, desc->memory_offset, direction,
			 asyncId, desc->type_size, desc->dim,
			 lowers, lengths, distances);
  }else if(count == 1){
    size_t offset_size = offset * desc->type_size;
    size_t size = blocklength * desc->type_size;
    _ACC_memory_copy(desc->memory, offset_size - desc->memory_offset, size, direction, asyncId);
  }else{
    size_t type_size = desc->type_size;
    _ACC_memory_copy_vector(desc->memory, desc->memory_offset, direction, asyncId, type_size, offset, count, blocklength, stride);
  }
}

void _ACC_gpu_map_data(void *host_addr, void* device_addr, size_t size)
{
  if(_ACC_memory_table_find(host_addr, size) != NULL){
    _ACC_fatal("map_data: already mapped\n");
  }

  _ACC_memory_t *memory = _ACC_memory_alloc(host_addr, size, device_addr);
  if(memory == NULL){
    _ACC_fatal("failed to get memory");
  }
  _ACC_memory_table_add(host_addr, size, memory);

  _ACC_memory_increment_refcount(memory);
}

void _ACC_gpu_unmap_data(void *host_addr)
{
  _ACC_memory_t* data = _ACC_memory_table_find(host_addr, 1/*size*/);

  _ACC_memory_decrement_refcount(data);

  if(_ACC_memory_get_refcount(data) > 0){
    return;
  }

  
  void *addr = _ACC_memory_get_host_addr(data);
  size_t size = _ACC_memory_get_size(data);
  if(_ACC_memory_table_remove(addr, size) == NULL){
    _ACC_fatal("can't remove data from data table\n");
  }

  _ACC_memory_free(data);
}
