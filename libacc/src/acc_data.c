#include <stdio.h>
#include "acc_internal.h"
#include "acc_data_struct.h"
#include "acc_gpu_internal.h"

#define INIT_DEFAULT 0
#define INIT_PRESENT 1
#define INIT_PRESENTOR 2
#define INIT_DEVPTR 3

static void init_data(int mode, _ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[], int pointer_dim_bit);
void _ACC_init_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_DEFAULT, host_data_desc, device_addr, addr, type_size, dim, lower, length, pointer_dim_bit);
}
void _ACC_pinit_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENTOR, host_data_desc, device_addr, addr, type_size, dim, lower, length, pointer_dim_bit);
}
void _ACC_find_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENT, host_data_desc, device_addr, addr, type_size, dim, lower, length, pointer_dim_bit);
}
void _ACC_devptr_init_data(_ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, int pointer_dim_bit, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_DEVPTR, host_data_desc, device_addr, addr, type_size, dim, lower, length, pointer_dim_bit);
}

static _ACC_memory_t* init_memory(int mode, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[], int pointer_dim_bit, _ACC_array_t array_info[]){

  int num_top_array_dims = 0;
  for(int i = 0; i < dim; i++){
    if(i != 0 && (pointer_dim_bit & (1 << i))){
      break;
    }
    num_top_array_dims++;
  }
  
  const bool is_pointer_element = (num_top_array_dims != dim);

  //  printf("lead array dims = %d\n", num_leading_array_dims);
  if(is_pointer_element){
    //    printf("is pointer element\n");
  }else{
    //    printf("is normal element\n");
  }

  for(int i=0;i<num_top_array_dims;i++){
    array_info[i].dim_offset = lower[i];
    if(i != 0 && array_info[i].dim_offset != 0){
      _ACC_fatal("Non-zero lower is allowed only top dimension");
    }
    array_info[i].dim_elmnts = length[i];
  }

  unsigned long long accumulation = 1;
  for(int i=num_top_array_dims-1; i >= 0; i--){
    array_info[i].dim_acc = accumulation;
    accumulation *= array_info[i].dim_elmnts;
  }

  const size_t num_elements = accumulation;
  const size_t element_size = (is_pointer_element)? sizeof(void*) : type_size;
  const size_t size = accumulation * element_size;
  const size_t offset = (num_top_array_dims > 0)? array_info[0].dim_offset * array_info[0].dim_acc * element_size : 0;

  _ACC_memory_t *memory = NULL;
  if(mode == INIT_PRESENT || mode == INIT_PRESENTOR){
    memory = _ACC_memory_table_find((char*)addr + offset, size);
  }

  if(mode == INIT_PRESENT){
    if(memory == NULL){
      _ACC_fatal("data not found");
    }
  }

  // alloc & init host descriptor
  void *host_addr = (mode == INIT_DEVPTR)? NULL : addr; //host_data_d->host_addr 

  if(memory == NULL){
    memory = _ACC_memory_alloc((char*)host_addr + offset, size, (mode == INIT_DEVPTR)? addr: NULL);
    if(memory == NULL){
      _ACC_fatal("failed to alloc memory");
    }
    _ACC_memory_table_add((char*)host_addr + offset, size, memory);

    if(is_pointer_element){
      _ACC_memory_t **pointer_memories = (_ACC_memory_t**)_ACC_alloc((sizeof(_ACC_memory_t*) * num_elements));
      void **device_pointers = (void**)_ACC_alloc(sizeof(void*) * num_elements);
      ptrdiff_t *memory_offsets = (ptrdiff_t*)_ACC_alloc(sizeof(ptrdiff_t) * num_elements);
      for(int i = 0; i < num_elements; i++){
	void *sub_device_addr;
	void **element_addr = (void**)((char*)host_addr + offset + element_size * i);
	pointer_memories[i] = init_memory(mode, &sub_device_addr, *element_addr, type_size,
					  dim - num_top_array_dims,
					  &lower[num_top_array_dims],
					  &length[num_top_array_dims],
					  pointer_dim_bit >> num_top_array_dims,
					  &array_info[num_top_array_dims]);
					  
	device_pointers[i] = sub_device_addr;
	memory_offsets[i] = _ACC_memory_get_host_offset(pointer_memories[i], *element_addr);
      }
      _ACC_memory_set_pointees(memory, num_elements, pointer_memories, memory_offsets, device_pointers);
    }
  }

  _ACC_memory_increment_refcount(memory);

  const size_t memory_offset = _ACC_memory_get_host_offset(memory, addr);
  *device_addr = _ACC_memory_get_device_addr(memory, memory_offset);
  return memory;
}

static void init_data(int mode, _ACC_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[], int pointer_dim_bit){
  _ACC_data_t *host_data_d = NULL;

  // set array info
  _ACC_array_t *array_info = (_ACC_array_t *)_ACC_alloc(dim * sizeof(_ACC_array_t));

  void* dev_addr;
  _ACC_memory_t *memory = init_memory(mode, &dev_addr, addr, type_size, dim, lower, length, pointer_dim_bit, array_info);

  size_t num_elements = 1;
  for(int i=dim-1; i >= 0; i--){
    num_elements *= length[i];
  }

  size_t element_size = _ACC_memory_is_pointer(memory)? sizeof(void*) : type_size;
  size_t size = num_elements * element_size;
  size_t offset = (dim > 0)? array_info[0].dim_offset * array_info[0].dim_acc * element_size : 0;
  const ptrdiff_t memory_offset = _ACC_memory_get_host_offset(memory, addr);

  // alloc & init host descriptor
  host_data_d = (_ACC_data_t *)_ACC_alloc(sizeof(_ACC_data_t));
  host_data_d->host_addr = (mode == INIT_DEVPTR)? NULL : addr;
  host_data_d->memory = memory;
  host_data_d->memory_offset = memory_offset;
  host_data_d->offset = offset;
  host_data_d->size = size;
  host_data_d->dim = dim;
  host_data_d->array_info = array_info;
  host_data_d->type_size = type_size;
  host_data_d->pointer_dim_bit = pointer_dim_bit;

  // init params
  *host_data_desc = host_data_d;
  *device_addr = _ACC_memory_get_device_addr(memory, memory_offset);
}

void free_memory(_ACC_memory_t* const memory)
{
  if(_ACC_memory_is_pointer(memory)){
    int const num_pointees = _ACC_memory_get_num_pointees(memory);
    _ACC_memory_t** const pointees = _ACC_memory_get_pointees(memory);
    for(int i = 0; i < num_pointees; i++){
      free_memory(pointees[i]);
    }
  }

  _ACC_memory_decrement_refcount(memory);

  if(_ACC_memory_get_refcount(memory) == 0){
    void* addr = _ACC_memory_get_host_addr(memory);
    size_t size = _ACC_memory_get_size(memory);
    if(_ACC_memory_table_remove(addr, size) == NULL){
      _ACC_fatal("can't remove data from data table\n");
    }
    _ACC_memory_free(memory);
  }
}

void _ACC_finalize_data(_ACC_data_t *desc, int type) {
  //type 0:data, 1:enter data, 2:exit data

  if(type == 2){
    _ACC_memory_decrement_refcount(desc->memory);
  }

  if(type == 0 || type == 2){
    free_memory(desc->memory);
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

static
bool _ACC_is_subarray(int const num_dims, _ACC_array_t const *array_info,
		      unsigned long long starts[],
		      unsigned long long lengths[])
{
  bool is_subarray = false;
  for(int i = 0; i < num_dims; i++){
    unsigned long long const array_upper    = array_info[i].dim_offset + array_info[i].dim_elmnts;
    unsigned long long const subarray_upper = starts[i] + lengths[i];
    if(array_info[i].dim_offset <= starts[i] && subarray_upper <= array_upper){
      if(array_info[i].dim_elmnts != lengths[i]){
	is_subarray = true;
      }
    }else{
      _ACC_fatal("invalid subarray");
    }
  }
  return is_subarray;
}

void _ACC_copy_subdata(_ACC_data_t *desc, int direction, int asyncId, unsigned long long lowers[], unsigned long long lengths[]){
  int dim = desc->dim;
  _ACC_array_t *array_info = desc->array_info;

  bool const is_subarray = _ACC_is_subarray(dim, array_info, lowers, lengths);
  bool const is_pointer  = _ACC_memory_is_pointer(desc->memory);

  if( !is_subarray && !is_pointer ){
    _ACC_copy_data(desc, direction, asyncId);
    return;
  }

  if(is_pointer){
    unsigned long long distances[8];
    unsigned long long offsets[8];
    for(int i = 0; i < desc->dim; i++){
      distances[i] = desc->array_info[i].dim_acc;
      offsets[i]   = desc->array_info[i].dim_offset;
    }
    _ACC_memory_copy_sub(desc->memory, desc->memory_offset, direction,
			 asyncId, desc->type_size, desc->dim, desc->pointer_dim_bit,
			 offsets,
			 lowers, lengths, distances);
    return;
  }

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
    unsigned long long offsets[8];
    for(int i = 0;i < desc->dim; i++){
      distances[i] = desc->array_info[i].dim_acc;
      offsets[i]   = desc->array_info[i].dim_offset;
    }

    _ACC_memory_copy_sub(desc->memory, desc->memory_offset, direction,
			 asyncId, desc->type_size, desc->dim,
			 0, offsets,
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
