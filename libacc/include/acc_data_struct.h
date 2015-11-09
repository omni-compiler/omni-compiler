#ifndef _ACC_DATA_STRUCT
#define _ACC_DATA_STRUCT

#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>

struct _ACC_array_type{
  unsigned long long dim_offset; //offset of the dimension. same as "lower"
  unsigned long long dim_elmnts; //the number of elements of the dimension. same as "length"
  unsigned long long dim_acc; //the accumulation of lower dimensional elements
};


struct _ACC_data_type {
  void *host_addr; //pointer of var or first element of array on host
  _ACC_memory_t *memory;
  ptrdiff_t memory_offset;
  size_t offset;
  size_t size;

  //for array
  int dim; //the number of dimension
  _ACC_array_t *array_info;
  size_t type_size; //the size of an element
};

#endif //_ACC_DATA_STRUCT
