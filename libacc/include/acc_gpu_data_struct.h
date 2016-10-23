#ifndef _ACC_GPU_DATA_STRUCT
#define _ACC_GPU_DATA_STRUCT

#include <stdbool.h>
#include <stddef.h>

struct _ACC_memory_type{
  void *host_addr;
  void *device_addr;
  size_t size;
  bool is_alloced;
  
  bool is_pagelocked;
  bool is_registered;
  unsigned int ref_count;
  bool is_pointer;
  unsigned int num_pointers;
  struct _ACC_memory_type **pointees;
  ptrdiff_t *pointee_offsets;
};

#endif //_ACC_GPU_DATA_STRUCT
