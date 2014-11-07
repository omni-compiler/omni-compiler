#include "xmpf_internal.h"

void xmpf_coarray_malloc_(void **pointer, int *size, int *unit)
{
  int n_elems = *size;
  size_t elem_size = (size_t)(*unit);
  void *co_desc;
  void *co_addr;

  _XMP_coarray_malloc_info_1(n_elems, elem_size);   // in xmp_coarray_set.c
  _XMP_coarray_malloc_image_info_1();            // in xmp_coarray_set.c
  _XMP_coarray_malloc_do(&co_desc, &co_addr);    // in xmp_coarray_set.c
  *pointer = co_addr;
}



