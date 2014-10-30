#include "xmpf_internal.h"
/*********
void xmpf_coarray_malloc__(int *size, size_t *element, void **pointer)
{
  void *co_desc;
  void *co_addr;

  fprintf(stdout, "here libxmpf/src/xmpf_coarray.c *size=%d, *element=%d\n", *size, (int)(*element));
  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_info_1(*size, *element);\n");
  _XMP_coarray_malloc_info_1(*size, *element);   // in xmp_coarray_set.c
  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_image_info_1();\n");
  _XMP_coarray_malloc_image_info_1();            // in xmp_coarray_set.c
  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_do(&co_desc, &co_addr);\n");
  _XMP_coarray_malloc_do(&co_desc, &co_addr);    // in xmp_coarray_set.c
  fprintf(stdout, "  BEFORE *pointer = co_addr;\n");
  *pointer = co_addr;
  fprintf(stdout, "finished libxmpf/src/xmpf_coarray.c\n");
}
************/

void xmpf_coarray_malloc_(void **pointer, int *size, int *unit)
{
  int n_elems = *size;
  size_t elem_size = (size_t)(*unit);

  fprintf(stdout, "here libxmpf/src/xmpf_coarray.c n_elems=%d, elem_size=%d\n", n_elems, (int)(elem_size));

  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_info_1(n_elems, elem_size);\n");
  _XMP_coarray_malloc_info_1(n_elems, elem_size);   // in xmp_coarray_set.c

  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_image_info_1();\n");
  _XMP_coarray_malloc_image_info_1();            // in xmp_coarray_set.c

  void *co_desc;
  void *co_addr;
  fprintf(stdout, "  BEFORE _XMP_coarray_malloc_do(&co_desc, &co_addr);\n");
  _XMP_coarray_malloc_do(&co_desc, &co_addr);    // in xmp_coarray_set.c

  fprintf(stdout, "  BEFORE *pointer = co_addr;\n");
  *pointer = co_addr;

  fprintf(stdout, "finished libxmpf/src/xmpf_coarray.c\n");
}



