#include <stdio.h>
#include "xmp_internal.h"

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
#ifdef _XMP_TCA
  printf("Aacc_addr = %p\n", acc_addr);
  int dim = array_desc->dim;
  printf("dim = %d\n", dim);

  for(int i=0;i<dim;i++){
    int left  = array_desc->info[i].shadow_size_lo;
    int right = array_desc->info[i].shadow_size_hi;
    printf("left = %d right=%d\n", left, right);
    }
#endif
}

void _XMP_reflect_do_acc(void *addr)
{
#ifdef _XMP_TCA
#endif
}

void _XMP_reflect_acc(void *addr)
{
#ifdef _XMP_TCA
#endif
}
