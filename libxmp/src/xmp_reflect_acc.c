#include <stdio.h>
#include "xmp_internal.h"
extern void _XMP_reflect_init_gpu(void *acc_addr, _XMP_array_t *array_desc);
extern void _XMP_reflect_do_gpu(_XMP_array_t *array_desc);

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
  _XMP_create_TCA_handle(acc_addr, array_desc);
  _XMP_create_TCA_desc(array_desc);
#else
  _XMP_reflect_init_gpu(acc_addr, array_desc);
#endif
}

void _XMP_reflect_do_acc(_XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
  _XMP_reflect_do_tca(array_desc);
#else
  _XMP_reflect_do_gpu(array_desc);
#endif
}

void _XMP_reflect_acc(void *addr)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
#endif
}

void _XMP_set_reflect_acc__(_XMP_array_t *a, int dim, int lwidth, int uwidth, int is_periodic)
{
#ifdef _XMP_TCA
  //
#else
  _XMP_set_reflect_gpu(a, dim, lwidth, uwidth, is_periodic);
#endif

}
