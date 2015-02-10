#include <stdio.h>
#include <assert.h>
#include "xmp_internal.h"

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
  assert(useTCAHybridFlag == 1);

  if (useTCAHybrid) {
    printf("Use TCA Hybrid reflect init\n");
    _XMP_create_TCA_handle(acc_addr, array_desc);
    _XMP_create_TCA_desc(array_desc);
    _XMP_reflect_init_gpu(acc_addr, array_desc);
  } else {
    printf("Use TCA reflect init\n");
    _XMP_create_TCA_handle(acc_addr, array_desc);
    _XMP_create_TCA_desc(array_desc);
  }
#else
  printf("Use MPI reflect init\n");
  _XMP_reflect_init_gpu(acc_addr, array_desc);
#endif
}

void _XMP_reflect_do_acc(_XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
  assert(useTCAHybridFlag == 1);

  if (useTCAHybrid) {
    printf("Use TCA Hybrid reflect do\n");
    _XMP_reflect_do_tca(array_desc);
    _XMP_reflect_do_gpu(array_desc);
  } else {
    printf("Use TCA reflect do\n");
    _XMP_reflect_do_tca(array_desc);
  }
#else
  printf("Use MPI reflect do\n");
  _XMP_reflect_do_gpu(array_desc);
#endif
}

void _XMP_reflect_acc(void *addr)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
#endif
}
