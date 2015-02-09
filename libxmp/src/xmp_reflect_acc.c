#include <stdio.h>
#include "xmp_internal.h"

static char useTCAHybrid = 0;

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA
  char *mode_str = getenv("USE_TCA_HYBRID");
  if(mode_str !=  NULL){
    int mode = atoi(mode_str);
    switch(mode){
    default:
    case 0:
      useTCAHybrid = 0;
      break;
    case 1:
      useTCAHybrid = 1;
      break;
    }
  }

  if (useTCAHybrid) {
    /* printf("Use TCA Hybrid reflect init\n"); */
    _XMP_create_TCA_handle(acc_addr, array_desc);
    _XMP_create_TCA_desc(array_desc);
    _XMP_reflect_init_gpu(acc_addr, array_desc);
  } else {
    /* printf("Use TCA reflect init\n"); */
    _XMP_create_TCA_handle(acc_addr, array_desc);
    _XMP_create_TCA_desc(array_desc);
  }
#else
  _XMP_reflect_init_gpu(acc_addr, array_desc);
#endif
}

void _XMP_reflect_do_acc(_XMP_array_t *array_desc)
{
  if(_XMP_world_size == 1) return;

#ifdef _XMP_TCA

  if (useTCAHybrid) {
    /* printf("Use TCA Hybrid reflect do\n"); */
    _XMP_reflect_do_tca(array_desc);
    _XMP_reflect_do_gpu(array_desc);
  } else {
    /* printf("Use TCA reflect do\n"); */
    _XMP_reflect_do_tca(array_desc);
    _XMP_reflect_do_gpu(array_desc);
  }
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
