#include <stdio.h>
#include "xmp_internal.h"

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
#ifdef _XMP_TCA
  _XMP_create_TCA_handle(acc_addr, array_desc);
  _XMP_create_TCA_desc(array_desc);
#endif
}

void _XMP_reflect_do_acc(_XMP_array_t *array_desc)
{
#ifdef _XMP_TCA
  _XMP_reflect_do_tca(array_desc);
#endif
}

void _XMP_reflect_acc(void *addr)
{
#ifdef _XMP_TCA
#endif
}
