#include "xmp_internal.h"
#include "xmpf_internal.h"

void xaccf_set_reflect__(_XMP_array_t **a_desc, int *dim, int *lwidth, int *uwidth,
			 int *is_periodic)
{
  _XMP_set_reflect_acc__(*a_desc, *dim, *lwidth, *uwidth, *is_periodic);
}


void xaccf_reflect__(void *acc_addr, _XMP_array_t **a_desc)
{
  _XMP_reflect_acc__(acc_addr, *a_desc);
}

void xaccf_reflect_async__(_XMP_array_t **a_desc, int *async_id)
{
  _XMP_fatal("reflect_async is not implemented for XACC");
  //_XMP_reflect_async__(*a_desc, *async_id);
}

