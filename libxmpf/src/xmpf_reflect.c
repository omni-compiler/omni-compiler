#include "xmpf_internal.h"

void xmpf_set_reflect__(_XMP_array_t **a_desc, int *dim, int *lwidth, int *uwidth,
			int *is_periodic)
{
  _XMP_set_reflect__(*a_desc, *dim, *lwidth, *uwidth, *is_periodic);
}


void xmpf_reflect__(_XMP_array_t **a_desc)
{
  _XMP_reflect__(*a_desc);
}


void xmpf_reflect_async__(_XMP_array_t **a_desc, int *async_id)
{
  _XMP_reflect_async__(*a_desc, *async_id);
}
