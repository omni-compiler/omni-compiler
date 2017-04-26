#include "xmpf_internal.h"

//
// reduction
//

extern void *_XMP_reduction_loc_vars[_XMP_N_MAX_LOC_VAR];
extern int _XMP_reduction_loc_types[_XMP_N_MAX_LOC_VAR];

void xmpf_reduction_loc__(int *dim, void *loc, int *datatype)
{
  _XMP_reduction_loc_vars[*dim] = loc;
  _XMP_reduction_loc_types[*dim] = *datatype;
}

void xmpf_reduction__(void *data_addr, int *count, int *datatype, int *op,
		      _XMP_object_ref_t **r_desc, int *num_locs){
  _XMP_reduction(data_addr, *count, *datatype, *op, *r_desc, *num_locs);
}

//
// bcast
//

void xmpf_bcast__(void *data_addr, int *count, int *datatype,
		  _XMP_object_ref_t **from_desc, _XMP_object_ref_t **on_desc)
{
  int size = (int)_XMP_get_datatype_size(*datatype);
  if (*on_desc && (*on_desc)->ref_kind == XMP_OBJ_REF_TEMPL)
    _XMP_bcast_on_template(data_addr, *count, size, *from_desc, *on_desc);
  else
    _XMP_bcast_on_nodes(data_addr, *count, size, *from_desc, *on_desc);
}


//
// barrier
//

void xmpf_barrier__(_XMP_object_ref_t **desc)
{
  _XMP_barrier(*desc);
}
