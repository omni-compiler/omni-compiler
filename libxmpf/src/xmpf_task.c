#include "xmpf_internal.h"

_Bool xmpf_task__(_XMP_object_ref_t2 **r_desc)
{

  _XMP_object_ref_t2 *rp = *r_desc;
  int *dim_size = (int *)malloc(sizeof(int)*rp->ndims);

  for (int i = 0; i < rp->ndims; i++){
    dim_size[i] = _XMP_M_COUNT_TRIPLETi(rp->lb[i], rp->ub[i], rp->st[i]);
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(rp->ndims, rp->n_desc, false,
						       rp->lb, rp->ub, rp->st,
                                                       dim_size, true);

  free(dim_size);

  //_XMP_set_task_desc(desc, n, n->is_member, ref_nodes, ref_lower, ref_upper, ref_stride);

  if (n->is_member) {
    _XMP_push_nodes(n);
    return true;
  }
  else {
    return false;
  }

}
