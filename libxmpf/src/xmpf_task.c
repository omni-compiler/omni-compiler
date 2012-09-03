#include "xmpf_internal.h"

_Bool xmpf_test_task_on__(_XMP_object_ref_t **r_desc)
{

  _XMP_object_ref_t *rp = *r_desc;

  _XMP_ASSERT(rp->ndims <= _XMP_N_ALIGN_BLOCK);
  
  int dim_size[XMPF_MAX_DIM];
  int asterisk[XMPF_MAX_DIM];

  if (rp->ref_kind == XMP_OBJ_REF_NODES){

    for (int i = 0; i < rp->ndims; i++){

      asterisk[i] = (rp->subscript_type[i] == SUBSCRIPT_ASTERISK);
      if (!asterisk[i]){
	dim_size[i] = _XMP_M_COUNT_TRIPLETi(rp->REF_LBOUND[i], rp->REF_UBOUND[i],
					    rp->REF_STRIDE[i]);
/*       xmpf_dbg_printf("lower = %d, upper = %d, stride = %d\n", */
/* 		      rp->REF_LBOUND[i], rp->REF_UBOUND[i], rp->REF_STRIDE[i]); */
      }
      else {
	dim_size[i] = 1;
      }

    }

    _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(rp->ndims, rp->n_desc, asterisk,
							 rp->REF_LBOUND, rp->REF_UBOUND,
							 rp->REF_STRIDE,
							 dim_size, true);

/*     _XMP_set_task_desc(desc, n, n->is_member, ref_nodes, */
/* 		       ref_lower, ref_upper, ref_stride); */

    //xmpf_dbg_printf("is_menber = %d\n", n->is_member);

    if (n->is_member){
      _XMP_push_nodes(n);
      return true;
    }
    else {
      return false;
    }

  }
  else { // XMP_OBJ_REF_TEMPL
    // does nothing and return false now.
    return false;
  }

}


void xmpf_end_task__(void)
{
  _XMP_pop_nodes();
}
