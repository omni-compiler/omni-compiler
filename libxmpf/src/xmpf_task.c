#include "xmpf_internal.h"

void xmpf_create_task_nodes__(_XMP_nodes_t **n, _XMP_object_ref_t **r_desc)
{

  _XMP_object_ref_t *rp = *r_desc;

  _XMP_ASSERT(rp->ndims <= _XMP_N_ALIGN_BLOCK);
  
  if (rp->ref_kind == XMP_OBJ_REF_NODES){

    int ref_lower[_XMP_N_MAX_DIM];
    int ref_upper[_XMP_N_MAX_DIM];
    int ref_stride[_XMP_N_MAX_DIM];

    int asterisk[_XMP_N_MAX_DIM];
    int dim_size[_XMP_N_MAX_DIM];

    for (int i = 0; i < rp->ndims; i++){

      ref_lower[i] = rp->REF_LBOUND[i];
      ref_upper[i] = rp->REF_UBOUND[i];
      ref_stride[i] = rp->REF_STRIDE[i];

      asterisk[i] = (rp->subscript_type[i] == SUBSCRIPT_ASTERISK);

      if (!asterisk[i]){
	dim_size[i] = _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);
      }
      else {
	dim_size[i] = 1;
      }

    }

    *n = _XMP_init_nodes_struct_NODES_NAMED(rp->ndims, rp->n_desc, asterisk,
					   ref_lower, ref_upper, ref_stride,
					   dim_size, true);
  }
  else {

    long long ref_lower[_XMP_N_MAX_DIM];
    long long ref_upper[_XMP_N_MAX_DIM];
    long long ref_stride[_XMP_N_MAX_DIM];

    int asterisk[_XMP_N_MAX_DIM];

    for (int i = 0; i < rp->ndims; i++){
      ref_lower[i] = (long long)rp->REF_LBOUND[i];
      ref_upper[i] = (long long)rp->REF_UBOUND[i];
      ref_stride[i] = (long long)rp->REF_STRIDE[i];
      asterisk[i] = (rp->subscript_type[i] == SUBSCRIPT_ASTERISK);
    }

    *n = _XMP_create_nodes_by_template_ref(rp->t_desc, asterisk, ref_lower, ref_upper, ref_stride);

  }

}


_Bool xmpf_test_task_on_nodes__(_XMP_nodes_t **n)
{
  if ((*n)->is_member){
    _XMP_push_nodes(*n);
    return true;
  }
  else {
    return false;
  }
}


/* _Bool xmpf_test_task_on__(_XMP_object_ref_t **r_desc) */
/* { */

/*   _XMP_object_ref_t *rp = *r_desc; */

/*   _XMP_nodes_t *n; */

/*   _XMP_ASSERT(rp->ndims <= _XMP_N_ALIGN_BLOCK); */
  
/*   if (rp->ref_kind == XMP_OBJ_REF_NODES){ */

/*     int ref_lower[_XMP_N_MAX_DIM]; */
/*     int ref_upper[_XMP_N_MAX_DIM]; */
/*     int ref_stride[_XMP_N_MAX_DIM]; */

/*     int asterisk[_XMP_N_MAX_DIM]; */
/*     int dim_size[_XMP_N_MAX_DIM]; */

/*     for (int i = 0; i < rp->ndims; i++){ */

/*       ref_lower[i] = rp->REF_LBOUND[i]; */
/*       ref_upper[i] = rp->REF_UBOUND[i]; */
/*       ref_stride[i] = rp->REF_STRIDE[i]; */

/*       asterisk[i] = (rp->subscript_type[i] == SUBSCRIPT_ASTERISK); */

/*       if (!asterisk[i]){ */
/* 	dim_size[i] = _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]); */
/*       } */
/*       else { */
/* 	dim_size[i] = 1; */
/*       } */

/*     } */

/*     n = _XMP_init_nodes_struct_NODES_NAMED(rp->ndims, rp->n_desc, asterisk, */
/* 					   ref_lower, ref_upper, ref_stride, */
/* 					   dim_size, true); */
/*   } */
/*   else { */

/*     long long ref_lower[_XMP_N_MAX_DIM]; */
/*     long long ref_upper[_XMP_N_MAX_DIM]; */
/*     long long ref_stride[_XMP_N_MAX_DIM]; */

/*     int asterisk[_XMP_N_MAX_DIM]; */

/*     for (int i = 0; i < rp->ndims; i++){ */
/*       ref_lower[i] = (long long)rp->REF_LBOUND[i]; */
/*       ref_upper[i] = (long long)rp->REF_UBOUND[i]; */
/*       ref_stride[i] = (long long)rp->REF_STRIDE[i]; */
/*       asterisk[i] = (rp->subscript_type[i] == SUBSCRIPT_ASTERISK); */
/*     } */

/*     n = _XMP_create_nodes_by_template_ref(rp->t_desc, asterisk, ref_lower, ref_upper, ref_stride); */

/*   } */

/*   if (n->is_member){ */
/*     _XMP_push_nodes(n); */
/*     return true; */
/*   } */
/*   else { */
/*     return false; */
/*   } */

/* } */


void xmpf_end_task__(void)
{
  _XMP_pop_nodes();
}
