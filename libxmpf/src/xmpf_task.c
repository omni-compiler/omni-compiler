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

_Bool union_triplet(int lb0, int ub0, int st0, int lb1, int ub1, int st1);

_Bool xmpf_test_task_nocomm__(_XMP_object_ref_t **r_desc){

  _XMP_object_ref_t *rp = *r_desc;

  if (rp->ref_kind == XMP_OBJ_REF_NODES){

    _XMP_nodes_t *n = rp->n_desc;

    if (!n->is_member) return false;

    for (int i = 0; i < rp->ndims; i++){

      if (rp->subscript_type[i] == SUBSCRIPT_ASTERISK) continue;

      int me = n->info[i].rank + 1;

      int lb = rp->REF_LBOUND[i];
      int ub = rp->REF_UBOUND[i];
      int st = rp->REF_STRIDE[i];

      if (me < lb || ub < me) return false;
      if ((me - lb) % st != 0) return false;

    }

  }
  else {

    _XMP_template_t *t = rp->t_desc;

    if (!t->is_owner) return false;

    for (int i = 0; i < rp->ndims; i++){

      if (rp->subscript_type[i] == SUBSCRIPT_ASTERISK) continue;

      int lb = rp->REF_LBOUND[i];
      int ub = rp->REF_UBOUND[i];
      int st = rp->REF_STRIDE[i];

      _XMP_template_chunk_t *chunk = &t->chunk[i];
      long long plb = chunk->par_lower;
      long long pub = chunk->par_upper;
      int pst = chunk->par_stride;

      switch (chunk->dist_manner){

      case _XMP_N_DIST_DUPLICATION:
	break;

      case _XMP_N_DIST_BLOCK:
      case _XMP_N_DIST_GBLOCK:
	if (pub < lb || ub < plb) return false;
	break;

      case _XMP_N_DIST_CYCLIC:
	if (union_triplet(lb, ub, st, plb, pub, pst)) break;
	return false;

      case _XMP_N_DIST_BLOCK_CYCLIC:
	for (int i = 0; i < chunk->par_width; i++){
	  if (union_triplet(lb, ub, st, plb+i, pub+i, pst)) goto next;
	}
	return false;
      next:
	break;
      default:
	_XMP_fatal("xmpf_test_task_nocmm: unknown dist_manner");
      }

    }

  }

  return true;

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
