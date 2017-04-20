#include "xmp_internal.h"

void _XMP_ref_templ_alloc(_XMP_object_ref_t **r_desc, _XMP_template_t *t_desc, int n)
{
    _XMP_object_ref_t *rp;
    int *ip, *iq, *ir, *is;

    rp = (_XMP_object_ref_t *)_XMP_alloc(sizeof(*rp));
    ip = (int *)_XMP_alloc(sizeof(int)*n);
    iq = (int *)_XMP_alloc(sizeof(int)*n);
    ir = (int *)_XMP_alloc(sizeof(int)*n);
    is = (int *)_XMP_alloc(sizeof(int)*n);
    if (rp == NULL || ip == NULL || iq == NULL || ir == NULL) 
      _XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_TEMPL;

    rp->ndims = n ? n : t_desc->dim;

    rp->REF_OFFSET = ip;
    rp->t_desc = t_desc;
    rp->REF_INDEX = iq;
    rp->REF_STRIDE = ir;
    rp->subscript_type = is;
    for (int i = 0; i < rp->ndims; i++) rp->subscript_type[i] = SUBSCRIPT_NONE;
    *r_desc = rp;
}


void _XMP_ref_nodes_alloc(_XMP_object_ref_t **r_desc, _XMP_nodes_t *n_desc, int n)
{
    _XMP_object_ref_t *rp;
    int *ip, *iq, *ir, *is;

    rp = (_XMP_object_ref_t *)_XMP_alloc(sizeof(*rp));
    ip = (int *)_XMP_alloc(sizeof(int)*n);
    iq = (int *)_XMP_alloc(sizeof(int)*n);
    ir = (int *)_XMP_alloc(sizeof(int)*n);
    is = (int *)_XMP_alloc(sizeof(int)*n);
    if(rp == NULL || ip == NULL || iq == NULL) 
      _XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_NODES;

    rp->ndims = n ? n : n_desc->dim;

    rp->REF_OFFSET = ip;
    rp->n_desc = n_desc;
    rp->REF_INDEX = iq;
    rp->REF_STRIDE = ir;
    rp->subscript_type = is;
    for (int i = 0; i < rp->ndims; i++) rp->subscript_type[i] = SUBSCRIPT_NONE;
    *r_desc = rp;
}


void _XMP_ref_set_loop_info(_XMP_object_ref_t *rp, int i, int t_idx, int off)
{
    rp->subscript_type[i] = SUBSCRIPT_SCALAR;
    rp->REF_OFFSET[i] = off;
    rp->REF_INDEX[i] = t_idx;
}


void _XMP_ref_set_dim_info(_XMP_object_ref_t *rp, int i, int type, int lb, int ub, int st)
{
    if (type == SUBSCRIPT_SCALAR){
      rp->subscript_type[i] = type;
      rp->REF_LBOUND[i] = lb;
      rp->REF_UBOUND[i] = lb;
      rp->REF_STRIDE[i] = 1;
    }
    else {
      rp->subscript_type[i] = type;

      if (type == SUBSCRIPT_NOLB || type == SUBSCRIPT_NOLBUB){
	_XMP_ASSERT(rp->ref_kind == XMP_OBJ_REF_TEMPLATE);
	rp->REF_LBOUND[i] = rp->t_desc->info[i].ser_lower;
	rp->subscript_type[i] = SUBSCRIPT_TRIPLET;
      }
      else {
	rp->REF_LBOUND[i] = lb;
      }

      if (type == SUBSCRIPT_NOUB || type == SUBSCRIPT_NOLBUB){
	if (rp->ref_kind == XMP_OBJ_REF_NODES){
	  rp->REF_UBOUND[i] = rp->n_desc->info[i].size;
	}
	else { // XMP_OBJ_REF_TEMPLATE
	  rp->REF_UBOUND[i] = rp->t_desc->info[i].ser_upper;
	}
	rp->subscript_type[i] = SUBSCRIPT_TRIPLET;
      }
      else {
	rp->REF_UBOUND[i] = ub;
      }
      rp->REF_STRIDE[i] = st;
    }
}


void _XMP_ref_init(_XMP_object_ref_t *rp)
{
  if (rp->ref_kind == XMP_OBJ_REF_NODES){
    _XMP_nodes_t *n = rp->n_desc;
    for (int i = 0; i < rp->ndims; i++){
      if (rp->subscript_type[i] == SUBSCRIPT_NONE){
	rp->REF_LBOUND[i] = 1;
	rp->REF_UBOUND[i] = n->info[i].size;
	rp->REF_STRIDE[i] = 1;
      }
    }
  }
  else {
    _XMP_template_t *t = rp->t_desc;
    for (int i = 0; i < rp->ndims; i++){
      if (rp->subscript_type[i] == SUBSCRIPT_NONE){
	rp->REF_LBOUND[i] = t->info[i].ser_lower;
	rp->REF_UBOUND[i] = t->info[i].ser_upper;
	rp->REF_STRIDE[i] = 1;
      }
    }
  }
}


void _XMP_ref_dealloc(_XMP_object_ref_t *rp)
{
  _XMP_free(rp->REF_OFFSET);
  _XMP_free(rp->REF_INDEX);
  _XMP_free(rp->REF_STRIDE);
  _XMP_free(rp->subscript_type);
  _XMP_free(rp);
}


_Bool _XMP_is_entire(_XMP_object_ref_t *rp)
{
  if (rp->ref_kind == XMP_OBJ_REF_NODES){
    _XMP_nodes_t *n = rp->n_desc;
    for (int i = 0; i < rp->ndims; i++){
      if (rp->subscript_type[i] != SUBSCRIPT_NONE &&
	  (rp->REF_LBOUND[i] != 1 ||
	   rp->REF_UBOUND[i] != n->info[i].size ||
	   rp->REF_STRIDE[i] != 1)){
	return false;
      }
    }

    return true;
  }
  else {
    _XMP_template_t *t = rp->t_desc;
    for (int i = 0; i < rp->ndims; i++){
      if (rp->subscript_type[i] != SUBSCRIPT_NONE &&
	  (rp->REF_LBOUND[i] != t->info[i].ser_lower ||
	   rp->REF_UBOUND[i] != t->info[i].ser_upper ||
	   rp->REF_STRIDE[i] != 1)){
	return false;
      }
    }

    return true;
  }

}
