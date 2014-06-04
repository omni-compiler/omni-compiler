#include "xmpf_internal.h"

/* 
 * template API's
 */

void xmpf_template_alloc__(_XMP_template_t **t_desc, int *n_dim, int *is_fixed)
{
  *t_desc = _XMP_create_template_desc(*n_dim, *is_fixed);
}


void xmpf_template_dealloc__(_XMP_template_t **t_desc)
{
  _XMP_free(*t_desc);
}


/* temporary */
static int xmpf_template_dist_manner[_XMP_N_MAX_DIM];
static int xmpf_template_dist_chunk[_XMP_N_MAX_DIM];
static int *xmpf_template_mapping_array[_XMP_N_MAX_DIM];


void xmpf_template_dim_info__(_XMP_template_t **t_desc, int *i_dim, 
			      int *lb, int *ub, 
			      int *dist_manner, int *dist_chunk)
{
  _XMP_template_t *t = *t_desc;
  t->info[*i_dim].ser_lower = *lb;
  t->info[*i_dim].ser_upper = *ub;
  xmpf_template_dist_manner[*i_dim] = *dist_manner;

  if (*dist_chunk == 0 && *dist_manner == _XMP_N_DIST_CYCLIC){
    xmpf_template_dist_chunk[*i_dim] = 1;
  }
  else if (*dist_manner == _XMP_N_DIST_GBLOCK){
    xmpf_template_mapping_array[*i_dim] = dist_chunk;
  }
  else {
    xmpf_template_dist_chunk[*i_dim] = *dist_chunk;
  }

  /*  xmpf_dbg_printf("template: i_dim=%d, [%d,%d]\n",*i_dim,*lb,*ub);
      xmpf_dbg_printf("template??: [%lld,%lld]\n",
		  t->info[0].ser_lower,
		  t->info[0].ser_upper);*/

}


void xmpf_template_init__(_XMP_template_t **t_desc, _XMP_nodes_t  **n_desc)
{
  int t_idx, n_idx, chunk_size;

  _XMP_template_t *t = *t_desc;

  t->is_fixed = 1;

  _XMP_calc_template_size(t);
  _XMP_init_template_chunk(t, *n_desc);
  n_idx = 0;

  for (t_idx = 0; t_idx < t->dim; t_idx++){

    chunk_size = xmpf_template_dist_chunk[t_idx];

    switch (xmpf_template_dist_manner[t_idx]){
    case _XMP_N_DIST_DUPLICATION:
      _XMP_dist_template_DUPLICATION(t, t_idx);
      break;
    case _XMP_N_DIST_BLOCK:
      if (chunk_size == 0) 
	_XMP_dist_template_BLOCK(t, t_idx, n_idx);
      else if (chunk_size < 0)
	_XMP_fatal("chunk size is nagative in DIST_BLOCK");
      else {
	/* size must be check */
	_XMP_dist_template_BLOCK_CYCLIC(t, t_idx, n_idx, chunk_size);
      }
      n_idx++;
      break;
    case _XMP_N_DIST_CYCLIC:
      if (chunk_size == 1) _XMP_dist_template_CYCLIC(t, t_idx, n_idx);
      else if (chunk_size <= 0)
	_XMP_fatal("chunk size is nagative in DIST_CYCLIC");
      else
	_XMP_dist_template_BLOCK_CYCLIC(t, t_idx, n_idx, chunk_size);
      n_idx++;
      break;
    case _XMP_N_DIST_GBLOCK:
      {
	int *mapping_array = xmpf_template_mapping_array[t_idx];
	_XMP_dist_template_GBLOCK(t, t_idx, n_idx, mapping_array);
      }
      n_idx++;
      break;
    default:
      _XMP_fatal("unknown dist_manner");
    }

  }
    
  /* debug */
  //xmpf_dbg_printf("template: [%lld,%lld] local[%lld,%lld]\n",
  //                   t->info[0].ser_lower,
  //		   t->info[0].ser_upper,
  //		   t->chunk[0].par_lower,
  //		   t->chunk[0].par_upper);
}


void xmpf_ref_templ_alloc__(_XMP_object_ref_t **r_desc,
			    _XMP_template_t **t_desc, int *n_dim)
{
    _XMP_object_ref_t *rp;
    int *ip, *iq, *ir, *is;
    int n = *n_dim;

    rp = (_XMP_object_ref_t *)malloc(sizeof(*rp));
    ip = (int *)malloc(sizeof(int)*n);
    iq = (int *)malloc(sizeof(int)*n);
    ir = (int *)malloc(sizeof(int)*n);
    is = (int *)malloc(sizeof(int)*n);
    if (rp == NULL || ip == NULL || iq == NULL || ir == NULL) 
      _XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_TEMPL;

    rp->ndims = n ? n : (*t_desc)->dim;

    //rp->offset = ip;
    rp->REF_OFFSET = ip;
    rp->t_desc = *t_desc;
    //rp->index = iq;
    rp->REF_INDEX = iq;
    rp->REF_STRIDE = ir;
    rp->subscript_type = is;
    for (int i = 0; i < rp->ndims; i++) rp->subscript_type[i] = SUBSCRIPT_NONE;
    *r_desc = rp;
}


void xmpf_ref_nodes_alloc__(_XMP_object_ref_t **r_desc,
			    _XMP_nodes_t **n_desc, int *n_dim)
{
    _XMP_object_ref_t *rp;
    int *ip, *iq, *ir, *is;
    int n = *n_dim;

    rp = (_XMP_object_ref_t *)malloc(sizeof(*rp));
    ip = (int *)malloc(sizeof(int)*n);
    iq = (int *)malloc(sizeof(int)*n);
    ir = (int *)malloc(sizeof(int)*n);
    is = (int *)malloc(sizeof(int)*n);
    if(rp == NULL || ip == NULL || iq == NULL) 
	_XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_NODES;

    rp->ndims = n ? n : (*n_desc)->dim;

    //rp->offset = ip;
    rp->REF_OFFSET = ip;
    rp->n_desc = *n_desc;
    //rp->index = iq;
    rp->REF_INDEX = iq;
    rp->REF_STRIDE = ir;
    rp->subscript_type = is;
    for (int i = 0; i < rp->ndims; i++) rp->subscript_type[i] = SUBSCRIPT_NONE;
    *r_desc = rp;
}


/* void xmpf_ref_set_info__(_XMP_object_ref_t **r_desc,int *i_dim, */
/* 			 int *t_idx,int *off) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */
/*     int i = *i_dim; */
/*     rp->offset[i] = *off; */
/*     rp->index[i] = *t_idx; */
/* } */


void xmpf_ref_set_loop_info__(_XMP_object_ref_t **r_desc, int *i_dim,
			      int *t_idx, int *off)
{
    _XMP_object_ref_t *rp = *r_desc;
    int i = *i_dim;
/*     rp->offset[i] = *off; */
/*     rp->index[i] = *t_idx; */
    rp->subscript_type[i] = SUBSCRIPT_SCALAR;
    rp->REF_OFFSET[i] = *off;
    rp->REF_INDEX[i] = *t_idx;
}


void xmpf_ref_set_dim_info__(_XMP_object_ref_t **r_desc, int *i_dim, int *type, 
			     int *lb, int *ub, int *st)
{
    _XMP_object_ref_t *rp = *r_desc;
    int i = *i_dim;
    
    if (*type == SUBSCRIPT_SCALAR){
      rp->subscript_type[i] = *type;
      rp->REF_LBOUND[i] = *lb;
      rp->REF_UBOUND[i] = *lb;
      rp->REF_STRIDE[i] = 1;
    }
    else {

      rp->subscript_type[i] = *type;

      if (*type == SUBSCRIPT_NOLB || *type == SUBSCRIPT_NOLBUB){
	_XMP_ASSERT(rp->ref_kind == XMP_OBJ_REF_TEMPLATE);
	rp->REF_LBOUND[i] = rp->t_desc->info[i].ser_lower;
	rp->subscript_type[i] = SUBSCRIPT_TRIPLET;
      }
      else {
	rp->REF_LBOUND[i] = *lb;
      }

      if (*type == SUBSCRIPT_NOUB || *type == SUBSCRIPT_NOLBUB){
	if (rp->ref_kind == XMP_OBJ_REF_NODES){
	  rp->REF_UBOUND[i] = rp->n_desc->info[i].size;
	}
	else { // XMP_OBJ_REF_TEMPLATE
	  rp->REF_UBOUND[i] = rp->t_desc->info[i].ser_upper;
	}
	rp->subscript_type[i] = SUBSCRIPT_TRIPLET;
      }
      else {
	rp->REF_UBOUND[i] = *ub;
      }

      rp->REF_STRIDE[i] = *st;
    }

}


void xmpf_ref_init__(_XMP_object_ref_t **r_desc)
{
  _XMP_object_ref_t *rp = *r_desc;

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
