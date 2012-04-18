#include "xmpf_internal.h"

/* 
 * template API's
 */

void xmpf_template_alloc__(_XMP_template_t **t_desc, int *n_dim, int *is_fixed)
{
  *t_desc = _XMP_create_template_desc(*n_dim,*is_fixed); /* fixed don't care */
}


/* temporary */
static int xmpf_template_dist_manner[XMPF_MAX_DIM];
static int xmpf_template_dist_chunk[XMPF_MAX_DIM];


void xmpf_template_dim_info__(_XMP_template_t **t_desc, int *i_dim, 
			      int *lb, int *ub, 
			      int *dist_manner, int *dist_chunk)
{
  _XMP_template_t *t = *t_desc;
  t->info[*i_dim].ser_lower = *lb;
  t->info[*i_dim].ser_upper = *ub;
  xmpf_template_dist_manner[*i_dim] = *dist_manner;
  xmpf_template_dist_chunk[*i_dim] = *dist_chunk;

  /*  xmpf_dbg_printf("template: i_dim=%d, [%d,%d]\n",*i_dim,*lb,*ub);
      xmpf_dbg_printf("template??: [%lld,%lld]\n",
		  t->info[0].ser_lower,
		  t->info[0].ser_upper);*/

}


void xmpf_template_init__(_XMP_template_t **t_desc,_XMP_nodes_t  **n_desc)
{
  int t_idx, n_idx,chunk_size;

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
			    _XMP_template_t **t_desc,int *n_dim)
{
    _XMP_object_ref_t *rp;
    int *ip,*iq;
    int n = *n_dim;

    rp = (_XMP_object_ref_t *)malloc(sizeof(*rp));
    ip = (int *)malloc(sizeof(int)*n);
    iq = (int *)malloc(sizeof(int)*n);
    if(rp == NULL || ip == NULL || iq == NULL) 
	_XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_TEMPL;
    rp->offset = ip;
    rp->t_desc = *t_desc;
    rp->index = iq;
    *r_desc = rp;
}


void xmpf_ref_nodes_alloc__(_XMP_object_ref_t **r_desc,
			    _XMP_nodes_t **n_desc,int *n_dim)
{
    _XMP_object_ref_t *rp;
    int *ip,*iq;
    int n = *n_dim;

    rp = (_XMP_object_ref_t *)malloc(sizeof(*rp));
    ip = (int *)malloc(sizeof(int)*n);
    iq = (int *)malloc(sizeof(int)*n);
    if(rp == NULL || ip == NULL || iq == NULL) 
	_XMP_fatal("ref_alloc: cannot alloc memory");
    rp->ref_kind = XMP_OBJ_REF_NODES;
    rp->offset = ip;
    rp->n_desc = *n_desc;
    rp->index = iq;
    *r_desc = rp;
}


void xmpf_ref_set_info__(_XMP_object_ref_t **r_desc,int *i_dim,
			 int *t_idx,int *off)
{
    _XMP_object_ref_t *rp = *r_desc;
    int i = *i_dim;
    rp->offset[i] = *off;
    rp->index[i] = *t_idx;
}


void xmpf_ref_init__(_XMP_object_ref_t **r_desc)
{
    /* nothing at this moment */
}
