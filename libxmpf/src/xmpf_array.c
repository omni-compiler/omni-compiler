#include "xmpf_internal.h"

/*
 * array APIs
 */


void xmpf_array_alloc__(_XMP_array_t **a_desc, int *n_dim, int *type,
			_XMP_template_t **t_desc)
{
  _XMP_array_t *a = _XMP_alloc(sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (*n_dim - 1));

  a->is_allocated = (*t_desc)->is_owner;
  a->is_align_comm_member = false;
  a->dim = *n_dim;
  a->type = *type;
  a->type_size = _XMP_get_datatype_size(a->type);
  a->total_elmts = 0;

  a->align_comm = NULL;
  a->align_comm_size = 1;
  a->align_comm_rank = _XMP_N_INVALID_RANK;

  a->align_template = *t_desc;

  *a_desc = a;

  for (int i = 0; i < *n_dim; i++) {
    _XMP_array_info_t *ai = &(a->info[i]);

    ai->is_shadow_comm_member = false;

    ai->ser_lower = 0;
    ai->ser_upper = 0;
    ai->ser_size = 0;

    ai->shadow_type = _XMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->shadow_comm = NULL;
    ai->shadow_comm_size = 1;
    ai->shadow_comm_rank = _XMP_N_INVALID_RANK;
  }

  //xmpf_dbg_printf("xmpf_array_alloc ends\n");
}


void xmpf_align_info__(_XMP_array_t **a_desc, int *a_idx, 
		       int *lower, int *upper, int *t_idx, int *off)
{
  _XMP_array_t *a = *a_desc;
  _XMP_array_info_t *ai = &(a->info[*a_idx]);
  int t_index = *t_idx;
  _XMP_template_chunk_t *chunk;
  int tmp; /* dummy */

  ai->ser_lower = *lower;
  ai->ser_upper = *upper;
  ai->ser_size = *upper - *lower +1;

  if (t_index < 0){  /* if t_index < 0, then not aligned to any template */
    _XMP_align_array_NOT_ALIGNED(a,*a_idx);
    return;
  } 

  chunk = &(a->align_template->chunk[t_index]);
  switch (chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    _XMP_align_array_DUPLICATION(a, *a_idx, *t_idx, *off);
    break;
  case _XMP_N_DIST_BLOCK:
    _XMP_align_array_BLOCK(a, *a_idx, *t_idx, *off, &tmp);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_align_array_CYCLIC(a, *a_idx, *t_idx, *off, &tmp);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    _XMP_align_array_BLOCK_CYCLIC(a, *a_idx, *t_idx, *off, &tmp);
    break;
  default:
    _XMP_fatal("xmpf_align_array: unknown chunk dist_manner");
  }
}


void xmpf_array_init__(_XMP_array_t **a_desc)
{
  _XMP_init_array_nodes(*a_desc);

  /* debug */
  //_XMP_array_t *ap = *a_desc;
  //xmpf_dbg_printf("array ser[%d,%d] par[%d,%d]\n",
  //		  ap->info[0].ser_lower,
  //	          ap->info[0].ser_upper, 
  //		  ap->info[0].par_lower,
  //		  ap->info[0].par_upper);
}


void xmpf_array_init_shadow__(_XMP_array_t **a_desc, int *i_dim,
			      int *lshadow, int *ushadow)
{
  _XMP_array_t *array = *a_desc;
  _XMP_array_info_t *ai = &(array->info[*i_dim]);

  if (*lshadow == 0 && *ushadow == 0){
    ai->shadow_type = _XMP_N_SHADOW_NONE;
  }
  else if (*lshadow > 0 || *ushadow > 0){

    _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);

    ai->shadow_type = _XMP_N_SHADOW_NORMAL;
    ai->shadow_size_lo = *lshadow;
    ai->shadow_size_hi = *ushadow;
      
    if (array->is_allocated){
      ai->local_lower += *lshadow;
      ai->local_upper += *lshadow;
      // ai->local_stride is not changed
      ai->alloc_size += *lshadow + *ushadow;
      // ai->temp0 shuld not be used in XMP/F.
      // *(ai->temp0) -= *lshadow;
      ai->temp0_v -= *lshadow;
    }

    _XMP_create_shadow_comm(array, *i_dim);

  }
  else { // *lshadow < 0 && *ushadow < 0
    ai->shadow_type = _XMP_N_SHADOW_FULL;

    if (array->is_allocated){
      ai->shadow_size_lo = ai->par_lower - ai->ser_lower;
      ai->shadow_size_hi = ai->ser_upper - ai->par_upper;
	
      ai->local_lower = ai->par_lower;
      ai->local_upper = ai->par_upper;
      ai->local_stride = ai->par_stride;
      ai->alloc_size = ai->ser_size;
    }
    
    _XMP_create_shadow_comm(array, *i_dim);
  }
}


/* get local size */
//void xmpf_array_get_local_size__(_XMP_array_t **a_desc, int *i_dim, int *size)
//{
//  _XMP_array_t *a = *a_desc;
//  *size = a->info[*i_dim].alloc_size;
//  xmpf_dbg_printf("array_get_size=%d\n",*size);
//}
void xmpf_array_get_local_size__(_XMP_array_t **a_desc, int *i_dim, int *lb, int *ub)
{
  _XMP_array_t *array = *a_desc;
  _XMP_array_info_t *ai = &(array->info[*i_dim]);

  if (ai->align_manner != _XMP_N_ALIGN_DUPLICATION &&
      ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED){
    *lb = - ai->shadow_size_lo;
    *ub = ai->alloc_size - ai->shadow_size_lo - 1;
  }
  else {
    *lb = ai->ser_lower;
    *ub = ai->ser_upper;
  }

  //xmpf_dbg_printf("array_get_size = (%d:%d)\n", *lb, *ub);
}

void *tmp[1024];
int jjj = 0;

void xmpf_array_set_local_array__(_XMP_array_t **a_desc, void *array_addr)
{
  _XMP_array_t *a = *a_desc;

  unsigned long long total_elmts = 1;
  int dim = a->dim;
  for (int i = 0; i < dim; i++) {
    a->info[i].dim_acc = total_elmts;
    total_elmts *= a->info[i].alloc_size;
  }

  for (int i = 0; i < dim; i++) {
    _XMP_calc_array_dim_elmts(a, i);
  }
  a->total_elmts = total_elmts;

  tmp[jjj] = array_addr;
  a->array_addr_p = &tmp[jjj];
  jjj++;
}
