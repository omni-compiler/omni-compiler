#include "xmpf_internal.h"

/*
 * array APIs
 */

void _XMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype);

void xmpf_array_alloc__(_XMP_array_t **a_desc, int *n_dim, int *type,
			_XMP_template_t **t_desc)
{
  _XMP_array_t *a = _XMP_alloc(sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (*n_dim - 1));

  // moved to xmpf_align_info
  //a->is_allocated = (*t_desc)->is_owner;

  a->desc_kind = _XMP_DESC_ARRAY;
  
  a->is_align_comm_member = false;
  a->dim = *n_dim;
  a->type = *type;
  a->type_size = _XMP_get_datatype_size(a->type);
  size_t dummy;
  _XMP_setup_reduce_type(&a->mpi_type, &dummy, *type);
  a->order = MPI_ORDER_FORTRAN;
  a->array_addr_p = NULL;
  a->total_elmts = 0;

  a->async_reflect = NULL;

  a->align_comm = NULL;
  a->align_comm_size = 1;
  a->align_comm_rank = _XMP_N_INVALID_RANK;

  a->array_nodes = NULL;

#ifdef _XMP_MPI3_ONESIDED
  a->coarray = NULL;
#endif

  //a->num_reqs = -1;
  //a->mpi_req_shadow = _XMP_alloc(sizeof(MPI_Request) * 4 * (*n_dim));

  a->align_template = *t_desc;

  *a_desc = a;

  for (int i = 0; i < *n_dim; i++) {
    _XMP_array_info_t *ai = &(a->info[i]);

    ai->is_shadow_comm_member = false;

    ai->ser_lower = 0;
    ai->ser_upper = 0;
    ai->ser_size = 0;
    ai->par_lower = 0;
    ai->par_upper = 0;
    ai->par_stride = 0;
    ai->par_size = 0;
    ai->local_lower = 0;
    ai->local_upper = 0;
    ai->local_stride = 0;
    ai->alloc_size = 0;

    ai->shadow_type = _XMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->reflect_sched = NULL;
#ifdef _XMP_XACC
    ai->reflect_acc_sched = NULL;
#endif

    ai->shadow_comm = NULL;
    ai->shadow_comm_size = 1;
    ai->shadow_comm_rank = _XMP_N_INVALID_RANK;
  }

  //xmpf_dbg_printf("xmpf_array_alloc ends\n");
}


void xmpf_array_dealloc__(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;

#if defined(_KCOMPUTER) && defined(K_RDMA_REFLECT)
  if (a->array_addr_p) FJMPI_Rdma_dereg_mem(a->rdma_memid);
#endif

  _XMP_finalize_array_desc(a);
}


void xmpf_array_deallocate__(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;

  a->array_addr_p = NULL;
  
#if defined(_KCOMPUTER) && defined(K_RDMA_REFLECT)
  FJMPI_Rdma_dereg_mem(a->rdma_memid);
#endif

}

void xmp_f_init_allocated__(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;
  a->is_allocated = a->align_template->is_owner;
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

  if (!a->align_template->is_fixed){
    _XMP_fatal("The align-target template is not fixed");
  }

  //a->is_allocated = a->align_template->is_owner;

  chunk = &(a->align_template->chunk[t_index]);
  switch (chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    _XMP_align_array_DUPLICATION(a, *a_idx, *t_idx, (long long)(*off));
    break;
  case _XMP_N_DIST_BLOCK:
    _XMP_align_array_BLOCK(a, *a_idx, *t_idx, (long long)(*off), &tmp);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_align_array_CYCLIC(a, *a_idx, *t_idx, (long long)(*off), &tmp);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    _XMP_align_array_BLOCK_CYCLIC(a, *a_idx, *t_idx, (long long)(*off), &tmp);
    break;
  case _XMP_N_DIST_GBLOCK:
    _XMP_align_array_GBLOCK(a, *a_idx, *t_idx, (long long)(*off), &tmp);
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

    _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK || ai->align_manner == _XMP_N_ALIGN_GBLOCK);

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

    if (!ai->reflect_sched){
      _XMP_reflect_sched_t *sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
      _XMP_init_reflect_sched(sched);
      ai->reflect_sched = sched;
    }

    //_XMP_create_shadow_comm(array, *i_dim);

  }
  else { // *lshadow < 0 && *ushadow < 0
    ai->shadow_type = _XMP_N_SHADOW_FULL;

    if (array->is_allocated){
      ai->shadow_size_lo = ai->par_lower - ai->ser_lower;
      ai->shadow_size_hi = ai->ser_upper - ai->par_upper;
	
      ai->local_lower = ai->par_lower - ai->ser_lower;
      ai->local_upper = ai->par_upper - ai->ser_lower;
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


void xmpf_array_get_local_size_off__(_XMP_array_t **a_desc, int *i_dim,
				     int *size, int *off, int *blk_off)
{
  _XMP_array_t *array = *a_desc;
  _XMP_array_info_t *ai = &(array->info[*i_dim]);
  _XMP_template_t *template = array->align_template;
  _XMP_template_chunk_t *tchunk = &(template->chunk[ai->align_template_index]);

  if (template->is_owner){

    if (ai->align_manner != _XMP_N_ALIGN_DUPLICATION &&
	ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED){

      *size = ai->alloc_size;

      int lidx_on_template, template_local_lower;

      _XMP_G2L(ai->par_lower + ai->align_subscript, &lidx_on_template,
	       template, ai->align_template_index);

      //      xmpf_dbg_printf("par_lower = %d\n", ai->par_lower);

      _XMP_G2L(tchunk->par_lower, &template_local_lower,
	       template, ai->align_template_index);

      *off = lidx_on_template - template_local_lower;

      if (blk_off) *blk_off = ai->par_lower - ai->shadow_size_lo;

/*       _XMP_template_chunk_t *tchunk = &(template->chunk[ai->align_template_index]); */
/*       //_XMP_template_info_t *ti = &(template->info[ai->align_template_index]); */

/*       *size = ai->alloc_size; */
/* /\*       *off = ai->ser_lower + ai->align_subscript - ti->ser_lower *\/ */
/* /\* 	   - (ai->shadow_size_lo); *\/ */
/*       *off = ai->par_lower - tchunk->par_lower - ai->shadow_size_lo; */
    }
    else {
      //*size = ai->ser_upper - ai->ser_lower + 1;
      *size = ai->ser_upper; // in this case, the lower bound is not zero.
      //xmpf_dbg_printf("size = %d\n", *size);
      *off = ai->ser_lower; // dummy
      if (blk_off) *blk_off = ai->ser_lower; // dummy
    }
  }
  else {
    *size = 0;
    *off = 0;
    if (blk_off) *blk_off = 0;
  }    

}

#if defined(_KCOMPUTER) && defined(K_RDMA_REFLECT)
int _memid = 0;
extern int _memid;
#endif

void xmpf_array_set_local_array__(_XMP_array_t **a_desc, void *array_addr, int *is_coarray)
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

  // clear reflect schedule
  if (a->array_addr_p && a->array_addr_p != array_addr){
    for (int i = 0; i < dim; i++) {
      _XMP_reflect_sched_t *sched = a->info[i].reflect_sched;
      if (sched){
	_XMP_finalize_reflect_sched(sched, (i != dim -1));
	_XMP_init_reflect_sched(sched);
      }
    }
  }

  a->array_addr_p = array_addr;

  // for gmove in/out

#ifdef _XMP_MPI3_ONESIDED
  if (*is_coarray){
    _XMP_coarray_t *c = (_XMP_coarray_t *)_XMP_alloc(sizeof(_XMP_coarray_t));

    long asize[dim];
    for (int i = 0; i < dim; i++){
      asize[i] = a->info[dim - 1 - i].alloc_size;
    }

    _XMP_coarray_malloc_info_n(asize, dim, a->type_size);

    //_XMP_nodes_t *ndesc = a->align_template->onto_nodes;
    //_XMP_nodes_t *ndesc = _XMP_get_execution_nodes();
    _XMP_nodes_t *ndesc = _XMP_world_nodes;
    int ndims_node = ndesc->dim;
    int nsize[ndims_node-1];
    for (int i = 0; i < ndims_node-1; i++){
      nsize[i] = ndesc->info[i].size;
    }
    _XMP_coarray_malloc_image_info_n(nsize, ndims_node);

    _XMP_coarray_attach(c, array_addr, a->total_elmts * a->type_size);

    a->coarray = c;
  }
#endif

#if defined(_KCOMPUTER) && defined(K_RDMA_REFLECT)
  _memid = _memid % 511;
  a->rdma_memid = _memid;
  a->rdma_addr = FJMPI_Rdma_reg_mem(_memid++, array_addr, total_elmts * a->type_size);
#endif

}
