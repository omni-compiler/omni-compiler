/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */
#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void xmp_dbg_printf(char *fmt, ...);

void _XMP_calc_array_dim_elmts(_XMP_array_t *array, int array_index) {
  _XMP_ASSERT(array->is_allocated);

  int dim = array->dim;

  unsigned long long dim_elmts = 1;
  for (int i = 0; i < dim; i++) {
    if (i != array_index) {
      dim_elmts *= array->info[i].par_size;
    }
  }

  array->info[array_index].dim_elmts = dim_elmts;
}

void _XMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype);

void _XMP_init_array_desc(_XMP_array_t **array, _XMP_template_t *template, int dim,
                          int type, size_t type_size, ...) {
  _XMP_array_t *a = _XMP_alloc(sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (dim - 1));

  a->is_allocated = template->is_owner;
  a->is_align_comm_member = false;
  a->dim = dim;
  a->type = type;
  a->type_size = type_size;
  size_t dummy;
  _XMP_setup_reduce_type(&a->mpi_type, &dummy, type);
  a->order = MPI_ORDER_C;
  a->array_addr_p = NULL;

  a->total_elmts = 0;

  a->async_reflect = NULL;

  a->align_comm = NULL;
  a->align_comm_size = 1;
  a->align_comm_rank = _XMP_N_INVALID_RANK;

  //a->num_reqs = -1;

  //if (!template->is_fixed) _XMP_fatal("target template is not fixed");
  a->align_template = template;

  va_list args;
  va_start(args, type_size);
  for (int i = 0; i < dim; i++) {
    int size = va_arg(args, int);
    _XMP_ASSERT(size > 0);

    _XMP_array_info_t *ai = &(a->info[i]);

    ai->is_shadow_comm_member = false;

    // array lower is always 0 in C
    ai->ser_lower = 0;
    ai->ser_upper = size - 1;
    ai->ser_size = size;

    ai->shadow_type = _XMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->reflect_sched = NULL;

    ai->shadow_comm = NULL;
    ai->shadow_comm_size = 1;
    ai->shadow_comm_rank = _XMP_N_INVALID_RANK;

    ai->align_template_index = -1;
  }
  va_end(args);

  *array = a;
}

// NOTE: adesc created by this function is NOT complete. For use only in gmove_1to1.
void _XMP_init_array_desc_NOT_ALIGNED(_XMP_array_t **adesc, _XMP_template_t *template, int ndims,
				      int type, size_t type_size, unsigned long long *dim_acc, void *ap){

  _XMP_array_t *a = _XMP_alloc(sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (ndims - 1));

  a->is_allocated = template->is_owner;
  a->is_align_comm_member = false;
  a->dim = ndims;
  a->type = type;
  a->type_size = type_size;
  size_t dummy;
  _XMP_setup_reduce_type(&a->mpi_type, &dummy, type);
  a->order = MPI_ORDER_C;
  a->array_addr_p = ap;

  a->total_elmts = -1; // local (non-aligned) data

  a->async_reflect = NULL;

  a->align_comm = NULL;
  a->align_comm_size = 1;
  a->align_comm_rank = _XMP_N_INVALID_RANK;

  a->is_shrunk_template = false;

  a->align_template = template;

  for (int i = 0; i < ndims; i++) {

    _XMP_array_info_t *ai = &(a->info[i]);

    ai->is_shadow_comm_member = false;
    ai->is_regular_chunk = true;
    ai->align_manner = _XMP_N_ALIGN_NOT_ALIGNED;

    ai->ser_lower = 0;
    /* ai->ser_upper = 0; */
    /* ai->ser_size = 0; */

    /* ai->par_lower = 0 */
    /* ai->par_upper = 0 */
    /* ai->par_stride = 0; */
    /* ai->par_size = 0; */

    /* ai->local_lower = 0; */
    /* ai->local_upper = 0; */
    /* ai->local_stride = 0; */
    /* ai->alloc_size = 0; */

    ai->dim_acc = dim_acc[i];
    //ai->dim_elmts = 0;

    ai->align_subscript = 0;

    ai->shadow_type = _XMP_N_SHADOW_NONE;
    //ai->shadow_size_lo  = 0;
    //ai->shadow_size_hi  = 0;

    ai->reflect_sched = NULL;
    ai->shadow_comm = NULL;
    ai->shadow_comm_size = 1;
    ai->shadow_comm_rank = _XMP_N_INVALID_RANK;

    ai->align_template_index = _XMP_N_NO_ALIGN_TEMPLATE;

  }

  *adesc = a;

}


void _XMP_finalize_array_desc(_XMP_array_t *array) {
  int dim = array->dim;

  for (int i = 0; i < dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);

    if (ai->is_shadow_comm_member) {
      _XMP_finalize_comm(ai->shadow_comm);
    }

    _XMP_reflect_sched_t *reflect_sched;

    if ((reflect_sched = ai->reflect_sched)){

      if (reflect_sched->datatype_lo != MPI_DATATYPE_NULL){
	MPI_Type_free(&reflect_sched->datatype_lo);
      }
      if (reflect_sched->datatype_hi != MPI_DATATYPE_NULL){
	MPI_Type_free(&reflect_sched->datatype_hi);
      }

      for (int j = 0; j < 4; j++){
	if (reflect_sched->req[j] != MPI_REQUEST_NULL){
	  MPI_Request_free(&reflect_sched->req[j]);
	}
      }

      if ((_XMPF_running && i != dim -1) || (_XMPC_running && i != 0)){
	_XMP_free(reflect_sched->lo_send_buf);
	_XMP_free(reflect_sched->lo_recv_buf);
	_XMP_free(reflect_sched->hi_send_buf);
	_XMP_free(reflect_sched->hi_recv_buf);
      }

      _XMP_free(reflect_sched);
    }

  }

  _XMP_async_reflect_t *async_reflect = array->async_reflect;
  if (async_reflect){
    for (int i = 0; i < async_reflect->nreqs; i++){
      if (async_reflect->datatype[i] != MPI_DATATYPE_NULL)
	MPI_Type_free(&async_reflect->datatype[i]);
      if (async_reflect->reqs[i] != MPI_REQUEST_NULL)
	MPI_Request_free(&async_reflect->reqs[i]);
    }
    _XMP_free(async_reflect->datatype);
    _XMP_free(async_reflect->reqs);
    _XMP_free(async_reflect);
  }

/*   for (int i = 0; i < array->num_reqs; i++){ */
/*     if (array->mpi_req_shadow[i] != MPI_REQUEST_NULL){ */
/*       MPI_Request_free(&array->mpi_req_shadow[i]); */
/*     } */
/*   } */
/*   _XMP_free(array->mpi_req_shadow); */

  if (array->is_align_comm_member) {
    _XMP_finalize_comm(array->align_comm);
  }

  if (array->is_shrunk_template){
    if (array->array_nodes)
      _XMP_finalize_nodes(array->array_nodes);
  }

  _XMP_free(array);
}

void _XMP_align_array_NOT_ALIGNED(_XMP_array_t *array, int array_index) {
  _XMP_array_info_t *ai = &(array->info[array_index]);

  int lower = ai->ser_lower;
  int upper = ai->ser_upper;
  int size = ai->ser_size;

  // set members
  ai->is_regular_chunk = true;
  ai->align_manner = _XMP_N_ALIGN_NOT_ALIGNED;

  ai->par_lower = lower;
  ai->par_upper = upper;
  ai->par_stride = 1;
  ai->par_size = size;

  // must be translated to as 0-origin in XMP/F.
  //ai->local_lower = lower;
  //ai->local_upper = upper;
  ai->local_lower = 0;
  ai->local_upper = upper - lower;
  ai->local_stride = 1;
  ai->alloc_size = size;

  ai->align_subscript = 0;

  ai->align_template_index = _XMP_N_NO_ALIGN_TEMPLATE;
}

void _XMP_align_array_DUPLICATION(_XMP_array_t *array, int array_index, int template_index,
                                  long long align_subscript) {
  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_array_info_t *ai = &(array->info[array_index]);

  int lower = ai->ser_lower;
  int upper = ai->ser_upper;
  int size = ai->ser_size;

  // check range
  long long align_lower = lower + align_subscript;
  long long align_upper = upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->is_regular_chunk = true;
  ai->align_manner = _XMP_N_ALIGN_DUPLICATION;

  if (template->is_owner) {
    ai->par_lower = lower;
    ai->par_upper = upper;
    ai->par_stride = 1;
    ai->par_size = size;

    ai->local_lower = lower;
    ai->local_upper = upper;
    ai->local_stride = 1;
    ai->alloc_size = size;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
}

void _XMP_align_array_BLOCK(_XMP_array_t *array, int array_index, int template_index,
                            long long align_subscript, int *temp0) {
  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_array_info_t *ai = &(array->info[array_index]);

  // check range
  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->is_regular_chunk = (ti->ser_lower == (ai->ser_lower + align_subscript)) && chunk->is_regular_chunk;
  ai->align_manner = _XMP_N_ALIGN_BLOCK;

  if (template->is_owner) {
    long long template_lower = chunk->par_lower;
    long long template_upper = chunk->par_upper;

    // set par_lower
    if (align_lower < template_lower) {
      ai->par_lower = template_lower - align_subscript;
    }
    else if (template_upper < align_lower) {
      array->is_allocated = false;
      goto EXIT_CALC_PARALLEL_MEMBERS;
    }
    else {
      ai->par_lower = ai->ser_lower;
    }

    // set par_upper
    if (align_upper < template_lower) {
      array->is_allocated = false;
      goto EXIT_CALC_PARALLEL_MEMBERS;
    }
    else if (template_upper < align_upper) {
      ai->par_upper = template_upper - align_subscript;
    }
    else {
      ai->par_upper = ai->ser_upper;
    }

    ai->par_stride = 1;
    ai->par_size = _XMP_M_COUNT_TRIPLETi(ai->par_lower, ai->par_upper, 1);

    // array lower is always 0 in C
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    *temp0 = ai->par_lower;
    ai->temp0 = temp0;
    ai->temp0_v = *temp0;
  }

EXIT_CALC_PARALLEL_MEMBERS:

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
}

void _XMP_align_array_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                             long long align_subscript, int *temp0) {
  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_array_info_t *ai = &(array->info[array_index]);

  // check range
  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->is_regular_chunk = (ti->ser_lower == (ai->ser_lower + align_subscript)) && chunk->is_regular_chunk;
  ai->align_manner = _XMP_N_ALIGN_CYCLIC;

  if (template->is_owner) {
    int cycle = chunk->par_stride;
    //int mod = _XMP_modi_ll_i(chunk->par_lower - align_subscript, cycle);
    int rank = chunk->onto_nodes_info->rank;
    int nsize = chunk->onto_nodes_info->size;
    int rank_lb = (ai->ser_lower + align_subscript - ti->ser_lower) % nsize;
    int mod = _XMP_modi_ll_i(rank - rank_lb, nsize);

    int dist = (ai->ser_upper - (mod + ai->ser_lower)) / cycle;

    //xmpf_dbg_printf("cycle = %d, mod = %d, dist = %d, rank_lb = %d, nsize = %d\n", cycle, mod, dist, rank_lb, nsize);

    ai->par_lower = mod + ai->ser_lower;
    ai->par_upper = ai->par_lower + (dist * cycle);
    ai->par_stride = cycle;
    ai->par_size = dist + 1;

    // array lower is always 0 in C
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    *temp0 = ai->par_stride;
    ai->temp0 = temp0;
    ai->temp0_v = *temp0;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
}

void _XMP_align_array_BLOCK_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                                   long long align_subscript, int *temp0) {
  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_array_info_t *ai = &(array->info[array_index]);

  // check range
  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->is_regular_chunk = _XMP_N_INT_TRUE;
  ai->align_manner = _XMP_N_ALIGN_BLOCK_CYCLIC;

  if (template->is_owner) {

    int cycle = chunk->par_stride;
    int rank = chunk->onto_nodes_info->rank;
    int nsize = chunk->onto_nodes_info->size;

    //
    // set par_lower
    //

/*     ai->par_lower = chunk->par_lower - ti->ser_lower; */
/*     ai->par_upper = chunk->par_upper - ti->ser_lower; */
/*     ai->par_stride = chunk->par_stride; */
/*     ai->par_size = chunk->par_chunk_width; */

    /* //int rank_lb = ((ai->ser_lower + align_subscript - ti->ser_lower) / 2) % nsize; */
    /* int rank_lb = ((ai->ser_lower + align_subscript - ti->ser_lower) / chunk->par_width) % nsize; */
    /* //int mod = _XMP_modi_ll_i(rank - rank_lb, nsize); */
    /* int mod = rank + rank_lb; */

    /* int v_lower = ti->ser_lower - align_subscript; */

    /* ai->par_lower = MAX(ai->ser_lower, (long long)(mod * chunk->par_width) + v_lower); */

    /* //ai->par_lower = mod * chunk->par_width + ai->ser_lower; */

    /* xmp_dbg_printf("par_lower = %d, mod = %d, par_width = %d, ser_lower = %d\n", */
    /* 		   ai->par_lower, mod, chunk->par_width, ai->ser_lower); */

    /* // */
    /* // set par_upper and par_size */
    /* // */

    /* int dist = (ai->ser_upper - (mod * chunk->par_width + ai->ser_lower)) / cycle; */
    /* int diff = ai->ser_upper - (ai->par_lower + (dist * cycle) + (chunk->par_width - 1)); */
    /* if (diff > 0){ */
    /*   ai->par_upper = ai->par_lower + (dist * cycle) + (chunk->par_width - 1); */
    /*   ai->par_size = (dist + 1) * chunk->par_width; */
    /* } */
    /* else { */
    /*   ai->par_upper = ai->ser_upper; */
    /*   ai->par_size = (dist + 1) * chunk->par_width + diff; */
    /* } */

    int rank_dist = (ai->ser_lower + align_subscript - ti->ser_lower) / chunk->par_width;
    int v_lower = ti->ser_lower + chunk->par_width * rank_dist - align_subscript;
    int rank_lb = rank_dist % nsize;
    int mod = _XMP_modi_ll_i(rank - rank_lb, nsize);
    ai->par_lower = MAX(ai->ser_lower, (long long)(mod * chunk->par_width) + v_lower);

    /* xmp_dbg_printf("par_lower = %d, mod = %d, par_width = %d, ser_lower = %d\n", */
    /* 		   ai->par_lower, mod, chunk->par_width, ai->ser_lower); */

    //
    // set par_upper and par_size
    //

    int dist = (ai->ser_upper - (mod * chunk->par_width + v_lower)) / cycle;
    int diff = ai->ser_upper - (ai->par_lower + (dist * cycle) + (chunk->par_width - 1));
    if (diff > 0){
      ai->par_upper = ai->par_lower + (dist * cycle) + (chunk->par_width - 1);
      ai->par_size = (dist + 1) * chunk->par_width;
    }
    else {
      ai->par_upper = ai->ser_upper;
      ai->par_size = (dist + 1) * chunk->par_width + diff;
    }

    ai->par_stride = chunk->par_stride;

    //xmpf_dbg_printf("par_lower = %d, par_upper = %d, par_stride = %d, par_size = %d\n",
    //		    ai->par_lower, ai->par_upper, ai->par_stride, ai->par_size);

    // array lower is always 0 in C
    // FIXME works when only data is divided equally
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    // FIXME needs more variables
    *temp0 = ai->par_stride;
    ai->temp0 = temp0;
    ai->temp0_v = *temp0;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
}

void _XMP_align_array_GBLOCK(_XMP_array_t *array, int array_index, int template_index,
			     long long align_subscript, int *temp0) {

  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_array_info_t *ai = &(array->info[array_index]);

  // check range
  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->is_regular_chunk = (ti->ser_lower == (ai->ser_lower + align_subscript)) && chunk->is_regular_chunk;
  ai->align_manner = _XMP_N_ALIGN_GBLOCK;

  if (template->is_owner) {
    long long template_lower = chunk->par_lower;
    long long template_upper = chunk->par_upper;

    // set par_lower
    if (align_lower < template_lower) {
      ai->par_lower = template_lower - align_subscript;
    }
    else if (template_upper < align_lower) {
      array->is_allocated = false;
      goto EXIT_CALC_PARALLEL_MEMBERS;
    }
    else {
      ai->par_lower = ai->ser_lower;
    }

    // set par_upper
    if (align_upper < template_lower) {
      array->is_allocated = false;
      goto EXIT_CALC_PARALLEL_MEMBERS;
    }
    else if (template_upper < align_upper) {
      ai->par_upper = template_upper - align_subscript;
    }
    else {
      ai->par_upper = ai->ser_upper;
    }

    ai->par_stride = 1;
    ai->par_size = _XMP_M_COUNT_TRIPLETi(ai->par_lower, ai->par_upper, 1);

    // array lower is always 0 in C
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    *temp0 = ai->par_lower;
    ai->temp0 = temp0;
    ai->temp0_v = *temp0;
  }

EXIT_CALC_PARALLEL_MEMBERS:

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
}

void _XMP_alloc_array(void **array_addr, _XMP_array_t *array_desc, ...) {
  if (!array_desc->is_allocated) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, array_desc);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    array_desc->info[i].dim_acc = total_elmts;

    total_elmts *= array_desc->info[i].alloc_size;
  }
  va_end(args);

  for (int i = 0; i < dim; i++) {
    _XMP_calc_array_dim_elmts(array_desc, i);
  }

  *array_addr = _XMP_alloc(total_elmts * (array_desc->type_size));

  // set members
  array_desc->array_addr_p = *array_addr;
  array_desc->total_elmts = total_elmts;

#ifdef _XMP_TCA
  _XMP_alloc_tca(array_desc);
#endif
}

void _XMP_alloc_array2(void **array_addr, _XMP_array_t *array_desc,
		       unsigned long long *acc[]) {
  if (!array_desc->is_allocated) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;

  for (int i = dim - 1; i >= 0; i--) {
    *acc[i] = total_elmts;

    array_desc->info[i].dim_acc = total_elmts;

    total_elmts *= array_desc->info[i].alloc_size;
  }

  for (int i = 0; i < dim; i++) {
    _XMP_calc_array_dim_elmts(array_desc, i);
  }

  *array_addr = _XMP_alloc(total_elmts * (array_desc->type_size));

  // set members
  array_desc->array_addr_p = *array_addr;
  array_desc->total_elmts = total_elmts;

#ifdef _XMP_TCA
  _XMP_alloc_tca(array_desc);
#endif
}

void _XMP_alloc_array_EXTERN(void **array_addr, _XMP_array_t *array_desc, ...) {
  if (!array_desc->is_allocated) {
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, array_desc);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    array_desc->info[i].dim_acc = total_elmts;

    total_elmts *= array_desc->info[i].alloc_size;
  }
  va_end(args);

  for (int i = 0; i < dim; i++) {
    _XMP_calc_array_dim_elmts(array_desc, i);
  }

  // set members
  array_desc->array_addr_p = *array_addr;
  array_desc->total_elmts = total_elmts;
}

void _XMP_dealloc_array(_XMP_array_t *array_desc){
  _XMP_free(array_desc->array_addr_p);
}

void _XMP_init_array_addr(void **array_addr, void *init_addr,
                          _XMP_array_t *array_desc, ...) {
  if (!array_desc->is_allocated) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, array_desc);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    array_desc->info[i].dim_acc = total_elmts;

    total_elmts *= array_desc->info[i].alloc_size;
  }
  va_end(args);

  for (int i = 0; i < dim; i++) {
    _XMP_calc_array_dim_elmts(array_desc, i);
  }

  *array_addr = init_addr;

  // set members
  array_desc->array_addr_p = *array_addr;
  array_desc->total_elmts = total_elmts;
}

void _XMP_init_array_comm(_XMP_array_t *array, ...) {
  _XMP_template_t *align_template = array->align_template;
  _XMP_ASSERT(align_template->is_distributed);

  _XMP_nodes_t *onto_nodes = align_template->onto_nodes;

  int color = 1;
  if (onto_nodes->is_member) {
    int acc_nodes_size = 1;
    int template_dim = align_template->dim;

    va_list args;
    va_start(args, array);
    for (int i = 0; i < template_dim; i++) {
      _XMP_template_chunk_t *chunk = &(align_template->chunk[i]);

      int size, rank;
      if (chunk->dist_manner == _XMP_N_DIST_DUPLICATION) {
        size = 1;
        rank = 0;
      } else {
        _XMP_nodes_info_t *onto_nodes_info = chunk->onto_nodes_info;
        size = onto_nodes_info->size;
        rank = onto_nodes_info->rank;
      }

      if (va_arg(args, int) == 1) {
        color += (acc_nodes_size * rank);
      }

      acc_nodes_size *= size;
    }
    va_end(args);
  } else {
    color = 0;
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  // set members
  array->is_align_comm_member = true;

  array->align_comm = comm;
  MPI_Comm_size(*comm, &(array->align_comm_size));
  MPI_Comm_rank(*comm, &(array->align_comm_rank));
}

void _XMP_init_array_comm2(_XMP_array_t *array, int args[]) {
  _XMP_template_t *align_template = array->align_template;
  _XMP_ASSERT(align_template->is_distributed);

  _XMP_nodes_t *onto_nodes = align_template->onto_nodes;

  int color = 1;
  if (onto_nodes->is_member) {
    int acc_nodes_size = 1;
    int template_dim = align_template->dim;

    for (int i = 0; i < template_dim; i++) {
      _XMP_template_chunk_t *chunk = &(align_template->chunk[i]);

      int size, rank;
      if (chunk->dist_manner == _XMP_N_DIST_DUPLICATION) {
        size = 1;
        rank = 0;
      } else {
        _XMP_nodes_info_t *onto_nodes_info = chunk->onto_nodes_info;
        size = onto_nodes_info->size;
        rank = onto_nodes_info->rank;
      }

      if (args[i] == 1) {
        color += (acc_nodes_size * rank);
      }

      acc_nodes_size *= size;
    }
  } else {
    color = 0;
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  // set members
  array->is_align_comm_member = true;

  array->align_comm = comm;
  MPI_Comm_size(*comm, &(array->align_comm_size));
  MPI_Comm_rank(*comm, &(array->align_comm_rank));
}

void _XMP_init_array_nodes(_XMP_array_t *array) {
  _XMP_template_t *align_template = array->align_template;

  int template_dim = align_template->dim;
  int align_template_shrink[template_dim];
  int align_template_num=0;
  long long align_template_lower[template_dim];
  long long align_template_upper[template_dim];
  long long align_template_stride[template_dim];
  for (int i = 0; i < template_dim; i++) {
    align_template_shrink[i] = 1;
  }

  int array_dim = array->dim;
  for (int i = 0; i < array_dim; i++) {
    _XMP_array_info_t *info = &(array->info[i]);
    int align_template_index = info->align_template_index;
    if (align_template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
      align_template_shrink[align_template_index] = 0;
      align_template_num++;
      align_template_lower[align_template_index] = info->ser_lower + info->align_subscript;
      align_template_upper[align_template_index] = info->ser_upper + info->align_subscript;
      align_template_stride[align_template_index] = 1;
    }
  }

  if(template_dim == align_template_num){
    array->is_shrunk_template = false;
  }else if (template_dim > align_template_num){
    array->is_shrunk_template = true;
  }

  if (array->is_shrunk_template){
    array->array_nodes = _XMP_create_nodes_by_template_ref(align_template, align_template_shrink,
								     align_template_lower, align_template_upper, align_template_stride);
  }else{
    array->array_nodes =_XMP_create_temporary_nodes(align_template->onto_nodes);
  }

}

unsigned long long _XMP_get_array_total_elmts(_XMP_array_t *array) {
  if (array->is_allocated) {
    return array->total_elmts;
  }
  else {
    return 0;
  }
}


void _XMP_align_array_noalloc(_XMP_array_t *a, int adim, int tdim, long long align_subscript, 
			     int *temp0, unsigned long long *acc0){
  _XMP_ASSERT(a->dim == 1);

  a->is_allocated = false;

  _XMP_array_info_t *ai = &(a->info[adim]);

  ai->align_template_index = tdim;
  ai->align_subscript = align_subscript; /* not normalized, to be done at xmp_malloc */

  ai->temp0 = temp0;
  ai->acc = acc0; // temporarily stored to this member

  a->array_nodes = NULL;
}


int _XMP_lidx_GBLOCK(_XMP_array_t *a, int i_dim, int global_idx)
{
    int l_idx, l_base;
    _XMP_array_info_t *ai = &a->info[i_dim];
    int off = ai->align_subscript;
    int tdim = ai->align_template_index;
    int lshadow = ai->shadow_size_lo;

    _XMP_template_t *template = a->align_template;
    _XMP_template_chunk_t *template_chunk = &(template->chunk[tdim]);

    int rank = template_chunk->onto_nodes_info->rank;
    long long *mapping_array = template_chunk->mapping_array;

    //_XMP_G2L(global_idx + off, &l_idx, a->align_template, tdim);
    //_XMP_G2L(ai->par_lower + off, &l_base, a->align_template, tdim);
    l_idx = global_idx + off - mapping_array[rank];
    l_base = ai->par_lower + off -mapping_array[rank];

    l_idx = l_idx - l_base + lshadow;

    return l_idx;
}
