/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include "mpi.h"
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

static void _XMP_calc_array_dim_elmts(_XMP_array_t *array, int array_index) {
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

void _XMP_init_array_desc(_XMP_array_t **array, _XMP_template_t *template, int dim,
                          int type, size_t type_size, ...) {
  _XMP_array_t *a = _XMP_alloc(sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (dim - 1));

  a->is_allocated = template->is_owner;
  a->is_align_comm_member = false;
  a->dim = dim;
  a->type = type;
  a->type_size = type_size;

  a->addr = NULL;
  a->total_elmts = 0;

  a->align_comm = NULL;
  a->align_comm_size = 1;
  a->align_comm_rank = _XMP_N_INVALID_RANK;

  a->align_template = template;

  va_list args;
  va_start(args, type_size);
  for (int i = 0; i < dim; i++) {
    int size = va_arg(args, int);
    _XMP_ASSERT(size > 0);

    _XMP_array_info_t *ai = &(a->info[i]);

    ai->is_shadow_comm_member = false;

    // XXX array lower is always 0 in C
    ai->ser_lower = 0;
    ai->ser_upper = size - 1;
    ai->ser_size = size;

    ai->shadow_type = _XMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->shadow_comm = NULL;
    ai->shadow_comm_size = 1;
    ai->shadow_comm_rank = _XMP_N_INVALID_RANK;
  }
  va_end(args);

  *array = a;
}

void _XMP_finalize_array_desc(_XMP_array_t *array) {
  int dim = array->dim;
  for (int i = 0; i < dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);

    if (ai->is_shadow_comm_member) {
      _XMP_finalize_comm(ai->shadow_comm);
    }
  }

  if (array->is_align_comm_member) {
    _XMP_finalize_comm(array->align_comm);
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

  ai->local_lower = lower;
  ai->local_upper = upper;
  ai->local_stride = 1;
  ai->alloc_size = size;

  ai->align_subscript = 0;

  ai->align_template_index = _XMP_N_NO_ALIGNED_TEMPLATE;
  ai->align_template_info = NULL;
  ai->align_template_chunk = NULL;
}

void _XMP_align_array_DUPLICATION(_XMP_array_t *array, int array_index, int template_index,
                                  long long align_subscript) {
  _XMP_template_t *template = array->align_template;
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
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
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;
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

    // FIXME array lower is always 0 in C
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    *temp0 = ai->par_lower;
    ai->temp0 = temp0;
  }

EXIT_CALC_PARALLEL_MEMBERS:

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;
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
    int mod = _XMP_modi_ll_i(chunk->par_lower - align_subscript, cycle);

    int dist = (ai->ser_upper - mod) / cycle;

    ai->par_lower = mod;
    ai->par_upper = mod + (dist * cycle);
    ai->par_stride = cycle;
    ai->par_size = dist + 1;

    // FIXME array lower is always 0 in C
    ai->local_lower = 0;
    ai->local_upper = ai->par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = ai->par_size;

    *temp0 = ai->par_stride;
    ai->temp0 = temp0;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_index = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;
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
  array_desc->addr = *array_addr;
  array_desc->total_elmts = total_elmts;
}

void _XMP_init_array_alloc_params(void **array_addr, _XMP_array_t *array_desc, ...) {
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
  array_desc->addr = *array_addr;
  array_desc->total_elmts = total_elmts;
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
  array_desc->addr = init_addr;
  array_desc->total_elmts = total_elmts;
}

void _XMP_init_array_comm(_XMP_array_t *array, ...) {
  _XMP_template_t *align_template = array->align_template;
  _XMP_ASSERT(align_template->is_distributed);

  _XMP_nodes_t *onto_nodes = align_template->onto_nodes;
  if (!onto_nodes->is_member) {
    return;
  }

  int color = 1;
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
    }
    else {
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

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)onto_nodes->comm), color, onto_nodes->comm_rank, comm);

  // set members
  array->is_align_comm_member = true;

  array->align_comm = comm;
  MPI_Comm_size(*comm, &(array->align_comm_size));
  MPI_Comm_rank(*comm, &(array->align_comm_rank));
}

unsigned long long _XMP_get_array_total_elmts(_XMP_array_t *array) {
  if (array->is_allocated) {
    return array->total_elmts;
  }
  else {
    return 0;
  }
}
