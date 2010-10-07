#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_macro.h"

static void _XCALABLEMP_calc_array_dim_elmts(_XCALABLEMP_array_t *array, int array_index) {
  int dim = array->dim;

  unsigned long long dim_elmts = 1;
  for (int i = 0; i < dim; i++) {
    if (i != array_index) {
      dim_elmts *= array->info[i].par_size;
    }
  }

  array->info[array_index].dim_elmts = dim_elmts;
}

void _XCALABLEMP_init_array_desc(_XCALABLEMP_array_t **array, _XCALABLEMP_template_t *template, int dim, ...) {
  _XCALABLEMP_array_t *a = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_array_t) + sizeof(_XCALABLEMP_array_info_t) * (dim - 1));

  a->is_allocated = true;
  a->dim = dim;

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int size = va_arg(args, int);
    int lower = 0;
    int upper = size - 1;

    _XCALABLEMP_array_info_t *ai = &(a->info[i]);

    ai->ser_lower = lower;
    ai->ser_upper = upper;
    ai->ser_size = size;

    ai->par_lower = lower;
    ai->par_upper = upper;
    ai->par_stride = 1;
    ai->par_size = size;

    ai->local_lower = lower;
    ai->local_upper = upper;
    ai->local_stride = 1;
    ai->alloc_size = size;

 // ai->dim_acc is calculated in _XCALABLEMP_alloc_array, _XCALABLEMP_init_array_addr
 // ai->dim_elmts is calculated in _XCALABLEMP_alloc_array, _XCALABLEMP_init_array_addr
    
    ai->align_subscript = 0;

    ai->shadow_type = _XCALABLEMP_N_SHADOW_NONE;
    ai->shadow_size_lo  = 0;
    ai->shadow_size_hi  = 0;

    ai->align_template_dim = _XCALABLEMP_N_NO_ALIGNED_TEMPLATE;
    ai->align_template_info = NULL;
    ai->align_template_chunk = NULL;
  }
  va_end(args);

  a->align_template = template;

  *array = a;
}

void _XCALABLEMP_finalize_array_desc(_XCALABLEMP_array_t *array) {
  if (array != NULL) {
    _XCALABLEMP_free(array->comm);
    _XCALABLEMP_free(array);
  }
}

void _XCALABLEMP_align_array_DUPLICATION(_XCALABLEMP_array_t *array, int array_index, int template_index,
                                         long long align_subscript) {
  _XCALABLEMP_template_t *template = array->align_template;
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_array_info_t *ai = &(array->info[array_index]);

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XCALABLEMP_fatal("aligned array is out of template bound");
  }

  // set members
  ai->align_subscript = align_subscript;

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = &(template->chunk[template_index]);
}

void _XCALABLEMP_align_array_BLOCK(_XCALABLEMP_array_t *array, int array_index, int template_index,
                                   long long align_subscript, int *temp0) {
  _XCALABLEMP_template_t *template = array->align_template;
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XCALABLEMP_array_info_t *ai = &(array->info[array_index]);

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;
  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XCALABLEMP_fatal("aligned array is out of template bound");
  }

  int par_lower, par_upper, par_size;
  if (template->is_owner) {
    long long template_lower = chunk->par_lower;
    long long template_upper = chunk->par_upper;

    // set par_lower
    if (align_lower < template_lower) {
      par_lower = template_lower - align_subscript;
    }
    else if (template_upper < align_lower) {
      array->is_allocated = false;
      return;
    }
    else {
      par_lower = ai->ser_lower;
    }

    // set par_upper
    if (align_upper < template_lower) {
      array->is_allocated = false;
      return;
    }
    else if (template_upper < align_upper) {
      par_upper = template_upper - align_subscript;
    }
    else {
      par_upper = ai->ser_upper;
    }

    // set par_size
    par_size = _XCALABLEMP_M_COUNT_TRIPLETi(par_lower, par_upper, 1);
  }

  // set members
  if (array->is_allocated) {
    ai->par_lower = par_lower;
    ai->par_upper = par_upper;
    ai->par_stride = 1;
    ai->par_size = par_size;

    ai->local_lower = 0;
    ai->local_upper = par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = par_size;

    *temp0 = ai->par_lower;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;
}

void _XCALABLEMP_align_array_CYCLIC(_XCALABLEMP_array_t *array, int array_index, int template_index,
                                    long long align_subscript, int *temp0) {
  _XCALABLEMP_template_t *template = array->align_template;
  _XCALABLEMP_template_info_t *ti = &(template->info[template_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XCALABLEMP_array_info_t *ai = &(array->info[array_index]);

  long long align_lower = ai->ser_lower + align_subscript;
  long long align_upper = ai->ser_upper + align_subscript;

  if (((align_lower < ti->ser_lower) || (align_upper > ti->ser_upper))) {
    _XCALABLEMP_fatal("aligned array is out of template bound");
  }

  int par_lower, par_upper, par_stride, par_size;
  if (template->is_owner) {
    int cycle = chunk->par_stride;
    int mod = (chunk->par_lower - align_subscript) % cycle;
    if (mod < 0) mod += cycle;

    int dist = (ai->ser_upper - mod) / cycle;

    par_lower = mod;
    par_upper = mod + (dist * cycle);
    par_stride = cycle;
    par_size = dist + 1;
  }

  // set members
  if (array->is_allocated) {
    ai->par_lower = par_lower;
    ai->par_upper = par_upper;
    ai->par_stride = par_stride;
    ai->par_size = par_size;

    ai->local_lower = 0;
    ai->local_upper = par_size - 1;
    ai->local_stride = 1;
    ai->alloc_size = par_size;

    *temp0 = par_stride;
  }

  ai->align_subscript = align_subscript;

  ai->align_template_dim = template_index;
  ai->align_template_info = ti;
  ai->align_template_chunk = chunk;
}

void _XCALABLEMP_alloc_array(void **array_addr, _XCALABLEMP_array_t *array_desc, int datatype_size, ...) {
  if (!(array_desc->is_allocated)) {
    *array_addr = NULL;
    return;
  }

  unsigned long long total_elmts = 1;
  int dim = array_desc->dim;
  va_list args;
  va_start(args, datatype_size);
  for (int i = dim - 1; i >= 0; i--) {
    unsigned long long *acc = va_arg(args, unsigned long long *);
    *acc = total_elmts;

    array_desc->info[i].dim_acc = total_elmts;

    total_elmts *= array_desc->info[i].alloc_size;
  }
  va_end(args);

  for (int i = 0; i < dim; i++) {
    _XCALABLEMP_calc_array_dim_elmts(array_desc, i);
  }

  *array_addr = _XCALABLEMP_alloc(total_elmts * datatype_size);
}

void _XCALABLEMP_init_array_addr(void **array_addr, void *param_addr,
                                 _XCALABLEMP_array_t *array_desc, ...) {
  if (!(array_desc->is_allocated)) {
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
    _XCALABLEMP_calc_array_dim_elmts(array_desc, i);
  }

  *array_addr = param_addr;
}

void _XCALABLEMP_init_array_comm(_XCALABLEMP_array_t *array, ...) {
  _XCALABLEMP_template_t *align_template = array->align_template;
  _XCALABLEMP_nodes_t *onto_nodes = align_template->onto_nodes;
  if (!(onto_nodes->is_member)) {
    return;
  }

  int color = 1;
  int acc_nodes_size = 1;
  int template_dim = align_template->dim;

  va_list args;
  va_start(args, array);
  for (int i = 0; i < template_dim; i++) {
    _XCALABLEMP_template_chunk_t *chunk = &(align_template->chunk[i]);

    int size, rank;
    _XCALABLEMP_nodes_info_t *onto_nodes_info = chunk->onto_nodes_info;
    if (chunk->dist_manner == _XCALABLEMP_N_DIST_DUPLICATION) {
      size = 1;
      rank = 0;
    }
    else {
      size = onto_nodes_info->size;
      rank = onto_nodes_info->rank;
    }

    if (va_arg(args, int) == 1) {
      color += (acc_nodes_size * rank);
    }

    acc_nodes_size *= size;
  }
  va_end(args);

  MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*(onto_nodes->comm), color, onto_nodes->comm_rank, comm);

  // set members
  array->comm = comm;
  MPI_Comm_size(*comm, &(array->comm_size));
  MPI_Comm_rank(*comm, &(array->comm_rank));
}
