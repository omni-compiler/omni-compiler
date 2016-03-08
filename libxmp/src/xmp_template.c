#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

_XMP_template_t *_XMP_create_template_desc(int dim, _Bool is_fixed)
{
  // alloc descriptor
  _XMP_template_t *t = _XMP_alloc(sizeof(_XMP_template_t) +
                                  sizeof(_XMP_template_info_t) * (dim - 1));

  // calc members
  t->on_ref_id      = _XMP_get_on_ref_id();
  t->is_fixed       = is_fixed;
  t->is_distributed = false;
  t->is_owner       = false;
  t->dim            = dim;
  t->onto_nodes     = NULL;
  t->chunk          = NULL;

  return t;
}

void _XMP_calc_template_size(_XMP_template_t *t)
{
  int dim;
  if(t->is_fixed){
    dim = t->dim;
  }
  else{
    dim = t->dim-1;
  }

  for(int i=0;i<dim;i++){
    int ser_lower = t->info[i].ser_lower;
    int ser_upper = t->info[i].ser_upper;

    if(ser_lower > ser_upper)
      _XMP_fatal("the lower bound of template should be less than or equal to the upper bound");

    t->info[i].ser_size = _XMP_M_COUNTi(ser_lower, ser_upper);
  }
}

static void _XMP_validate_template_ref(int *lower, int *upper, int *stride, int lb, int ub)
{
  // setup temporary variables
  int l, u, s = *stride;
  if (s > 0) {
    l = *lower;
    u = *upper;
  }
  else if (s < 0) {
    l = *upper;
    u = *lower;
  }
  else {
    _XMP_fatal("the stride of <template-ref> is 0");
    l = 0; u = 0; // XXX dummy
  }

  // check boundary
  if (lb > l) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-lower> is less than the template lower bound");
  }

  if (l > u) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-upper> is less than <ref-lower>");
  }

  if (u > ub) {
    _XMP_fatal("<template-ref> is out of bounds, <ref-upper> is greater than the template upper bound");
  }

  // validate values
  if (s > 0) {
    u = u - ((u - l) % s);
    *upper = u;
  }
  else {
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }
}

static int _XMP_check_template_ref_inclusion_width_1(int ref_lower, int ref_upper, int ref_stride,
                                                     int template_lower, int template_upper, int template_stride) {
  if (ref_stride != 1) {
    _XMP_fatal("ref stride is not 1, -1: unsupported case");
  }

  int x_max = _XMP_floori((template_upper - template_lower), template_stride);

  // check ref_lower
  int x_st = _XMP_ceili((ref_lower - template_lower), template_stride);
  if (x_st > x_max) return _XMP_N_INT_FALSE;

  // check ref_upper
  int x_ed = _XMP_floori((ref_upper - template_lower), template_stride);
  if (x_ed < 0) return _XMP_N_INT_FALSE;

  if (x_st <= x_ed) {
    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

static int _XMP_check_template_ref_inclusion_width_N(int ref_lower, int ref_upper, int ref_stride,
                                                     _XMP_template_t *t, int index) {
  _XMP_template_info_t *info = &(t->info[index]);
  int template_ser_lower = info->ser_lower;

  _XMP_template_chunk_t *chunk = &(t->chunk[index]);
  int template_lower = chunk->par_lower;
  int template_upper = chunk->par_upper;
  int template_stride = chunk->par_stride;
  int width = chunk->par_width;

  int rl = ((ref_lower - template_ser_lower) / width) + template_ser_lower;
  int ru = ((ref_upper - template_ser_lower) / width) + template_ser_lower;
  int tl = ((template_lower - template_ser_lower) / width) + template_ser_lower;
  int tu = ((template_upper - template_ser_lower) / width) + template_ser_lower;
  int ts = template_stride / width; \

  // FIXME HOW IMPLEMENT???
  return _XMP_check_template_ref_inclusion_width_1(rl, ru, 1, tl, tu, ts);
}

static void _XMP_set_task_desc(_XMP_task_desc_t *desc, _XMP_nodes_t *n, int execute,
                               _XMP_template_t *ref_template, long long *ref_lower, long long *ref_upper, long long *ref_stride) {
  desc->nodes = n;
  desc->execute = execute;

  desc->on_ref_id = ref_template->on_ref_id;

  int dim = ref_template->dim;
  for (int i = 0; i < dim; i++) {
    desc->ref_lower[i] = ref_lower[i];
    desc->ref_upper[i] = ref_upper[i];
    desc->ref_stride[i] = ref_stride[i];
  }
}

static int _XMP_compare_task_exec_cond(_XMP_task_desc_t *task_desc,
                                       _XMP_template_t *ref_template,
                                       long long *ref_lower, long long *ref_upper, long long *ref_stride) {
  if (ref_template->on_ref_id != task_desc->on_ref_id) {
    return _XMP_N_INT_FALSE;
  }

  int dim = ref_template->dim;
  for (int i = 0; i < dim; i++) {
    if ((task_desc->ref_lower[i] != (int)ref_lower[i]) ||
        (task_desc->ref_upper[i] != (int)ref_upper[i]) ||
        (task_desc->ref_stride[i] != (int)ref_stride[i])) {
      return _XMP_N_INT_FALSE;
    }
  }

  return _XMP_N_INT_TRUE;
}

static void _XMP_dist_template_CYCLIC_WIDTH(_XMP_template_t *template, int template_index, int nodes_index,
                                            unsigned long long width) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_nodes_t *nodes = template->onto_nodes;

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  unsigned long long nodes_size = ni->size;
  unsigned long long template_size = _XMP_M_CEILi(ti->ser_size, width);

  // calc parallel members
  if (nodes->is_member) {
    unsigned long long nodes_rank = ni->rank;

    if (template_size < nodes_size) {
      if (nodes_rank < template_size) {
        long long par_index = ti->ser_lower + (nodes_rank * width);

        chunk->par_lower = par_index;
        chunk->par_upper = par_index;
      }
      else {
        template->is_owner = false;
      }
    }
    else {
      unsigned long long div = template_size / nodes_size;
      unsigned long long mod = template_size % nodes_size;
      unsigned long long par_size = 0;
      if(mod == 0) {
        par_size = div;
      }
      else {
        if(nodes_rank >= mod) {
          par_size = div;
        }
        else {
          par_size = div + 1;
        }
      }

      chunk->par_lower = ti->ser_lower + (nodes_rank * width);
      chunk->par_upper = chunk->par_lower + (nodes_size * (par_size - 1) * width);
    }
  }

  chunk->par_width = width;

  chunk->par_stride = nodes_size * width;
  chunk->par_chunk_width = _XMP_M_CEILi(template_size, nodes_size) * width;
  if (width == 1) {
    chunk->dist_manner = _XMP_N_DIST_CYCLIC;
  } else {
    chunk->dist_manner = _XMP_N_DIST_BLOCK_CYCLIC;
  }
  // FIXME consider block width
  if ((template_size % nodes_size) == 0) {
    chunk->is_regular_chunk = true;
  } else {
    chunk->is_regular_chunk = false;
  }

  chunk->onto_nodes_index = nodes_index;
  chunk->onto_nodes_info = ni;
}

int _XMP_check_template_ref_inclusion(int ref_lower, int ref_upper, int ref_stride,
                                      _XMP_template_t *t, int index) {
  _XMP_template_info_t *info = &(t->info[index]);
  _XMP_template_chunk_t *chunk = &(t->chunk[index]);

  _XMP_validate_template_ref(&ref_lower, &ref_upper, &ref_stride, info->ser_lower, info->ser_upper);

  switch (chunk->dist_manner) {
    case _XMP_N_DIST_DUPLICATION:
      return _XMP_N_INT_TRUE;
    case _XMP_N_DIST_BLOCK:
    case _XMP_N_DIST_CYCLIC:
      return _XMP_check_template_ref_inclusion_width_1(ref_lower, ref_upper, ref_stride,
                                                       chunk->par_lower, chunk->par_upper, chunk->par_stride);
    case _XMP_N_DIST_BLOCK_CYCLIC:
      return _XMP_check_template_ref_inclusion_width_N(ref_lower, ref_upper, ref_stride, t, index);
    default:
      _XMP_fatal("unknown distribution manner");
      return _XMP_N_INT_FALSE; // XXX dummy
  }
}

void _XMP_init_template_FIXED(_XMP_template_t **template, int dim, ...)
{
  // alloc descriptor
  _XMP_template_t *t = _XMP_create_template_desc(dim, true);

  // calc info
  va_list args;
  va_start(args, dim);
  for(int i=0;i<dim;i++){
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  _XMP_calc_template_size(t);

  *template = t;
}

void _XMP_init_template_UNFIXED(_XMP_template_t **template, int dim)
{
  *template = _XMP_create_template_desc(dim, false);
}

void _XMP_set_template_size(_XMP_template_t *t, int dim, ...) {

  // calc info
  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    t->info[i].ser_lower = va_arg(args, long long);
    t->info[i].ser_upper = va_arg(args, long long);
  }
  va_end(args);

  t->is_fixed = true;
  _XMP_calc_template_size(t);

}

void _XMP_init_template_chunk(_XMP_template_t *template, _XMP_nodes_t *nodes)
{
  template->is_distributed = true;
  template->is_owner       = nodes->is_member;
  template->onto_nodes     = nodes;
  template->chunk          = _XMP_alloc(sizeof(_XMP_template_chunk_t) * (template->dim));
}

void _XMP_finalize_template(_XMP_template_t *template) {
  if (template->is_distributed) {
    for (int i = 0; i < template->dim; i++){
      _XMP_template_chunk_t *chunk = &(template->chunk[i]);
      if (chunk->dist_manner == _XMP_N_DIST_GBLOCK){
	_XMP_free(chunk->mapping_array);
      }
    }
    _XMP_free(template->chunk);
  }

  _XMP_free(template);
}

void _XMP_dist_template_DUPLICATION(_XMP_template_t *template, int template_index) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);

  chunk->par_lower = ti->ser_lower;
  chunk->par_upper = ti->ser_upper;
  chunk->par_width = 1;

  chunk->par_stride = 1;
  chunk->par_chunk_width = ti->ser_size;
  chunk->dist_manner = _XMP_N_DIST_DUPLICATION;
  chunk->is_regular_chunk = true;

  chunk->onto_nodes_index = _XMP_N_NO_ONTO_NODES;
  chunk->onto_nodes_info = NULL;
}

void _XMP_dist_template_BLOCK(_XMP_template_t *template, int template_index, int nodes_index)
{
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_nodes_t *nodes          = template->onto_nodes;
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti     = &(template->info[template_index]);
  _XMP_nodes_info_t *ni        = &(nodes->info[nodes_index]);

  unsigned long long nodes_size = ni->size;

  // calc parallel members
  unsigned long long chunk_width = _XMP_M_CEILi(ti->ser_size, nodes_size);

  if(nodes->is_member){
    unsigned long long nodes_rank = ni->rank;
    unsigned long long owner_nodes_size = _XMP_M_CEILi(ti->ser_size, chunk_width);

    chunk->par_lower = nodes_rank * chunk_width + ti->ser_lower;
    if(nodes_rank == (owner_nodes_size - 1)){
      chunk->par_upper = ti->ser_upper;
    }
    else if (nodes_rank >= owner_nodes_size){
      template->is_owner = false;
    }
    else{
      chunk->par_upper = chunk->par_lower + chunk_width - 1;
    }
  }

  chunk->par_width       = 1;
  chunk->par_stride      = 1;
  chunk->par_chunk_width = chunk_width;
  chunk->dist_manner     = _XMP_N_DIST_BLOCK;
  
  if((ti->ser_size % nodes_size) == 0){
    chunk->is_regular_chunk = true;
  }
  else{
    chunk->is_regular_chunk = false;
  }

  chunk->onto_nodes_index = nodes_index;
  chunk->onto_nodes_info  = ni;
}

void _XMP_dist_template_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index)
{
  _XMP_dist_template_CYCLIC_WIDTH(template, template_index, nodes_index, 1);
}

void _XMP_dist_template_BLOCK_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index, unsigned long long width)
{
  _XMP_dist_template_CYCLIC_WIDTH(template, template_index, nodes_index, width);
}

#define min(x,y) ((x) < (y) ? (x) : (y))

void _XMP_dist_template_GBLOCK(_XMP_template_t *template, int template_index, int nodes_index,
			       int *mapping_array, int *temp0) {

  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_nodes_t *nodes = template->onto_nodes;

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_template_info_t *ti = &(template->info[template_index]);
  _XMP_nodes_info_t *ni = &(nodes->info[nodes_index]);

  if (mapping_array){

    long long *rsum_array = _XMP_alloc(sizeof(long long) * (ni->size + 1));
    rsum_array[0] = ti->ser_lower;
    for (int i = 1; i <= ni->size; i++){
      rsum_array[i] = rsum_array[i-1] + (long long)mapping_array[i-1];
    }
    chunk->mapping_array = rsum_array;

    if (ti->ser_upper >= rsum_array[ni->size])
      _XMP_fatal("The size of the template exceeds the sum of the mapping array.");

    if (ti->ser_upper + 1 < rsum_array[ni->size]) 
      _XMP_warning("The sum of the mapping array exceeds the size of the template.");

    template->is_owner = false;
    if (nodes->is_member && rsum_array[ni->rank + 1] != rsum_array[ni->rank]) {
      chunk->par_lower = rsum_array[ni->rank];
      if (chunk->par_lower <= ti->ser_upper){
	chunk->par_upper = min(ti->ser_upper, rsum_array[ni->rank + 1] - 1);
	template->is_owner = true;
      }
    }

    chunk->par_width = 1;
    chunk->par_stride = 1;
    chunk->par_chunk_width = chunk->par_upper - chunk->par_lower + 1;
  }
  else {
    template->is_fixed = false;
  }

  chunk->dist_manner = _XMP_N_DIST_GBLOCK;
  chunk->is_regular_chunk = false;

  chunk->onto_nodes_index = nodes_index;
  chunk->onto_nodes_info = ni;

  *temp0 = chunk->par_lower;
}

_XMP_nodes_t *_XMP_create_nodes_by_template_ref(_XMP_template_t *ref_template, int *shrink,
                                                long long *ref_lower, long long *ref_upper, long long *ref_stride)
{
  _XMP_ASSERT(ref_template->is_fixed);
  _XMP_ASSERT(ref_template->is_distributed);

  _XMP_nodes_t *onto_nodes = ref_template->onto_nodes;
  int onto_nodes_dim = onto_nodes->dim;
  int onto_nodes_shrink[onto_nodes_dim];
  int onto_nodes_ref_lower[onto_nodes_dim];
  int onto_nodes_ref_upper[onto_nodes_dim];
  int onto_nodes_ref_stride[onto_nodes_dim];
  
  for(int i=0;i<onto_nodes_dim;i++)
    onto_nodes_shrink[i] = 1;

  int new_nodes_dim = 0;
  int new_nodes_dim_size[_XMP_N_MAX_DIM];
  int acc_dim_size = 1;
  int ref_template_dim = ref_template->dim;
  
  for(int i=0;i<ref_template_dim;i++){
    if(shrink[i]){
      continue;
    }

    _XMP_template_chunk_t *chunk = &(ref_template->chunk[i]);
    int onto_nodes_index = chunk->onto_nodes_index;
    if (onto_nodes_index != _XMP_N_NO_ONTO_NODES) {
      onto_nodes_shrink[onto_nodes_index] = 0;

      int size = (chunk->onto_nodes_info)->size;
      // FIXME calc onto_nodes_ref_lower, onto_nodes_ref_upper, onto_nodes_ref_stride
      if (_XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]) == 1) {
        int j = _XMP_calc_template_owner_SCALAR(ref_template, i, ref_lower[i]) + 1;
        onto_nodes_ref_lower[onto_nodes_index] = j;
        onto_nodes_ref_upper[onto_nodes_index] = j;
        onto_nodes_ref_stride[onto_nodes_index] = 1;

        new_nodes_dim_size[new_nodes_dim] = 1;
      } else {
        onto_nodes_ref_lower[onto_nodes_index] = 1;
        onto_nodes_ref_upper[onto_nodes_index] = size;
        onto_nodes_ref_stride[onto_nodes_index] = 1;
        acc_dim_size *= _XMP_M_COUNT_TRIPLETi(1, size, 1);

        new_nodes_dim_size[new_nodes_dim] = size;
      }

      new_nodes_dim++;
    }
  }

  return _XMP_init_nodes_struct_NODES_NAMED(new_nodes_dim, onto_nodes, onto_nodes_shrink,
                                            onto_nodes_ref_lower, onto_nodes_ref_upper,
					    onto_nodes_ref_stride, new_nodes_dim_size, _XMP_N_INT_TRUE);
}

int _XMP_exec_task_TEMPLATE_PART(_XMP_task_desc_t **task_desc, _XMP_template_t *ref_template, ...) {
  int ref_dim = ref_template->dim;
  int shrink[ref_dim];
  long long lower[ref_dim], upper[ref_dim], stride[ref_dim];

  va_list args;
  va_start(args, ref_template);
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      lower[i] = va_arg(args, long long);
      upper[i] = va_arg(args, long long);
      stride[i] = va_arg(args, long long);
    }
  }
  va_end(args);

  _XMP_task_desc_t *desc = NULL;
  if (*task_desc == NULL) {
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
  } else {
    desc = *task_desc;
    if (_XMP_compare_task_exec_cond(desc, ref_template, lower, upper, stride)) {
      if (desc->execute) {
        _XMP_push_nodes(desc->nodes);
        return _XMP_N_INT_TRUE;
      } else {
        return _XMP_N_INT_FALSE;
      }
    } else {
      if (desc->nodes != NULL) {
        _XMP_finalize_nodes(desc->nodes);
      }
    }
  }

  _XMP_nodes_t *n = _XMP_create_nodes_by_template_ref(ref_template, shrink, lower, upper, stride);
  _XMP_set_task_desc(desc, n, n->is_member, ref_template, lower, upper, stride);
  if (n->is_member) {
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

_Bool union_triplet(int lb0, int ub0, int st0, int lb1, int ub1, int st1);

int _XMP_exec_task_TEMPLATE_PART_nocomm(_XMP_template_t *ref_template, ...) {

  if (!ref_template->is_owner) return _XMP_N_INT_FALSE;

  int ref_dim = ref_template->dim;

  int shrink[ref_dim];
  long long lower[ref_dim], upper[ref_dim], stride[ref_dim];

  va_list args;
  va_start(args, ref_template);

  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      lower[i] = va_arg(args, long long);
      upper[i] = va_arg(args, long long);
      stride[i] = va_arg(args, long long);
    }
  }

  va_end(args);

  for (int i = 0; i < ref_dim; i++){

    if (shrink[i]) continue;

    _XMP_template_chunk_t *chunk = &ref_template->chunk[i];
    long long plb = chunk->par_lower;
    long long pub = chunk->par_upper;
    int pst = chunk->par_stride;

    switch (chunk->dist_manner){

    case _XMP_N_DIST_DUPLICATION:
      break;

    case _XMP_N_DIST_BLOCK:
    case _XMP_N_DIST_GBLOCK:
      if (pub < lower[i] || upper[i] < plb) return _XMP_N_INT_FALSE;
      break;

    case _XMP_N_DIST_CYCLIC:
      if (union_triplet(lower[i], upper[i], stride[i], plb, pub, pst)) break;
      return _XMP_N_INT_FALSE;

    case _XMP_N_DIST_BLOCK_CYCLIC:
      for (int i = 0; i < chunk->par_width; i++){
	if (union_triplet(lower[i], upper[i], stride[i], plb+i, pub, pst)) goto next;
      }
      return _XMP_N_INT_FALSE;
    next:
      break;
    default:
      _XMP_fatal("_XMP_exec_task_TEMPLATE_PART_nocomm: unknown dist_manner");
    }

  }

  return _XMP_N_INT_TRUE;

}

// 0-origin
int _XMP_calc_template_owner_SCALAR(_XMP_template_t *template, int dim_index, long long ref_index) {
  _XMP_ASSERT(template->is_fixed);
  _XMP_ASSERT(template->is_distributed);

  _XMP_template_info_t *info = &(template->info[dim_index]);
  _XMP_template_chunk_t *chunk = &(template->chunk[dim_index]);
  _XMP_ASSERT(chunk->dist_manner != _XMP_N_DIST_DUPLICATION);

  switch (chunk->dist_manner) {
    case _XMP_N_DIST_BLOCK:
      return (ref_index - (info->ser_lower)) / (chunk->par_chunk_width);
    case _XMP_N_DIST_CYCLIC:
      return (ref_index - (info->ser_lower)) % (chunk->par_stride);
    case _XMP_N_DIST_BLOCK_CYCLIC:
      {
        int width = chunk->par_width;
        return ((ref_index - (info->ser_lower)) / width) % ((chunk->par_stride) / width);
      }
    case _XMP_N_DIST_GBLOCK:
      {
	long long *m = chunk->mapping_array;
	int np = chunk->onto_nodes_info->size;
	for (int i = 0; i < np; i++){
	  if (m[i] <= ref_index && ref_index < m[i+1]){
	    return i;
	  }
	}
      }
      return _XMP_N_INVALID_RANK;
    default:
      _XMP_fatal("unknown distribute manner");
      return _XMP_N_INVALID_RANK; // XXX dummy
  }
}

int _XMP_calc_template_par_triplet(_XMP_template_t *template, int template_index, int nodes_rank,
                                   int *template_lower, int *template_upper, int *template_stride) {
  _XMP_ASSERT(template->is_distributed);

  int par_lower = 0, par_upper = 0, par_stride = 0;

  _XMP_template_info_t *ti = &(template->info[template_index]);
  int ser_lower = ti->ser_lower;
  int ser_upper = ti->ser_upper;
  int ser_size = ti->ser_size;

  _XMP_template_chunk_t *tc = &(template->chunk[template_index]);
  int par_width = tc->par_width;
  int par_chunk_width = tc->par_chunk_width;

  switch (tc->dist_manner) {
    case _XMP_N_DIST_DUPLICATION:
      {
        par_lower = ser_lower;
        par_upper = ser_upper;

        par_stride = 1;
      } break;
    case _XMP_N_DIST_BLOCK:
      {
        par_lower = nodes_rank * par_chunk_width + ser_lower;
        int owner_nodes_size = _XMP_M_CEILi(ser_size, par_chunk_width);
        if (nodes_rank == (owner_nodes_size - 1)) {
          par_upper = ser_upper;
        } else if (nodes_rank >= owner_nodes_size) {
          return _XMP_N_INT_FALSE;
        } else {
          par_upper = par_lower + par_chunk_width - 1;
        }

        par_stride = 1;
      } break;
    case _XMP_N_DIST_CYCLIC:
    case _XMP_N_DIST_BLOCK_CYCLIC:
      {
        _XMP_ASSERT(tc->onto_nodes_index != _XMP_N_NO_ONTO_NODES);
        _XMP_nodes_info_t *ni = tc->onto_nodes_info;

        int nodes_size = ni->size;
        int template_size = _XMP_M_CEILi(ser_size, par_width);

        if (template_size < nodes_size) {
          if (nodes_rank < template_size) {
            int par_index = ser_lower + (nodes_rank * par_width);

            par_lower = par_index;
            par_upper = par_index;
          } else {
            return _XMP_N_INT_FALSE;
          }
        } else {
          int div = template_size / nodes_size;
          int mod = template_size % nodes_size;
          int par_size = 0;
          if(mod == 0) {
            par_size = div;
          } else {
            if(nodes_rank >= mod) {
              par_size = div;
            } else {
              par_size = div + 1;
            }
          }

          par_lower = ser_lower + (nodes_rank * par_width);
          par_upper = par_lower + (nodes_size * (par_size - 1) * par_width);
        }

        par_stride = nodes_size * par_width;
      } break;
    default:
      _XMP_fatal("unknown distribution manner");
  }

  // return triplet
  *template_lower = par_lower;
  *template_upper = par_upper;
  *template_stride = par_stride;

  return _XMP_N_INT_TRUE;
}


/* long long int _XMP_L2G_GBLOCK(int local_idx, _XMP_template_t *template, int template_index) */
/* { */
/*   _XMP_template_chunk_t *chunk = &(template->chunk[template_index]); */
/*   _XMP_ASSERT(chunk->dist_manner == _XMP_N_DIST_GBLOCK); */
/*   _XMP_nodes_info_t *n_info = chunk->onto_nodes_info; */
/*   long long int global_idx = local_idx + chunk->mapping_array[n_info->rank]; */
/*   return global_idx; */
/* } */
