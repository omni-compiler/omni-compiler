// FIXME delete stdio.h
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "xmp_constant.h"
#include "xmp_internal.h"

static _Bool _XCALABLEMP_check_gmove_inclusion_SCALAR(long long ref_index, _XCALABLEMP_template_chunk_t *chunk);
static int _XCALABLEMP_calc_gmove_owner_SCALAR(long long ref_index, _XCALABLEMP_template_t *template, int dim_index);
static int _XCALABLEMP_calc_gmove_nodes_rank(int *rank_array, _XCALABLEMP_nodes_t *nodes);

static _Bool _XCALABLEMP_check_gmove_inclusion_SCALAR(long long ref_index, _XCALABLEMP_template_chunk_t *chunk) {
  switch (chunk->dist_manner) {
    case _XCALABLEMP_N_DIST_DUPLICATION:
      return true;
    case _XCALABLEMP_N_DIST_BLOCK:
      {
        long long template_lower = chunk->par_lower;
        long long template_upper = chunk->par_upper;

        if (ref_index < template_lower) return false;
        if (template_upper < ref_index) return false;
        return true;
      }
    case _XCALABLEMP_N_DIST_CYCLIC:
      {
        long long par_stride = chunk->par_stride;
        if (((chunk->par_lower) % par_stride) == (ref_index % par_stride)) return true;
        else return false;
      }
    default:
      _XCALABLEMP_fatal("unknown distribute manner");
  }
}

static int _XCALABLEMP_calc_gmove_owner_SCALAR(long long ref_index, _XCALABLEMP_template_t *template, int dim_index) {
  _XCALABLEMP_template_info_t *info = &(template->info[dim_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[dim_index]);

  switch (chunk->dist_manner) {
    case _XCALABLEMP_N_DIST_DUPLICATION:
      return _XCALABLEMP_N_INVALID_RANK;
    case _XCALABLEMP_N_DIST_BLOCK:
      return (ref_index - (info->ser_lower)) / (chunk->par_chunk_width);
    case _XCALABLEMP_N_DIST_CYCLIC:
      return (ref_index - (info->ser_lower)) % (chunk->par_stride);
    default:
      _XCALABLEMP_fatal("unknown distribute manner");
  }
}

static int _XCALABLEMP_calc_gmove_nodes_rank(int *rank_array, _XCALABLEMP_nodes_t *nodes) {
  int nodes_dim = nodes->dim;

  int acc_rank = 0;
  int acc_nodes_size = 1;
  for (int i = 0; i < nodes_dim; i++) {
    int rank = rank_array[i];

    if (rank != _XCALABLEMP_N_INVALID_RANK) {
      acc_rank += rank * acc_nodes_size;
      acc_nodes_size *= nodes->info[i].size;
    }
  }

  return acc_rank;
}

void _XCALABLEMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, size_t type_size, _XCALABLEMP_array_t *array, ...) {
  // calc source rank
  if (array == NULL) return;

  _XCALABLEMP_template_t *template = array->align_template;
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  _XCALABLEMP_nodes_t *nodes = template->onto_nodes;
  if (nodes == NULL)
    _XCALABLEMP_fatal("null nodes descriptor detected");

  int array_dim = array->dim;
  int template_dim = template->dim;
  int nodes_dim = nodes->dim;

  int *src_rank_array = _XCALABLEMP_alloc(sizeof(int) * nodes_dim);
  for (int i = 0; i < nodes_dim; i++)
    src_rank_array[i] = _XCALABLEMP_N_INVALID_RANK;

  va_list args;
  va_start(args, array);
  for(int i = 0; i < array_dim; i++) {
    int ref_index = va_arg(args, int);

    int template_dim_index = array->info[i].align_template_dim;
    int owner = _XCALABLEMP_calc_gmove_owner_SCALAR(ref_index, template, template_dim_index);
    if (owner != _XCALABLEMP_N_INVALID_RANK)
      src_rank_array[template->chunk[template_dim_index].onto_nodes_dim] = owner;
  }
  va_end(args);

  int src_rank = _XCALABLEMP_calc_gmove_nodes_rank(src_rank_array, nodes);

  _XCALABLEMP_free(src_rank_array);

  // broadcast
  if (src_rank == array->comm_rank)
    memcpy(dst_addr, src_addr, type_size);

  MPI_Bcast(dst_addr, type_size, MPI_BYTE, src_rank, *(array->comm));
}

// FIXME change NULL check rule!!! (IMPORTANT, to all library functions)
_Bool _XCALABLEMP_gmove_exec_home_SCALAR(_XCALABLEMP_array_t *array, ...) {
  if (array == NULL) return false;

  _XCALABLEMP_template_t *ref_template = array->align_template;
  if (ref_template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  if (ref_template->chunk == NULL) return false;

  _Bool execHere = true;
  int ref_dim = array->dim;

  va_list args;
  va_start(args, array);
  for (int i = 0; i < ref_dim; i++) {
    int ref_index = va_arg(args, int);

    _XCALABLEMP_template_chunk_t *chunk = &(ref_template->chunk[i]);
    execHere = execHere && _XCALABLEMP_check_gmove_inclusion_SCALAR(ref_index + (array->info[i].align_subscript), chunk);
  }

  return execHere;
}

void _XCALABLEMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, size_t type_size,
                                       _XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array, ...) {
  // isend
  if (src_array == NULL) return;

  // recv
  if (dst_array == NULL) return;
}
