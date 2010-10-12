#include <stdarg.h>
#include <string.h>
#include "xmp_constant.h"
#include "xmp_internal.h"

#define _XCALABLEMP_M_GMOVE_BCAST_ARRAY(array, dst_addr, src_addr, type_size, src_rank) \
{ \
  int my_rank = array->comm_rank; \
  if (src_rank == my_rank) { \
    memcpy(dst_addr, src_addr, type_size); \
  } \
\
  MPI_Bcast(dst_addr, type_size, MPI_BYTE, src_rank, *(array->comm)); \
}

#define _XCALABLEMP_M_GMOVE_BCAST_EXEC(exec_nodes, array, dst_addr, src_addr, type_size, src_rank) \
{ \
  int my_rank = array->comm_rank; \
  if (src_rank == my_rank) { \
    memcpy(dst_addr, src_addr, type_size); \
\
    int exec_size = exec_nodes->comm_size; \
    int exec_rank = exec_nodes->comm_rank; \
    for (int i = 0; i < exec_size; i++) { \
      if (i != exec_rank) { \
        MPI_Send(dst_addr, type_size, MPI_BYTE, i, _XCALABLEMP_N_MPI_TAG_GMOVE, *(exec_nodes->comm)); \
      } \
    } \
  } \
  else { \
    MPI_Status stat; \
    MPI_Recv(dst_addr, type_size, MPI_BYTE, MPI_ANY_SOURCE, _XCALABLEMP_N_MPI_TAG_GMOVE, *(exec_nodes->comm), &stat); \
  } \
}

#define _XCALABLEMP_M_GMOVE_BCAST(array, dst_addr, src_addr, type_size, src_rank) \
{ \
  _XCALABLEMP_nodes_t *onto_nodes = (array->align_template)->onto_nodes; \
  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes(); \
\
  if ((exec_nodes == _XCALABLEMP_world_nodes) && (exec_nodes->comm_size == onto_nodes->comm_size)) { \
    _XCALABLEMP_M_GMOVE_BCAST_ARRAY(array, dst_addr, src_addr, type_size, src_rank); \
  } \
  else { \
    if (exec_nodes == onto_nodes) { \
      _XCALABLEMP_M_GMOVE_BCAST_ARRAY(array, dst_addr, src_addr, type_size, src_rank); \
    } \
    else { \
      _XCALABLEMP_M_GMOVE_BCAST_EXEC(exec_nodes, array, dst_addr, src_addr, type_size, src_rank); \
    } \
  } \
}

static _Bool _XCALABLEMP_check_gmove_inclusion_SCALAR(long long ref_index, _XCALABLEMP_template_chunk_t *chunk);
static int _XCALABLEMP_calc_gmove_owner_SCALAR(long long ref_index, _XCALABLEMP_template_t *template, int dim_index);
static int _XCALABLEMP_calc_gmove_nodes_rank(int *rank_array, _XCALABLEMP_nodes_t *nodes);
static int _XCALABLEMP_calc_gmove_target_nodes_size(_XCALABLEMP_nodes_t *nodes, int *rank_array);

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

  _Bool is_valid = false;
  int acc_rank = 0;
  int acc_nodes_size = 1;
  for (int i = 0; i < nodes_dim; i++) {
    int rank = rank_array[i];

    if (rank != _XCALABLEMP_N_INVALID_RANK) {
      is_valid = true;
      acc_rank += rank * acc_nodes_size;
      acc_nodes_size *= nodes->info[i].size;
    }
  }

  if (is_valid) return acc_rank;
  else          return _XCALABLEMP_N_INVALID_RANK;
}

static int _XCALABLEMP_calc_gmove_target_nodes_size(_XCALABLEMP_nodes_t *nodes, int *rank_array) {
  int nodes_dim = nodes->dim;

  int acc = 1;
  for (int i = 0; i < nodes_dim; i++) {
    int rank = rank_array[i];

    if (rank == _XCALABLEMP_N_INVALID_RANK) {
      acc *= nodes->info[i].size;
    }
  }

  return acc;
}

void _XCALABLEMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, size_t type_size, _XCALABLEMP_array_t *array, ...) {
  _XCALABLEMP_template_t *template = array->align_template;
  _XCALABLEMP_nodes_t *nodes = template->onto_nodes;

  int array_dim = array->dim;
  int nodes_dim = nodes->dim;

  int *src_rank_array = _XCALABLEMP_alloc(sizeof(int) * nodes_dim);
  for (int i = 0; i < nodes_dim; i++) {
    src_rank_array[i] = _XCALABLEMP_N_INVALID_RANK;
  }

  va_list args;
  va_start(args, array);
  for (int i = 0; i < array_dim; i++) {
    int ref_index = va_arg(args, int);

    int template_index = array->info[i].align_template_index;
    if (template_index != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
      int nodes_index = template->chunk[template_index].onto_nodes_index;
      if (nodes_index != _XCALABLEMP_N_NO_ONTO_NODES) {
        int owner = _XCALABLEMP_calc_gmove_owner_SCALAR(ref_index, template, template_index);
        if (owner != _XCALABLEMP_N_INVALID_RANK) {
          src_rank_array[nodes_index] = owner;
        }
      }
    }
  }
  va_end(args);

  int src_rank = _XCALABLEMP_calc_gmove_nodes_rank(src_rank_array, nodes);
  if (src_rank == _XCALABLEMP_N_INVALID_RANK) {
    // local copy
    memcpy(dst_addr, src_addr, type_size);
  }
  else {
    // broadcast
    _XCALABLEMP_M_GMOVE_BCAST(array, dst_addr, src_addr, type_size, src_rank);
  }

  // clean up
  _XCALABLEMP_free(src_rank_array);
}

// FIXME change NULL check rule!!! (IMPORTANT, to all library functions)
_Bool _XCALABLEMP_gmove_exec_home_SCALAR(_XCALABLEMP_array_t *array, ...) {
  if (!(array->is_allocated)) return false;

  _XCALABLEMP_template_t *ref_template = array->align_template;

  // FIXME how implement???
  if (ref_template->chunk == NULL) return false;

  _Bool execHere = true;
  int ref_dim = array->dim;

  va_list args;
  va_start(args, array);
  for (int i = 0; i < ref_dim; i++) {
    int ref_index = va_arg(args, int);

    int template_index = array->info[i].align_template_index;
    if (template_index != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
      execHere = execHere && _XCALABLEMP_check_gmove_inclusion_SCALAR(ref_index + (array->info[i].align_subscript),
                                                                      &(ref_template->chunk[template_index]));
    }
  }
  va_end(args);

  return execHere;
}

void _XCALABLEMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, size_t type_size,
                                       _XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array, ...) {
  va_list args;
  va_start(args, src_array);

  // calc destination rank
  if (dst_array == NULL) return;

  _XCALABLEMP_template_t *dst_template = dst_array->align_template;
  if (dst_template == NULL) {
    _XCALABLEMP_fatal("null template descriptor detected");
  }

  _XCALABLEMP_nodes_t *dst_nodes = dst_template->onto_nodes;
  if (dst_nodes == NULL) {
    _XCALABLEMP_fatal("null nodes descriptor detected");
  }

  int dst_array_dim = dst_array->dim;
  int dst_nodes_dim = dst_nodes->dim;

  int *dst_rank_array = _XCALABLEMP_alloc(sizeof(int) * dst_nodes_dim);
  for (int i = 0; i < dst_nodes_dim; i++) {
    dst_rank_array[i] = _XCALABLEMP_N_INVALID_RANK;
  }

  for (int i = 0; i < dst_array_dim; i++) {
    int dst_ref_index = va_arg(args, int);

    int dst_template_index = dst_array->info[i].align_template_index;
    if (dst_template_index != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
      int dst_nodes_index = dst_template->chunk[dst_template_index].onto_nodes_index;
      if (dst_nodes_index != _XCALABLEMP_N_NO_ONTO_NODES) {
        int dst_owner = _XCALABLEMP_calc_gmove_owner_SCALAR(dst_ref_index, dst_template, dst_template_index);
        if (dst_owner != _XCALABLEMP_N_INVALID_RANK) {
          dst_rank_array[dst_nodes_index] = dst_owner;
        }
      }
    }
  }

  int dst_rank = _XCALABLEMP_calc_gmove_nodes_rank(dst_rank_array, dst_nodes);

  // calc source rank
  if (src_array == NULL) return;

  _XCALABLEMP_template_t *src_template = src_array->align_template;
  if (src_template == NULL) {
    _XCALABLEMP_fatal("null template descriptor detected");
  }

  _XCALABLEMP_nodes_t *src_nodes = src_template->onto_nodes;
  if (src_nodes == NULL) {
    _XCALABLEMP_fatal("null nodes descriptor detected");
  }

  int src_array_dim = src_array->dim;
  int src_nodes_dim = src_nodes->dim;

  int *src_rank_array = _XCALABLEMP_alloc(sizeof(int) * src_nodes_dim);
  for (int i = 0; i < src_nodes_dim; i++) {
    src_rank_array[i] = _XCALABLEMP_N_INVALID_RANK;
  }

  for (int i = 0; i < src_array_dim; i++) {
    int src_ref_index = va_arg(args, int);

    int src_template_index = src_array->info[i].align_template_index;
    if (src_template_index != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
      int src_nodes_index = src_template->chunk[src_template_index].onto_nodes_index;
      if (src_nodes_index != _XCALABLEMP_N_NO_ONTO_NODES) {
        int src_owner = _XCALABLEMP_calc_gmove_owner_SCALAR(src_ref_index, src_template, src_template_index);
        if (src_owner != _XCALABLEMP_N_INVALID_RANK) {
          src_rank_array[src_nodes_index] = src_owner;
        }
      }
    }
  }

  int src_rank = _XCALABLEMP_calc_gmove_nodes_rank(src_rank_array, src_nodes);

  va_end(args);

  if ((dst_rank == _XCALABLEMP_N_INVALID_RANK) && (src_rank == _XCALABLEMP_N_INVALID_RANK)) {
    // local copy
    memcpy(dst_addr, src_addr, type_size);
  }
  else if ((dst_rank != _XCALABLEMP_N_INVALID_RANK) && (src_rank == _XCALABLEMP_N_INVALID_RANK)) {
    // local copy on dst_rank
    if (dst_rank == dst_array->comm_rank) {
      memcpy(dst_addr, src_addr, type_size);
    }
  }
  else if ((dst_rank == _XCALABLEMP_N_INVALID_RANK) && (src_rank != _XCALABLEMP_N_INVALID_RANK)) {
    // broadcast
    _XCALABLEMP_M_GMOVE_BCAST(src_array, dst_addr, src_addr, type_size, src_rank);
  }
  else { //(dst_rank != _XCALABLEMP_N_INVALID_RANK) && (src_rank != _XCALABLEMP_N_INVALID_RANK)
    // send/recv FIXME limitation: arrays should be distributed by the same nodes
    if (dst_nodes != src_nodes) {
      _XCALABLEMP_fatal("arrays used in a gmove directive should be distributed by the same nodes set");
    }

    // FIXME use execution nodes set
    _XCALABLEMP_nodes_t *comm_nodes = dst_nodes;

    // irecv
    MPI_Request recv_req;
    if (dst_rank == dst_array->comm_rank) {
      MPI_Irecv(dst_addr, type_size, MPI_BYTE, MPI_ANY_SOURCE, _XCALABLEMP_N_MPI_TAG_GMOVE, *(comm_nodes->comm), &recv_req);
    }

    // send
    if (src_rank == src_array->comm_rank) {
      // FIXME master sends all
      if (src_rank == comm_nodes->comm_rank) {
        int num_targets = _XCALABLEMP_calc_gmove_target_nodes_size(dst_nodes, dst_rank_array);
        if (num_targets == 1) {
          MPI_Send(src_addr, type_size, MPI_BYTE, dst_rank, _XCALABLEMP_N_MPI_TAG_GMOVE, *(comm_nodes->comm));
        }
        else {
          // FIXME implement
          _XCALABLEMP_fatal("not supported yet");
        }
      }
    }

    // wait
    if (dst_rank == dst_array->comm_rank) {
      MPI_Status recv_stat;
      MPI_Wait(&recv_req, &recv_stat);
    }
  }

  // clean up
  _XCALABLEMP_free(dst_rank_array);
  _XCALABLEMP_free(src_rank_array);
}
