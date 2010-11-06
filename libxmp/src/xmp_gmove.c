#include <stdarg.h>
#include <string.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

typedef struct _XCALABLEMP_bcast_array_section_info_type {
  int lower;
  int upper;
  int stride;
} _XCALABLEMP_bcast_array_section_info_t;

// ----- gmove scalar to scalar ----------------------------------------------------------

#define _XCALABLEMP_M_GMOVE_BCAST_ARRAY(array, dst_addr, src_addr, type_size, src_rank) \
{ \
  int my_rank = array->align_comm_rank; \
  if (src_rank == my_rank) { \
    memcpy(dst_addr, src_addr, type_size); \
  } \
\
  MPI_Bcast(dst_addr, type_size, MPI_BYTE, src_rank, *(array->align_comm)); \
}

#define _XCALABLEMP_M_GMOVE_BCAST_EXEC(exec_nodes, array, dst_addr, src_addr, type_size, src_rank) \
{ \
  int my_rank = array->align_comm_rank; \
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
static _Bool _XCALABLEMP_calc_local_copy_template_BLOCK(_XCALABLEMP_template_chunk_t *chunk,
                                                        long long *lower, long long *upper, int s);
static _Bool _XCALABLEMP_calc_local_copy_template_CYCLIC1(_XCALABLEMP_template_chunk_t *chunk,
                                                          long long *lower, long long u, int *stride);
static _Bool _XCALABLEMP_calc_local_copy_home_ref(_XCALABLEMP_array_t *dst_array, int dst_dim_index,
                                                  int *dst_l, int *dst_u, int *dst_s,
                                                  int *src_l, int *src_u, int *src_s);
static void _XCALABLEMP_calc_array_local_index_triplet(_XCALABLEMP_array_t *array,
                                                       int dim_index, int *lower, int *upper, int *stride);
static _Bool _XCALABLEMP_gmove_check_array_ref_inclusion(_XCALABLEMP_array_info_t *array_info, int ref_index);

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
        int par_stride = chunk->par_stride;
        if (_XCALABLEMP_modi_ll_i(chunk->par_lower, par_stride) == _XCALABLEMP_modi_ll_i(ref_index, par_stride)) return true;
        else return false;
      }
    default:
      _XCALABLEMP_fatal("unknown distribute manner");
      return false; // XXX dummy
  }
}

static int _XCALABLEMP_calc_gmove_owner_SCALAR(long long ref_index, _XCALABLEMP_template_t *template, int dim_index) {
  assert(template != NULL);

  _XCALABLEMP_template_info_t *info = &(template->info[dim_index]);
  _XCALABLEMP_template_chunk_t *chunk = &(template->chunk[dim_index]);

  switch (chunk->dist_manner) {
    case _XCALABLEMP_N_DIST_DUPLICATION:
      return _XCALABLEMP_N_INVALID_RANK;
    case _XCALABLEMP_N_DIST_BLOCK:
      return (ref_index - (info->ser_lower)) / (chunk->par_chunk_width);
    case _XCALABLEMP_N_DIST_CYCLIC:
      return _XCALABLEMP_modi_ll_i(ref_index - (info->ser_lower), chunk->par_stride);
    default:
      _XCALABLEMP_fatal("unknown distribute manner");
      return 0; // XXX dummy
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

static _Bool _XCALABLEMP_calc_local_copy_template_BLOCK(_XCALABLEMP_template_chunk_t *chunk,
                                                        long long *lower, long long *upper, int s) {
  long long l = *lower;
  long long u = *upper;
  long long template_lower = chunk->par_lower;
  long long template_upper = chunk->par_upper;

  if (s != 1) {
    int dst_mod = _XCALABLEMP_modi_ll_i(l, s);
    // normalize template lower
    int lower_mod = _XCALABLEMP_modi_ll_i(template_lower, s);
    if (lower_mod != dst_mod) {
      if (lower_mod < dst_mod) {
        template_lower += (dst_mod - lower_mod);
      }
      else {
        template_lower += (s - lower_mod + dst_mod);
      }
    }

    if (template_lower > template_upper) return false;
  }

  // calc lower
  if (l < template_lower) {
    *lower = template_lower;
  }
  else if (template_upper < l) {
    return false;
  }
  else  {
    *lower = l;
  }

  // calc upper
  if (u < template_lower) {
    return false;
  }
  else if (template_upper < u) {
    *upper = template_upper;
  }
  else {
    *upper = u;
  }

  return true;
}

// XXX used when ref_stride is 1
static _Bool _XCALABLEMP_calc_local_copy_template_CYCLIC1(_XCALABLEMP_template_chunk_t *chunk,
                                                          long long *lower, long long u, int *stride) {
  long long l = *lower;
  long long template_lower = chunk->par_lower;
  int nodes_size = chunk->onto_nodes_info->size;

  // calc lower
  int dst_mod = _XCALABLEMP_modi_ll_i(template_lower, nodes_size);
  int lower_mod = _XCALABLEMP_modi_ll_i(l, nodes_size);
  if (lower_mod != dst_mod) {
    if (lower_mod < dst_mod) {
      l += (dst_mod - lower_mod);
    }
    else {
      l += (nodes_size - lower_mod + dst_mod);
    }
  }

  if (u < l) {
    return false;
  }
  else {
    *lower = l;
  }

  // calc stride;
  *stride = nodes_size;

  return true;
}

static _Bool _XCALABLEMP_calc_local_copy_home_ref(_XCALABLEMP_array_t *dst_array, int dst_dim_index,
                                                  int *dst_l, int *dst_u, int *dst_s,
                                                  int *src_l, int *src_u, int *src_s) {
  if (_XCALABLEMP_M_COUNT_TRIPLETi(*dst_l, *dst_u, *dst_s) != _XCALABLEMP_M_COUNT_TRIPLETi(*src_l, *src_u, *src_s)) {
    _XCALABLEMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  _XCALABLEMP_array_info_t *dst_array_info = &(dst_array->info[dst_dim_index]);
  if ((dst_array_info->align_template_index) == _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
    return true;
  }
  else {
    long long align_subscript = dst_array_info->align_subscript;
    long long l = *dst_l + align_subscript;
    long long u = *dst_u + align_subscript;
    int s = *dst_s;

    _XCALABLEMP_template_chunk_t *chunk = dst_array_info->align_template_chunk;
    switch (chunk->dist_manner) {
      case _XCALABLEMP_N_DIST_DUPLICATION:
        return true;
      case _XCALABLEMP_N_DIST_BLOCK:
        {
          _Bool res = _XCALABLEMP_calc_local_copy_template_BLOCK(chunk, &l, &u, s);
          if (res) {
            int new_dst_l = l - align_subscript;
            int new_dst_u = u - align_subscript;

            // update src ref
            *src_l += (((new_dst_l - (*dst_l)) / (*dst_s)) * (*src_s));
            *src_u = (*src_l) + ((_XCALABLEMP_M_COUNT_TRIPLETi(new_dst_l, new_dst_u, s) - 1) * (*src_s));

            // update dst ref
            *dst_l = new_dst_l;
            *dst_u = new_dst_u;
          }

          return res;
        }
      case _XCALABLEMP_N_DIST_CYCLIC:
        {
          if (s == 1) {
            _Bool res = _XCALABLEMP_calc_local_copy_template_CYCLIC1(chunk, &l, u, &s);
            if (res) {
              int new_dst_l = l - align_subscript;
              int new_dst_s = s;

              // update src ref
              *src_l += ((new_dst_l - (*dst_l)) * (*src_s));
              *src_s *= new_dst_s;

              // update dst ref
              *dst_l = new_dst_l;
              *dst_s = new_dst_s;
            }

            return res;
          }
          else {
            // FIXME
            _XCALABLEMP_fatal("not implemented yet");
            return false; // XXX dummy;
          }
        }
      default:
        _XCALABLEMP_fatal("unknown distribute manner");
        return false; // XXX dummy;
    }
  }
}

static void _XCALABLEMP_calc_array_local_index_triplet(_XCALABLEMP_array_t *array,
                                                       int dim_index, int *lower, int *upper, int *stride) {
  _XCALABLEMP_array_info_t *array_info = &(array->info[dim_index]);
  if ((array_info->align_template_index) != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
    int dist_manner = (array_info->align_template_chunk)->dist_manner;
    switch (array_info->shadow_type) {
      case _XCALABLEMP_N_SHADOW_NONE:
        {
          switch (dist_manner) {
            case _XCALABLEMP_N_DIST_BLOCK:
              {
                *lower -= (*(array_info->temp0));
                *upper -= (*(array_info->temp0));
                *stride = 1;
              } break;
            case _XCALABLEMP_N_DIST_CYCLIC:
              {
                *lower /= (*(array_info->temp0));
                *upper /= (*(array_info->temp0));
                *stride = 1;
              } break;
            default:
              _XCALABLEMP_fatal("wrong distribute manner for normal shadow");
          }
        } break;
      case _XCALABLEMP_N_SHADOW_NORMAL:
        {
          switch (dist_manner) {
            case _XCALABLEMP_N_DIST_BLOCK:
              {
                *lower -= (*(array_info->temp0));
                *upper -= (*(array_info->temp0));
                *stride = 1;
              } break;
            // FIXME normal shadow is not allowed in cyclic distribution
            default:
              _XCALABLEMP_fatal("wrong distribute manner for normal shadow");
          }
        } break;
      case _XCALABLEMP_N_SHADOW_FULL:
        return;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
}

static _Bool _XCALABLEMP_gmove_check_array_ref_inclusion(_XCALABLEMP_array_info_t *array_info, int ref_index) {
  int template_index = array_info->align_template_index;
  if (template_index == _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
    return true;
  }
  else {
    return _XCALABLEMP_check_gmove_inclusion_SCALAR(ref_index + array_info->align_subscript, array_info->align_template_chunk);
  }
}

void _XCALABLEMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, _XCALABLEMP_array_t *array, ...) {
  // FIXME type check here?
  size_t type_size = array->type_size;

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

  // FIXME delete after change manual bcast implementation
  _XCALABLEMP_barrier_EXEC();
}

// FIXME change NULL check rule!!! (IMPORTANT, to all library functions)
_Bool _XCALABLEMP_gmove_exec_home_SCALAR(_XCALABLEMP_array_t *array, ...) {
  if (!array->is_allocated) return false;

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

void _XCALABLEMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr,
                                       _XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array, ...) {
  // FIXME type check here?
  size_t type_size = dst_array->type_size;

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

  if (dst_rank == _XCALABLEMP_N_INVALID_RANK) {
    if (src_rank == _XCALABLEMP_N_INVALID_RANK) {
      // local copy
      memcpy(dst_addr, src_addr, type_size);
    }
    else {
      // broadcast
      _XCALABLEMP_M_GMOVE_BCAST(src_array, dst_addr, src_addr, type_size, src_rank);
    }
  }
  else {
    if (src_rank == _XCALABLEMP_N_INVALID_RANK) {
      // local copy on dst_rank
      if (dst_rank == dst_array->align_comm_rank) {
        memcpy(dst_addr, src_addr, type_size);
      }
    }
    else {
      // send/recv FIXME limitation: arrays should be distributed by the same nodes
      if (dst_nodes != src_nodes) {
        _XCALABLEMP_fatal("arrays used in a gmove directive should be distributed by the same nodes set");
      }

      // FIXME use execution nodes set
      _XCALABLEMP_nodes_t *comm_nodes = dst_nodes;

      // irecv
      MPI_Request recv_req;
      if (dst_rank == dst_array->align_comm_rank) {
        MPI_Irecv(dst_addr, type_size, MPI_BYTE, MPI_ANY_SOURCE, _XCALABLEMP_N_MPI_TAG_GMOVE, *(comm_nodes->comm), &recv_req);
      }

      // send
      if (src_rank == src_array->align_comm_rank) {
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
      if (dst_rank == dst_array->align_comm_rank) {
        MPI_Status recv_stat;
        MPI_Wait(&recv_req, &recv_stat);
      }
    }
  }

  // clean up
  _XCALABLEMP_free(dst_rank_array);
  _XCALABLEMP_free(src_rank_array);

  _XCALABLEMP_barrier_EXEC();
}

// ----- gmove vector to vector ----------------------------------------------------------

void _XCALABLEMP_gmove_local_copy(int type, size_t type_size, ...) {
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  unsigned long long dst_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);
  
  // alloc buffer
  if (dst_buffer_elmts != src_buffer_elmts) {
    _XCALABLEMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  void *buffer = _XCALABLEMP_alloc(dst_buffer_elmts * type_size);

  // pack/unpack
  if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
    _XCALABLEMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    _XCALABLEMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }
  else {
    _XCALABLEMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    _XCALABLEMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }

  // free buffer
  _XCALABLEMP_free(buffer);
}

void _XCALABLEMP_gmove_local_copy_home(_XCALABLEMP_array_t *dst_array, int type, size_t type_size, ...) {
  if (!dst_array->is_allocated) {
    return;
  }

  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  // calc index ref
  int src_dim_index = 0;
  unsigned long long dst_buffer_elmts = 1;
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    int dst_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_elmts == 1) {
      if(!_XCALABLEMP_gmove_check_array_ref_inclusion(&(dst_array->info[i]), dst_l[i])) {
        return;
      }
    }
    else {
      dst_buffer_elmts *= dst_elmts;

      int src_elmts;
      do {
        src_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(src_l[src_dim_index], src_u[src_dim_index], src_s[src_dim_index]);
        src_dim_index++;
      } while (src_elmts == 1);

      int j = src_dim_index - 1;
      if (_XCALABLEMP_calc_local_copy_home_ref(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]),
                                                             &(src_l[j]), &(src_u[j]), &(src_s[j]))) {
        src_buffer_elmts *= src_elmts;
      }
      else {
        return;
      }
    }

    _XCALABLEMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  for (int i = src_dim_index; i < src_dim; i++) {
    src_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  // alloc buffer
  if (dst_buffer_elmts != src_buffer_elmts) {
    _XCALABLEMP_fatal("wrong assign statement"); // FIXME fix error msg
  }

  void *buffer = _XCALABLEMP_alloc(dst_buffer_elmts * type_size);

  // pack/unpack
  if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
    _XCALABLEMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    _XCALABLEMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }
  else {
    _XCALABLEMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    _XCALABLEMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
  }

  // free buffer
  _XCALABLEMP_free(buffer);
}

void _XCALABLEMP_gmove_BCAST_ARRAY_SECTION(_XCALABLEMP_array_t *src_array, int type, size_t type_size, ...) {
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  // calc index ref
  int is_root = _XCALABLEMP_N_INT_TRUE;
  int dst_dim_index = 0;
  unsigned long long dst_buffer_elmts = 1;
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    int src_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    if (src_elmts == 1) {
      if(!_XCALABLEMP_gmove_check_array_ref_inclusion(&(src_array->info[i]), src_l[i])) {
        is_root = _XCALABLEMP_N_INT_FALSE;
        break;
      }
    }
    else {
      src_buffer_elmts *= src_elmts;

      int dst_elmts;
      do {
        dst_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[dst_dim_index], dst_u[dst_dim_index], dst_s[dst_dim_index]);
        dst_dim_index++;
      } while (dst_elmts == 1);

      int j = dst_dim_index - 1;
      if (_XCALABLEMP_calc_local_copy_home_ref(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]),
                                                             &(dst_l[j]), &(dst_u[j]), &(dst_s[j]))) {
        dst_buffer_elmts *= dst_elmts;
      }
      else {
        is_root = _XCALABLEMP_N_INT_FALSE;
        break;
      }
    }

    _XCALABLEMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  // bcast data
  void *pack_buffer = NULL;
  if (is_root) {
    for (int i = dst_dim_index; i < dst_dim; i++) {
      dst_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    }

    if (dst_buffer_elmts != src_buffer_elmts) {
      _XCALABLEMP_fatal("wrong assign statement"); // FIXME fix error msg
    }

    pack_buffer = _XCALABLEMP_alloc(src_buffer_elmts * type_size);
    if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
      _XCALABLEMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
    }
    else {
      _XCALABLEMP_pack_array_BASIC(pack_buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
    }
  }

  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes();
  MPI_Comm *exec_nodes_comm = exec_nodes->comm;
  int exec_nodes_size = exec_nodes->comm_size;
  int exec_nodes_rank = exec_nodes->comm_rank;

  int *root_nodes = _XCALABLEMP_alloc(exec_nodes_size * sizeof(int));
  MPI_Allgather(&is_root, 1, MPI_INT, root_nodes, 1, MPI_INT, *exec_nodes_comm);

  _XCALABLEMP_bcast_array_section_info_t bcast_info[dst_dim];

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  int bcast_l[dst_dim], bcast_u[dst_dim], bcast_s[dst_dim];
  unsigned long long bcast_elmts;
  for (int i = 0; i < exec_nodes_size; i++) {
    if (root_nodes[i]) {
      if (i == exec_nodes_rank) {
        for (int j = 0; j < dst_dim; j++) {
          bcast_info[j].lower = dst_l[j];
          bcast_info[j].upper = dst_u[j];
          bcast_info[j].stride = dst_s[j];
        }
      }

      MPI_Bcast(bcast_info, sizeof(_XCALABLEMP_bcast_array_section_info_t) * dst_dim, MPI_BYTE, i, *exec_nodes_comm);

      bcast_elmts = 1;
      for (int j = 0; j < dst_dim; j++) {
        bcast_l[j] = bcast_info[j].lower;
        bcast_u[j] = bcast_info[j].upper;
        bcast_s[j] = bcast_info[j].stride;
        bcast_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(bcast_l[j], bcast_u[j], bcast_s[j]);
      }

      void *bcast_buffer;
      if (i == exec_nodes_rank) {
        bcast_buffer = pack_buffer;
      }
      else {
        bcast_buffer = _XCALABLEMP_alloc(bcast_elmts * type_size);
      }
      MPI_Bcast(bcast_buffer, bcast_elmts, mpi_datatype, i, *exec_nodes_comm);

      if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
        _XCALABLEMP_unpack_array_GENERAL(dst_addr, bcast_buffer, type_size, dst_dim, bcast_l, bcast_u, bcast_s, dst_d);
      }
      else {
        _XCALABLEMP_unpack_array_BASIC(dst_addr, bcast_buffer, type, dst_dim, bcast_l, bcast_u, bcast_s, dst_d);
      }
      _XCALABLEMP_free(bcast_buffer);
    }
  }
}

// FIXME temporary implementation
static int _XCALABLEMP_calc_SENDRECV_owner(_XCALABLEMP_array_t *array, int *lower, int *upper, int *stride) {
  _XCALABLEMP_template_t *template = array->align_template;
  _XCALABLEMP_nodes_t *nodes = template->onto_nodes;

  int array_dim = array->dim;
  int nodes_dim = nodes->dim;

  int rank_array[nodes_dim];
  for (int i = 0; i < nodes_dim; i++) {
    rank_array[i] = _XCALABLEMP_N_INVALID_RANK;
  }

  for (int i = 0; i < array_dim; i++) {
    _XCALABLEMP_array_info_t *ai = &(array->info[i]);
    int template_index = ai->align_template_index;
    if (template_index != _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
      if (_XCALABLEMP_M_COUNT_TRIPLETi(lower[i], upper[i], stride[i]) == 1) {
        int nodes_index = (ai->align_template_chunk)->onto_nodes_index;
        if (nodes_index != _XCALABLEMP_N_NO_ONTO_NODES) {
          int owner = _XCALABLEMP_calc_gmove_owner_SCALAR(lower[i], template, template_index);
          if (owner != _XCALABLEMP_N_INVALID_RANK) {
            rank_array[nodes_index] = owner;
          }
        }
      }
      else {
        if (((ai->align_template_chunk)->dist_manner) != _XCALABLEMP_N_DIST_DUPLICATION) {
          return _XCALABLEMP_N_INVALID_RANK;
        }
      }
    }
  }

  return _XCALABLEMP_calc_gmove_nodes_rank(rank_array, nodes);
}

// FIXME does not has complete function for general usage
static void _XCALABLEMP_calc_SENDRECV_index_ref(int n, int target_rank, _XCALABLEMP_array_t *array, int dim_index,
                                                int *lower, int *upper, int *stride) {
  _XCALABLEMP_array_info_t *array_info = &(array->info[dim_index]);
  if ((array_info->align_template_index) == _XCALABLEMP_N_NO_ALIGNED_TEMPLATE) {
    *lower = target_rank * n;
    *upper = ((target_rank + 1) * n) - 1;
    *stride = 1;
  }
  else {
    *lower = 0;
    *upper = n - 1;
    *stride = 1;
  }
}

// FIXME does not has complete function for general usage
static void _XCALABLEMP_gmove_SENDRECV_all2all_2(void *dst_addr, void *src_addr,
                                                 _XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array,
                                                 int type, size_t type_size,
                                                 int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
                                                 int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  int dim = dst_array->dim;
  if (dim != src_array->dim) {
    _XCALABLEMP_fatal("dst/src array should have the same dimension");
  }

  MPI_Status stat;
  MPI_Comm *comm = ((dst_array->align_template)->onto_nodes)->comm;
  int size = ((dst_array->align_template)->onto_nodes)->comm_size;
  int rank = ((dst_array->align_template)->onto_nodes)->comm_rank;

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  unsigned long long buffer_elmts = 1;
  int elmts_base = _XCALABLEMP_M_COUNT_TRIPLETi(dst_lower[0], dst_upper[0], dst_stride[0]);
  int n = elmts_base/size;
  for (int i = 0; i < dim; i++) {
    int dst_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(dst_lower[i], dst_upper[i], dst_stride[i]);
    if (dst_elmts != elmts_base) {
      _XCALABLEMP_fatal("limitation:every dimension should has the same size");
    }

    int src_elmts = _XCALABLEMP_M_COUNT_TRIPLETi(src_lower[i], src_upper[i], src_stride[i]);
    if (src_elmts != elmts_base) {
      _XCALABLEMP_fatal("limitation:every dimension should has the same size");
    }

    buffer_elmts *= n;
  }

  int dst_l[dim], dst_u[dim], dst_s[dim];
  int src_l[dim], src_u[dim], src_s[dim];
  void *pack_buffer = _XCALABLEMP_alloc(buffer_elmts * type_size);
  for(int src_rank = 0; src_rank < size; src_rank++) {
    if(src_rank == rank) {
      // send my data to each node
      for(int dst_rank = 0; dst_rank < size; dst_rank++) {
        if(dst_rank == rank) {
          for (int i = 0; i < dim; i++) {
            _XCALABLEMP_calc_SENDRECV_index_ref(n, dst_rank, dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
            _XCALABLEMP_calc_SENDRECV_index_ref(n, dst_rank, src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
          }

          if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
            _XCALABLEMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, dim, src_l, src_u, src_s, src_dim_acc);
            _XCALABLEMP_unpack_array_GENERAL(dst_addr, pack_buffer, type_size, dim, dst_l, dst_u, dst_s, dst_dim_acc);
          }
          else {
            _XCALABLEMP_pack_array_BASIC(pack_buffer, src_addr, type, dim, src_l, src_u, src_s, src_dim_acc);
            _XCALABLEMP_unpack_array_BASIC(dst_addr, pack_buffer, type, dim, dst_l, dst_u, dst_s, dst_dim_acc);
          }
        }
        else {
          for (int i = 0; i < dim; i++) {
            _XCALABLEMP_calc_SENDRECV_index_ref(n, dst_rank, src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
          }

          if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
            _XCALABLEMP_pack_array_GENERAL(pack_buffer, src_addr, type_size, dim, src_l, src_u, src_s, src_dim_acc);
          }
          else {
            _XCALABLEMP_pack_array_BASIC(pack_buffer, src_addr, type, dim, src_l, src_u, src_s, src_dim_acc);
          }

          MPI_Send(pack_buffer, buffer_elmts, mpi_datatype, dst_rank, _XCALABLEMP_N_MPI_TAG_GMOVE, *comm);
        }
      }
    }
    else {
      MPI_Recv(pack_buffer, buffer_elmts, mpi_datatype, src_rank, _XCALABLEMP_N_MPI_TAG_GMOVE, *comm, &stat);

      for (int i = 0; i < dim; i++) {
        _XCALABLEMP_calc_SENDRECV_index_ref(n, src_rank, dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
      }

      if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
        _XCALABLEMP_unpack_array_GENERAL(dst_addr, pack_buffer, type_size, dim, dst_l, dst_u, dst_s, dst_dim_acc);
      }
      else {
        _XCALABLEMP_unpack_array_BASIC(dst_addr, pack_buffer, type, dim, dst_l, dst_u, dst_s, dst_dim_acc);
      }
    }
  }

  _XCALABLEMP_free(pack_buffer);
}

// FIXME does not has complete function for general usage
void _XCALABLEMP_gmove_SENDRECV_ARRAY_SECTION(_XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array,
                                              int type, size_t type_size, ...) {
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XCALABLEMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  _XCALABLEMP_nodes_t *dst_nodes = (dst_array->align_template)->onto_nodes;
  _XCALABLEMP_nodes_t *src_nodes = (src_array->align_template)->onto_nodes;

  int dst_rank = _XCALABLEMP_calc_SENDRECV_owner(dst_array, dst_l, dst_u, dst_s);
  int src_rank = _XCALABLEMP_calc_SENDRECV_owner(src_array, src_l, src_u, src_s);
  if ((dst_rank != _XCALABLEMP_N_INVALID_RANK) && (src_rank != _XCALABLEMP_N_INVALID_RANK)) {
    // send/recv FIXME limitation: arrays should be distributed by the same nodes
    if (dst_nodes != src_nodes) {
      _XCALABLEMP_fatal("arrays used in a gmove directive should be distributed by the same nodes set");
    }

    // FIXME use execution nodes set
    _XCALABLEMP_nodes_t *comm_nodes = dst_nodes;

    void *recv_buffer = NULL;
    void *send_buffer = NULL;

    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);

    // irecv
    MPI_Request recv_req;
    if (dst_rank == dst_array->align_comm_rank) {
      unsigned long long recv_elmts = 1;
      for (int i = 0; i < dst_dim; i++) {
        recv_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      }
      recv_buffer = _XCALABLEMP_alloc(recv_elmts * type_size);

      MPI_Irecv(recv_buffer, recv_elmts, mpi_datatype, MPI_ANY_SOURCE, _XCALABLEMP_N_MPI_TAG_GMOVE, *(comm_nodes->comm), &recv_req);
    }

    // pack & send
    if (src_rank == src_array->align_comm_rank) {
      unsigned long long send_elmts = 1;
      for (int i = 0; i < src_dim; i++) {
        _XCALABLEMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
        send_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
      }
      send_buffer = _XCALABLEMP_alloc(send_elmts * type_size);
      if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
        _XCALABLEMP_pack_array_GENERAL(send_buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
      }
      else {
        _XCALABLEMP_pack_array_BASIC(send_buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
      }

      MPI_Send(send_buffer, send_elmts, mpi_datatype, dst_rank, _XCALABLEMP_N_MPI_TAG_GMOVE, *(comm_nodes->comm));
      _XCALABLEMP_free(send_buffer);
    }

    // wait & unpack
    if (dst_rank == dst_array->align_comm_rank) {
      for (int i = 0; i < dst_dim; i++) {
        _XCALABLEMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
      }

      MPI_Status recv_stat;
      MPI_Wait(&recv_req, &recv_stat);

      if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
        _XCALABLEMP_unpack_array_GENERAL(dst_addr, recv_buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      else {
        _XCALABLEMP_unpack_array_BASIC(dst_addr, recv_buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      _XCALABLEMP_free(recv_buffer);
    }
  }
  else {
    if (dst_array == src_array) {
      unsigned long long dst_buffer_elmts = 1;
      for (int i = 0; i < dst_dim; i++) {
        _XCALABLEMP_calc_array_local_index_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
        dst_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      }

      unsigned long long src_buffer_elmts = 1;
      for (int i = 0; i < src_dim; i++) {
        _XCALABLEMP_calc_array_local_index_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
        src_buffer_elmts *= _XCALABLEMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
      }

      // alloc buffer
      if (dst_buffer_elmts != src_buffer_elmts) {
        _XCALABLEMP_fatal("wrong assign statement"); // FIXME fix error msg
      }

      void *buffer = _XCALABLEMP_alloc(dst_buffer_elmts * type_size);

      // pack/unpack
      if (type == _XCALABLEMP_N_TYPE_NONBASIC) {
        _XCALABLEMP_pack_array_GENERAL(buffer, src_addr, type_size, src_dim, src_l, src_u, src_s, src_d);
        _XCALABLEMP_unpack_array_GENERAL(dst_addr, buffer, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }
      else {
        _XCALABLEMP_pack_array_BASIC(buffer, src_addr, type, src_dim, src_l, src_u, src_s, src_d);
        _XCALABLEMP_unpack_array_BASIC(dst_addr, buffer, type, dst_dim, dst_l, dst_u, dst_s, dst_d);
      }

      // free buffer
      _XCALABLEMP_free(buffer);
    }
    else {
      if (dst_dim == src_dim) {
        if (dst_dim == 1) {
          _XCALABLEMP_array_info_t *ai = &(dst_array->info[0]);

          _XCALABLEMP_push_comm(src_array->align_comm);
          _XCALABLEMP_gmove_BCAST_ARRAY_SECTION(src_array, type, type_size,
                                                dst_addr, dst_dim, ai->local_lower, ai->local_upper, ai->local_stride, dst_d[0],
                                                src_addr, src_dim, ai->par_lower, ai->par_upper, ai->par_stride, dst_d[0]);
          _XCALABLEMP_pop_n_free_nodes_wo_finalize_comm();
        }
        else if (dst_dim == 2) {
          _XCALABLEMP_gmove_SENDRECV_all2all_2(dst_addr, src_addr,
                                               dst_array, src_array,
                                               type, type_size,
                                               dst_l, dst_u, dst_s, dst_d,
                                               src_l, src_u, src_s, src_d);
        }
        else {
          _XCALABLEMP_fatal("not implemented yet");
        }
      }
      else {
        _XCALABLEMP_fatal("not implemented yet");
      }
    }
  }

  _XCALABLEMP_barrier_EXEC();
}
