/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include <string.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

#define _XMP_SM_GTOL_BLOCK(_i, _m, _w) \
(((_i) - (_m)) % (_w))

#define _XMP_SM_GTOL_CYCLIC(_i, _m, _P) \
(((_i) - (_m)) / (_P))

#define _XMP_SM_GTOL_BLOCK_CYCLIC(_b, _i, _m, _P) \
(((((_i) - (_m)) / (((_P) * (_b)))) * (_b)) + (((_i) - (_m)) % (_b)))

// FIXME not completed
static void _XMP_gtol_array_ref_triplet(_XMP_array_t *array,
                                        int dim_index, int *lower, int *upper, int *stride) {
  _XMP_array_info_t *array_info = &(array->info[dim_index]);
  if (array_info->shadow_type == _XMP_N_SHADOW_FULL) {
    return;
  }

  _XMP_template_t *align_template = array->align_template;

  int align_template_index = array_info->align_template_index;
  if (align_template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
    int align_subscript = array_info->align_subscript;

    int t_lower = *lower - align_subscript,
        t_upper = *upper - align_subscript,
        t_stride = *stride;

    _XMP_template_info_t *ti = &(align_template->info[align_template_index]);
    int template_ser_lower = ti->ser_lower;

    _XMP_template_chunk_t *tc = &(align_template->chunk[align_template_index]);
    int template_par_width = tc->par_width;
    int template_par_nodes_size = (tc->onto_nodes_info)->size;
    int template_par_chunk_width = tc->par_chunk_width;

    // FIXME make a function for doing this
    switch (tc->dist_manner) {
      case _XMP_N_DIST_DUPLICATION:
        // do nothing
        break;
      case _XMP_N_DIST_BLOCK:
        {
          t_lower = _XMP_SM_GTOL_BLOCK(t_lower, template_ser_lower, template_par_chunk_width);
          t_upper = _XMP_SM_GTOL_BLOCK(t_upper, template_ser_lower, template_par_chunk_width);
          // t_stride does not change
        } break;
      case _XMP_N_DIST_CYCLIC:
        {
          t_lower = _XMP_SM_GTOL_CYCLIC(t_lower, template_ser_lower, template_par_nodes_size);
          t_upper = _XMP_SM_GTOL_CYCLIC(t_upper, template_ser_lower, template_par_nodes_size);
          t_stride = 1; // FIXME how implement???
        } break;
      case _XMP_N_DIST_BLOCK_CYCLIC:
        {
          t_lower = _XMP_SM_GTOL_BLOCK_CYCLIC(template_par_width, t_lower, template_ser_lower, template_par_nodes_size);
          t_upper = _XMP_SM_GTOL_BLOCK_CYCLIC(template_par_width, t_upper, template_ser_lower, template_par_nodes_size);
          t_stride = 1; // FIXME how implement???
        } break;
      default:
        _XMP_fatal("wrong distribute manner for normal shadow");
    }

    *lower = t_lower + align_subscript;
    *upper = t_upper + align_subscript;
    *stride = t_stride;
  }

  _XMP_ASSERT(array_info->shadow_type != _XMP_N_SHADOW_FULL);
  switch (array_info->shadow_type) {
    case _XMP_N_SHADOW_NONE:
      // do nothing
      break;
    case _XMP_N_SHADOW_NORMAL:
      switch (array_info->align_manner) {
        case _XMP_N_ALIGN_NOT_ALIGNED:
        case _XMP_N_ALIGN_DUPLICATION:
        case _XMP_N_ALIGN_BLOCK:
          {
            int shadow_size_lo = array_info->shadow_size_lo;
            *lower += shadow_size_lo;
            *upper += shadow_size_lo;
          } break;
        case _XMP_N_ALIGN_CYCLIC:
          // FIXME not supported now
        case _XMP_N_ALIGN_BLOCK_CYCLIC:
          // FIXME not supported now
        default:
          _XMP_fatal("gmove does not support shadow region for cyclic or block-cyclic distribution");
      } break;
    default:
      _XMP_fatal("unknown shadow type");
  }
}

static void _XMP_calc_gmove_rank_array_SCALAR(_XMP_array_t *array, int *ref_index, int *rank_array) {
  _XMP_template_t *template = array->align_template;

  int array_dim = array->dim;
  for (int i = 0; i < array_dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);
    int template_index = ai->align_template_index;
    if (template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
      _XMP_template_chunk_t *chunk = &(template->chunk[ai->align_template_index]);
      int onto_nodes_index = chunk->onto_nodes_index;
      _XMP_ASSERT(array_nodes_index != _XMP_N_NO_ONTO_NODES);

      int array_nodes_index = _XMP_calc_nodes_index_from_inherit_nodes_index(array->array_nodes, onto_nodes_index);
      rank_array[array_nodes_index] = _XMP_calc_template_owner_SCALAR(template, template_index,
                                                                      ref_index[i] + ai->align_subscript);
    }
  }
}

static int _XMP_calc_gmove_array_owner_linear_rank_SCALAR(_XMP_array_t *array, int *ref_index) {
  _XMP_nodes_t *array_nodes = array->array_nodes;
  int array_nodes_dim = array_nodes->dim;
  int rank_array[array_nodes_dim];

  _XMP_calc_gmove_rank_array_SCALAR(array, ref_index, rank_array);

  return _XMP_calc_linear_rank_on_target_nodes(array_nodes, rank_array, _XMP_get_execution_nodes());
}

static _XMP_nodes_ref_t *_XMP_create_gmove_nodes_ref_SCALAR(_XMP_array_t *array, int *ref_index) {
  _XMP_nodes_t *array_nodes = array->array_nodes;
  int array_nodes_dim = array_nodes->dim;
  int rank_array[array_nodes_dim];

  _XMP_calc_gmove_rank_array_SCALAR(array, ref_index, rank_array);

  return _XMP_create_nodes_ref_for_target_nodes(array_nodes, rank_array, _XMP_get_execution_nodes());
}

static void _XMP_gmove_bcast(void *buffer, size_t type_size, unsigned long long count, int root_rank) {
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  MPI_Bcast(buffer, count, mpi_datatype, root_rank, *((MPI_Comm *)exec_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

static void _XMP_gmove_bcast_SCALAR(void *dst_addr, void *src_addr,
                                    size_t type_size, int root_rank) {
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  if (root_rank == (exec_nodes->comm_rank)) {
    memcpy(dst_addr, src_addr, type_size);
  }

  _XMP_gmove_bcast(dst_addr, type_size, 1, root_rank);
}

static unsigned long long _XMP_gmove_bcast_ARRAY(void *dst_addr, int dst_dim,
                                                 int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
                                                 void *src_addr, int src_dim,
                                                 int *src_l, int *src_u, int *src_s, unsigned long long *src_d,
                                                 int type, size_t type_size, int root_rank) {
  unsigned long long dst_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    dst_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  if (root_rank == (exec_nodes->comm_rank)) {
    unsigned long long src_buffer_elmts = 1;
    for (int i = 0; i < src_dim; i++) {
      src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    }

    if (dst_buffer_elmts != src_buffer_elmts) {
      _XMP_fatal("bad assign statement for gmove");
    } else {
      _XMP_pack_array(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
    }
  }

  _XMP_gmove_bcast(buffer, type_size, dst_buffer_elmts, root_rank);

  _XMP_unpack_array(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  _XMP_free(buffer);

  return dst_buffer_elmts;
}

static int _XMP_check_gmove_array_ref_inclusion_SCALAR(_XMP_array_t *array, int array_index, int ref_index) {
  _XMP_ASSERT(!(array->align_template)->is_owner);

  _XMP_array_info_t *ai = &(array->info[array_index]);
  if (ai->align_manner == _XMP_N_ALIGN_NOT_ALIGNED) {
    return _XMP_N_INT_TRUE;
  } else {
    int template_ref_index = ref_index + ai->align_subscript;
    return _XMP_check_template_ref_inclusion(template_ref_index, template_ref_index, 1,
                                             array->align_template, ai->align_template_index);
  }
}

static void _XMP_gmove_localcopy_ARRAY(int type, int type_size,
                                       void *dst_addr, int dst_dim,
                                       int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
                                       void *src_addr, int src_dim,
                                       int *src_l, int *src_u, int *src_s, unsigned long long *src_d) {
  unsigned long long dst_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    dst_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  if (dst_buffer_elmts != src_buffer_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);
  _XMP_pack_array(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
  _XMP_unpack_array(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  _XMP_free(buffer);
}

static int _XMP_sched_gmove_triplet_1(int template_lower, int template_upper, int template_stride,
                                      _XMP_array_t *dst_array, int dst_dim_index,
                                      int *dst_l, int *dst_u, int *dst_s,
                                      int *src_l, int *src_u, int *src_s) {
  int src_lower = *src_l;                         int src_stride = *src_s;
  int dst_lower = *dst_l; int dst_upper = *dst_u; int dst_stride = *dst_s;

  _XMP_array_info_t *ai = &(dst_array->info[dst_dim_index]);
  _XMP_template_t *t = dst_array->align_template;

  int ret = _XMP_N_INT_TRUE;
  int align_template_index = ai->align_template_index;
  if (align_template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
    _XMP_template_info_t *ti = &(t->info[align_template_index]);
    _XMP_template_chunk_t *tc = &(t->chunk[align_template_index]);
    int align_subscript = ai->align_subscript;

    // calc dst_l, dst_u, dst_s (dst_s does not change)
    switch (tc->dist_manner) {
      case _XMP_N_DIST_DUPLICATION:
        return _XMP_N_INT_TRUE;
      case _XMP_N_DIST_BLOCK:
      case _XMP_N_DIST_CYCLIC:
        {
          // FIXME consider when stride is not 1
          ret = _XMP_sched_loop_template_width_1(dst_lower - align_subscript,
                                                 dst_upper - align_subscript,
                                                 dst_stride - align_subscript,
                                                 dst_l, dst_u, dst_s,
                                                 template_lower, template_upper, template_stride);
          *dst_l += align_subscript;
          *dst_u += align_subscript;
        } break;
      case _XMP_N_DIST_BLOCK_CYCLIC:
        {
          // FIXME consider when stride is not 1
          ret = _XMP_sched_loop_template_width_N(dst_lower - align_subscript,
                                                 dst_upper - align_subscript,
                                                 dst_stride - align_subscript,
                                                 dst_l, dst_u, dst_s,
                                                 template_lower, template_upper, template_stride,
                                                 tc->par_width, ti->ser_lower, ti->ser_upper);
          *dst_l += align_subscript;
          *dst_u += align_subscript;
        } break;
      default:
        _XMP_fatal("unknown distribution manner");
    }

    // calc src_l, src_u, src_s
    *src_l = (src_stride * ((*dst_l - dst_lower) / dst_stride)) + src_lower;
    *src_u = (src_stride * ((*dst_u - dst_lower) / dst_stride)) + src_lower;
    *src_s = *dst_s; // FIXME consider when stride is not 1, how implement???
  } // FIXME else how implement???

  return ret;
}

static int _XMP_calc_global_index_HOMECOPY(_XMP_array_t *dst_array, int dst_dim_index,
                                           int *dst_l, int *dst_u, int *dst_s,
                                           int *src_l, int *src_u, int *src_s) {
  int align_template_index = dst_array->info[dst_dim_index].align_template_index;
  if (align_template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
    _XMP_template_chunk_t *tc = &((dst_array->align_template)->chunk[align_template_index]);
    return _XMP_sched_gmove_triplet_1(tc->par_lower, tc->par_upper, tc->par_stride,
                                      dst_array, dst_dim_index,
                                      dst_l, dst_u, dst_s,
                                      src_l, src_u, src_s);
  } else {
    // FIXME else how implement???
    return _XMP_N_INT_TRUE;
  }
}

static int _XMP_calc_global_index_BCAST(int dst_dim, int *dst_l, int *dst_u, int *dst_s,
                                        _XMP_array_t *src_array, int *src_array_nodes_ref, int *src_l, int *src_u, int *src_s) {
  _XMP_template_t *template = src_array->align_template;

  int dst_dim_index = 0;
  int array_dim = src_array->dim;
  for (int i = 0; i < array_dim; i++) {
    int template_index = src_array->info[i].align_template_index;
    if (template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
      int onto_nodes_index = template->chunk[template_index].onto_nodes_index;
      _XMP_ASSERT(onto_nodes_index != _XMP_N_NO_ONTO_NODES);

      int array_nodes_index = _XMP_calc_nodes_index_from_inherit_nodes_index(src_array->array_nodes, onto_nodes_index);
      int rank = src_array_nodes_ref[array_nodes_index];

      // calc template info
      int template_lower, template_upper, template_stride;
      if (!_XMP_calc_template_par_triplet(template, template_index, rank, &template_lower, &template_upper, &template_stride)) {
        return _XMP_N_INT_FALSE;
      }

      do {
        if (_XMP_M_COUNT_TRIPLETi(dst_l[dst_dim_index], dst_u[dst_dim_index], dst_s[dst_dim_index]) != 1) {
          if (_XMP_sched_gmove_triplet_1(template_lower, template_upper, template_stride,
                                         src_array, i,
                                         &(src_l[i]), &(src_u[i]), &(src_s[i]),
                                         &(dst_l[dst_dim_index]), &(dst_u[dst_dim_index]), &(dst_s[dst_dim_index]))) {
            dst_dim_index++;
            break;
          } else {
            return _XMP_N_INT_FALSE;
          }
        } else if (dst_dim_index < dst_dim) {
          dst_dim_index++;
        } else {
          _XMP_fatal("bad assign statement for gmove");
        }
      } while (1);
    }
  }

  return _XMP_N_INT_TRUE;
}

static void _XMP_sendrecv_ARRAY(unsigned long long gmove_total_elmts,
                                int type, int type_size, MPI_Datatype *mpi_datatype,
                                _XMP_array_t *dst_array, int *dst_array_nodes_ref,
                                int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
                                _XMP_array_t *src_array, int *src_array_nodes_ref,
                                int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  void *dst_addr = *(dst_array->array_addr_p);
  void *src_addr = *(src_array->array_addr_p);
  int dst_dim = dst_array->dim;
  int src_dim = src_array->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  _XMP_nodes_ref_t *dst_ref = _XMP_create_nodes_ref_for_target_nodes(dst_array_nodes, dst_array_nodes_ref, exec_nodes);
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int *dst_ranks = _XMP_alloc(sizeof(int) * dst_shrink_nodes_size);
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  // calc src_ranks
  _XMP_nodes_ref_t *src_ref = _XMP_create_nodes_ref_for_target_nodes(src_array_nodes, src_array_nodes_ref, exec_nodes);
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int *src_ranks = _XMP_alloc(sizeof(int) * src_shrink_nodes_size);
  if (src_shrink_nodes_size == 1) {
    src_ranks[0] = _XMP_calc_linear_rank(src_ref->nodes, src_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(src_ref->nodes, src_ranks, src_ref->ref, src_shrink_nodes_size);
  }

  // recv phase
  void *recv_buffer = NULL;
  int wait_recv = _XMP_N_INT_FALSE;
  MPI_Request gmove_request;
  for (int i = 0; i < dst_shrink_nodes_size; i++) {
    if (dst_ranks[i] == exec_rank) {
      wait_recv = _XMP_N_INT_TRUE;

      int src_rank;
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        src_rank = src_ranks[i];
      } else {
        src_rank = src_ranks[i % src_shrink_nodes_size];
      }

      recv_buffer = _XMP_alloc(gmove_total_elmts * type_size);
      MPI_Irecv(recv_buffer, gmove_total_elmts, *mpi_datatype, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
    }
  }

  // send phase
  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      void *send_buffer = _XMP_alloc(gmove_total_elmts * type_size);
      for (int j = 0; j < src_dim; j++) {
        _XMP_gtol_array_ref_triplet(src_array, j, &(src_lower[j]), &(src_upper[j]), &(src_stride[j]));
      }
      _XMP_pack_array(send_buffer, src_addr, type, type_size, src_dim, src_lower, src_upper, src_stride, src_dim_acc);

      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        if (i < dst_shrink_nodes_size) {
          MPI_Send(send_buffer, gmove_total_elmts, *mpi_datatype, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
        }
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(send_buffer, gmove_total_elmts, *mpi_datatype, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm,
                    requests + request_count);
          request_count++;
        }

        MPI_Waitall(request_size, requests, MPI_STATUSES_IGNORE);
        _XMP_free(requests);
      }

      _XMP_free(send_buffer);
    }
  }

  // wait recv phase
  if (wait_recv) {
    MPI_Wait(&gmove_request, MPI_STATUS_IGNORE);
    for (int i = 0; i < dst_dim; i++) {
      _XMP_gtol_array_ref_triplet(dst_array, i, &(dst_lower[i]), &(dst_upper[i]), &(dst_stride[i]));
    }
    _XMP_unpack_array(dst_addr, recv_buffer, type, type_size, dst_dim, dst_lower, dst_upper, dst_stride, dst_dim_acc);
    _XMP_free(recv_buffer);
  }

  _XMP_free(dst_ranks);
  _XMP_free(src_ranks);
  _XMP_free(dst_ref);
  _XMP_free(src_ref);
}

// ----- gmove scalar to scalar --------------------------------------------------------------------------------------------------
void _XMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, _XMP_array_t *array, ...) {
  int type_size = array->type_size;

  if(_XMP_IS_SINGLE) {
    memcpy(dst_addr, src_addr, type_size);
    return;
  }

  va_list args;
  va_start(args, array);
  int root_rank;
  {
    int array_dim = array->dim;
    int ref_index[array_dim];
    for (int i = 0; i < array_dim; i++) {
      ref_index[i] = va_arg(args, int);
    }
    root_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, ref_index);
  }
  va_end(args);

  // broadcast
  _XMP_gmove_bcast_SCALAR(dst_addr, src_addr, type_size, root_rank);
}

int _XMP_gmove_HOMECOPY_SCALAR(_XMP_array_t *array, ...) {
  if (!array->is_allocated) {
    return _XMP_N_INT_FALSE;
  }

  if (_XMP_IS_SINGLE) {
    return _XMP_N_INT_TRUE;
  }

  _XMP_ASSERT((array->align_template)->is_distributed);
  _XMP_ASSERT((array->align_template)->is_owner);

  va_list args;
  va_start(args, array);
  int execHere = _XMP_N_INT_TRUE;
  int ref_dim = array->dim;
  for (int i = 0; i < ref_dim; i++) {
    int ref_index = va_arg(args, int);

    execHere = execHere && _XMP_check_gmove_array_ref_inclusion_SCALAR(array, i, ref_index);
  }
  va_end(args);

  return execHere;
}

void _XMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr,
                                _XMP_array_t *dst_array, _XMP_array_t *src_array, ...) {
  _XMP_ASSERT(dst_array->type_size == src_array->type_size); // FIXME checked by compiler
  size_t type_size = dst_array->type_size;

  if(_XMP_IS_SINGLE) {
    memcpy(dst_addr, src_addr, type_size);
    return;
  }

  va_list args;
  va_start(args, src_array);
  _XMP_nodes_ref_t *dst_ref;
  {
    int dst_array_dim = dst_array->dim;
    int dst_ref_index[dst_array_dim];
    for (int i = 0; i < dst_array_dim; i++) {
      dst_ref_index[i] = va_arg(args, int);
    }
    dst_ref = _XMP_create_gmove_nodes_ref_SCALAR(dst_array, dst_ref_index);
  }

  if (dst_ref == NULL) {
    goto END_GMOVE;
  }

  _XMP_nodes_ref_t *src_ref;
  {
    int src_array_dim = src_array->dim;
    int src_ref_index[src_array_dim];
    for (int i = 0; i < src_array_dim; i++) {
      src_ref_index[i] = va_arg(args, int);
    }
    src_ref = _XMP_create_gmove_nodes_ref_SCALAR(src_array, src_ref_index);
  }
  va_end(args);

  if (src_ref == NULL) {
    _XMP_free(dst_ref);
    goto END_GMOVE;
  }

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int *dst_ranks = _XMP_alloc(sizeof(int) * dst_shrink_nodes_size);
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  // calc src_ranks
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int *src_ranks = _XMP_alloc(sizeof(int) * src_shrink_nodes_size);
  if (src_shrink_nodes_size == 1) {
    src_ranks[0] = _XMP_calc_linear_rank(src_ref->nodes, src_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(src_ref->nodes, src_ranks, src_ref->ref, src_shrink_nodes_size);
  }

  int wait_recv = _XMP_N_INT_FALSE;
  MPI_Request gmove_request;
  for (int i = 0; i < dst_shrink_nodes_size; i++) {
    if (dst_ranks[i] == exec_rank) {
      wait_recv = _XMP_N_INT_TRUE;

      int src_rank;
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        src_rank = src_ranks[i];
      } else {
        src_rank = src_ranks[i % src_shrink_nodes_size];
      }

      MPI_Irecv(dst_addr, type_size, MPI_BYTE, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
    }
  }

  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        if (i < dst_shrink_nodes_size) {
          MPI_Send(src_addr, type_size, MPI_BYTE, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
        }
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(src_addr, type_size, MPI_BYTE, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm, requests + request_count);
          request_count++;
        }

        MPI_Waitall(request_size, requests, MPI_STATUSES_IGNORE);
        _XMP_free(requests);
      }
    }
  }

  if (wait_recv) {
    MPI_Wait(&gmove_request, MPI_STATUS_IGNORE);
  }

  _XMP_free(dst_ranks);
  _XMP_free(src_ranks);
  _XMP_free(dst_ref);
  _XMP_free(src_ref);

END_GMOVE:
  return;
}

// ----- gmove vector to vector --------------------------------------------------------------------------------------------------
void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...) {
  // skip counting elmts: _XMP_gmove_localcopy_ARRAY() counts elmts

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
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
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
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  _XMP_gmove_localcopy_ARRAY(type, type_size,
                             dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                             src_addr, src_dim, src_l, src_u, src_s, src_d);
}

void _XMP_gmove_BCAST_ARRAY(_XMP_array_t *src_array, int type, size_t type_size, ...) {
  unsigned long long gmove_total_elmts = 0;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = *(src_array->array_addr_p);
  int src_dim = src_array->dim;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    gmove_total_elmts = dst_total_elmts;
  }

  if (_XMP_IS_SINGLE) {
    for (int i = 0; i < src_dim; i++) {
      _XMP_gtol_array_ref_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    }

    _XMP_gmove_localcopy_ARRAY(type, type_size,
                               dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                               src_addr, src_dim, src_l, src_u, src_s, src_d);
    return;
  }

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  _XMP_nodes_t *array_nodes = src_array->array_nodes;
  int array_nodes_dim = array_nodes->dim;
  int array_nodes_ref[array_nodes_dim];
  for (int i = 0; i < array_nodes_dim; i++) {
    array_nodes_ref[i] = 0;
  }

  int dst_lower[dst_dim], dst_upper[dst_dim], dst_stride[dst_dim];
  int src_lower[src_dim], src_upper[src_dim], src_stride[src_dim];
  do {
    for (int i = 0; i < dst_dim; i++) {
      dst_lower[i] = dst_l[i]; dst_upper[i] = dst_u[i]; dst_stride[i] = dst_s[i];
    }

    for (int i = 0; i < src_dim; i++) {
      src_lower[i] = src_l[i]; src_upper[i] = src_u[i]; src_stride[i] = src_s[i];
    }

    if (_XMP_calc_global_index_BCAST(dst_dim, dst_lower, dst_upper, dst_stride,
                                     src_array, array_nodes_ref, src_lower, src_upper, src_stride)) {
      int root_rank = _XMP_calc_linear_rank_on_target_nodes(array_nodes, array_nodes_ref, exec_nodes);
      if (root_rank == (exec_nodes->comm_rank)) {
        for (int i = 0; i < src_dim; i++) {
          _XMP_gtol_array_ref_triplet(src_array, i, &(src_lower[i]), &(src_upper[i]), &(src_stride[i]));
        }
      }

      gmove_total_elmts -= _XMP_gmove_bcast_ARRAY(dst_addr, dst_dim, dst_lower, dst_upper, dst_stride, dst_d,
                                                  src_addr, src_dim, src_lower, src_upper, src_stride, src_d,
                                                  type, type_size, root_rank);

      _XMP_ASSERT(gmove_total_elmts >= 0);
      if (gmove_total_elmts == 0) {
        return;
      }
    }
  } while (_XMP_calc_next_next_rank(array_nodes, array_nodes_ref));
}

void _XMP_gmove_HOMECOPY_ARRAY(_XMP_array_t *dst_array, int type, size_t type_size, ...) {
  if (!dst_array->is_allocated) {
    return;
  }

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = *(dst_array->array_addr_p);
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  if (_XMP_IS_SINGLE) {
    for (int i = 0; i < dst_dim; i++) {
      _XMP_gtol_array_ref_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    }

    _XMP_gmove_localcopy_ARRAY(type, type_size,
                               dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                               src_addr, src_dim, src_l, src_u, src_s, src_d);
    return;
  }

  // calc index ref
  int src_dim_index = 0;
  unsigned long long dst_buffer_elmts = 1;
  unsigned long long src_buffer_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    int dst_elmts = _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_elmts == 1) {
      if(!_XMP_check_gmove_array_ref_inclusion_SCALAR(dst_array, i, dst_l[i])) {
        return;
      }
    } else {
      dst_buffer_elmts *= dst_elmts;

      int src_elmts;
      do {
        src_elmts = _XMP_M_COUNT_TRIPLETi(src_l[src_dim_index], src_u[src_dim_index], src_s[src_dim_index]);
        if (src_elmts != 1) {
          break;
        } else if (src_dim_index < src_dim) {
          src_dim_index++;
        } else {
          _XMP_fatal("bad assign statement for gmove");
        }
      } while (1);

      if (_XMP_calc_global_index_HOMECOPY(dst_array, i,
                                          &(dst_l[i]), &(dst_u[i]), &(dst_s[i]),
                                          &(src_l[src_dim_index]), &(src_u[src_dim_index]), &(src_s[src_dim_index]))) {
        src_buffer_elmts *= src_elmts;
        src_dim_index++;
      } else {
        return;
      }
    }

    _XMP_gtol_array_ref_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  for (int i = src_dim_index; i < src_dim; i++) {
    src_buffer_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  // alloc buffer
  if (dst_buffer_elmts != src_buffer_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);
  _XMP_pack_array(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
  _XMP_unpack_array(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  _XMP_free(buffer);
}

void _XMP_gmove_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
                               int type, size_t type_size, ...) {
  unsigned long long gmove_total_elmts = 0;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = *(dst_array->array_addr_p);
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = *(src_array->array_addr_p);
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    gmove_total_elmts = dst_total_elmts;
  }

  if (_XMP_IS_SINGLE) {
    for (int i = 0; i < dst_dim; i++) {
      _XMP_gtol_array_ref_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    }

    for (int i = 0; i < src_dim; i++) {
      _XMP_gtol_array_ref_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    }

    _XMP_gmove_localcopy_ARRAY(type, type_size,
                               dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                               src_addr, src_dim, src_l, src_u, src_s, src_d);
    return;
  }

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  int dst_array_nodes_dim = dst_array_nodes->dim;
  int dst_array_nodes_ref[dst_array_nodes_dim];
  for (int i = 0; i < dst_array_nodes_dim; i++) {
    dst_array_nodes_ref[i] = 0;
  }

  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  int src_array_nodes_dim = src_array_nodes->dim;
  int src_array_nodes_ref[src_array_nodes_dim];

  int dst_lower[dst_dim], dst_upper[dst_dim], dst_stride[dst_dim];
  int src_lower[src_dim], src_upper[src_dim], src_stride[src_dim];
  do {
    for (int i = 0; i < dst_dim; i++) {
      dst_lower[i] = dst_l[i]; dst_upper[i] = dst_u[i]; dst_stride[i] = dst_s[i];
    }

    for (int i = 0; i < src_dim; i++) {
      src_lower[i] = src_l[i]; src_upper[i] = src_u[i]; src_stride[i] = src_s[i];
    }

    if (_XMP_calc_global_index_BCAST(src_dim, src_lower, src_upper, src_stride,
                                     dst_array, dst_array_nodes_ref, dst_lower, dst_upper, dst_stride)) {
      for (int i = 0; i < src_array_nodes_dim; i++) {
        src_array_nodes_ref[i] = 0;
      }

      int recv_lower[dst_dim], recv_upper[dst_dim], recv_stride[dst_dim];
      int send_lower[src_dim], send_upper[src_dim], send_stride[src_dim];
      do {
        for (int i = 0; i < dst_dim; i++) {
          recv_lower[i] = dst_lower[i]; recv_upper[i] = dst_upper[i]; recv_stride[i] = dst_stride[i];
        }

        for (int i = 0; i < src_dim; i++) {
          send_lower[i] = src_lower[i]; send_upper[i] = src_upper[i]; send_stride[i] = src_stride[i];
        }

        if (_XMP_calc_global_index_BCAST(dst_dim, recv_lower, recv_upper, recv_stride,
                                         src_array, src_array_nodes_ref, send_lower, send_upper, send_stride)) {
          _XMP_sendrecv_ARRAY(gmove_total_elmts,
                              type, type_size, &mpi_datatype,
                              dst_array, dst_array_nodes_ref,
                              recv_lower, recv_upper, recv_stride, dst_d,
                              src_array, src_array_nodes_ref,
                              send_lower, send_upper, send_stride, src_d);
        }
      } while (_XMP_calc_next_next_rank(src_array_nodes, src_array_nodes_ref));
    }
  } while (_XMP_calc_next_next_rank(dst_array_nodes, dst_array_nodes_ref));
}
