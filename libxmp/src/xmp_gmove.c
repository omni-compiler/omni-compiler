/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "xmp.h"
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

#define _XMP_SM_GTOL_BLOCK(_i, _m, _w) \
(((_i) - (_m)) % (_w))

#define _XMP_SM_GTOL_CYCLIC(_i, _m, _P) \
(((_i) - (_m)) / (_P))

#define _XMP_SM_GTOL_BLOCK_CYCLIC(_b, _i, _m, _P) \
(((((_i) - (_m)) / (((_P) * (_b)))) * (_b)) + (((_i) - (_m)) % (_b)))

void _XMP_gtol_array_ref_triplet(_XMP_array_t *array,
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
          t_stride = t_stride / template_par_nodes_size;
        } break;
      case _XMP_N_DIST_BLOCK_CYCLIC:
        {
          t_lower = _XMP_SM_GTOL_BLOCK_CYCLIC(template_par_width, t_lower, template_ser_lower, template_par_nodes_size);
          t_upper = _XMP_SM_GTOL_BLOCK_CYCLIC(template_par_width, t_upper, template_ser_lower, template_par_nodes_size);
          // t_stride does not change (in block)
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

int _XMP_calc_gmove_array_owner_linear_rank_SCALAR(_XMP_array_t *array, int *ref_index) {
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

void _XMP_gmove_bcast_SCALAR(void *dst_addr, void *src_addr,
                                    size_t type_size, int root_rank) {
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  if (root_rank == (exec_nodes->comm_rank)) {
    memcpy(dst_addr, src_addr, type_size);
  }

  _XMP_gmove_bcast(dst_addr, type_size, 1, root_rank);
}

unsigned long long _XMP_gmove_bcast_ARRAY(void *dst_addr, int dst_dim,
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

int _XMP_check_gmove_array_ref_inclusion_SCALAR(_XMP_array_t *array, int array_index, int ref_index) {
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

void _XMP_gmove_localcopy_ARRAY(int type, int type_size,
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

static int _XMP_sched_gmove_triplet(int template_lower, int template_upper, int template_stride,
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
    *src_s = src_stride * (*dst_s / dst_stride);
  }

  return ret;
}

int _XMP_calc_global_index_HOMECOPY(_XMP_array_t *dst_array, int dst_dim_index,
				    int *dst_l, int *dst_u, int *dst_s,
				    int *src_l, int *src_u, int *src_s) {
  int align_template_index = dst_array->info[dst_dim_index].align_template_index;
  if (align_template_index != _XMP_N_NO_ALIGN_TEMPLATE) {
    _XMP_template_chunk_t *tc = &((dst_array->align_template)->chunk[align_template_index]);
    return _XMP_sched_gmove_triplet(tc->par_lower, tc->par_upper, tc->par_stride,
                                    dst_array, dst_dim_index,
                                    dst_l, dst_u, dst_s,
                                    src_l, src_u, src_s);
  } else {
    return _XMP_N_INT_TRUE;
  }
}

int _XMP_calc_global_index_BCAST(int dst_dim, int *dst_l, int *dst_u, int *dst_s,
                                 _XMP_array_t *src_array, int *src_array_nodes_ref, int *src_l, int *src_u, int *src_s) {
  _XMP_template_t *template = src_array->align_template;

  int dst_dim_index = 0;
  int array_dim = src_array->dim;
  for (int i = 0; i < array_dim; i++) {
    if (_XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]) == 1) {
      continue;
    }

    int template_index = src_array->info[i].align_template_index;
    if (template_index == _XMP_N_NO_ALIGN_TEMPLATE) {
      do {
        if (_XMP_M_COUNT_TRIPLETi(dst_l[dst_dim_index], dst_u[dst_dim_index], dst_s[dst_dim_index]) != 1) {
          dst_dim_index++;
          break;
        } else if (dst_dim_index < dst_dim) {
          dst_dim_index++;
        } else {
          _XMP_fatal("bad assign statement for gmove");
        }
      } while (1);
    } else {
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
          if (_XMP_sched_gmove_triplet(template_lower, template_upper, template_stride,
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

void _XMP_sendrecv_ARRAY(int type, int type_size, MPI_Datatype *mpi_datatype,
                         _XMP_array_t *dst_array, int *dst_array_nodes_ref,
                         int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
                         _XMP_array_t *src_array, int *src_array_nodes_ref,
                         int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  void *dst_addr = dst_array->array_addr_p;
  void *src_addr = src_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int src_dim = src_array->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  _XMP_nodes_ref_t *dst_ref = _XMP_create_nodes_ref_for_target_nodes(dst_array_nodes, dst_array_nodes_ref, exec_nodes);
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int dst_ranks[dst_shrink_nodes_size];
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  unsigned long long total_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_lower[i], dst_upper[i], dst_stride[i]);
  }

  // calc src_ranks
  _XMP_nodes_ref_t *src_ref = _XMP_create_nodes_ref_for_target_nodes(src_array_nodes, src_array_nodes_ref, exec_nodes);
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int src_ranks[src_shrink_nodes_size];
  if (src_shrink_nodes_size == 1) {
    src_ranks[0] = _XMP_calc_linear_rank(src_ref->nodes, src_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(src_ref->nodes, src_ranks, src_ref->ref, src_shrink_nodes_size);
  }

  unsigned long long src_total_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_lower[i], src_upper[i], src_stride[i]);
  }

  if (total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
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

      recv_buffer = _XMP_alloc(total_elmts * type_size);
      MPI_Irecv(recv_buffer, total_elmts, *mpi_datatype, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
    }
  }

  // send phase
  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      void *send_buffer = _XMP_alloc(total_elmts * type_size);
      for (int j = 0; j < src_dim; j++) {
        _XMP_gtol_array_ref_triplet(src_array, j, &(src_lower[j]), &(src_upper[j]), &(src_stride[j]));
      }
      _XMP_pack_array(send_buffer, src_addr, type, type_size, src_dim, src_lower, src_upper, src_stride, src_dim_acc);
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        if (i < dst_shrink_nodes_size) {
          MPI_Send(send_buffer, total_elmts, *mpi_datatype, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
        }
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size - 1, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(send_buffer, total_elmts, *mpi_datatype, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm,
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
  _XMP_ASSERT(dst_array->type_size == src_array->type_size);
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

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int dst_ranks[dst_shrink_nodes_size];
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  // calc src_ranks
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int src_ranks[src_shrink_nodes_size];
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

  _XMP_free(dst_ref);
  _XMP_free(src_ref);
}

// ----- gmove vector to vector --------------------------------------------------------------------------------------------------
void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...) {
  // skip counting elmts: _XMP_gmove_localcopy_ARRAY() counts elmts

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_u[i] = src_l[i] + (size - 1) * src_s[i];
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
  }

  va_end(args);

  _XMP_gmove_localcopy_ARRAY(type, type_size,
                             dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                             src_addr, src_dim, src_l, src_u, src_s, src_d);
}

static _Bool is_same_axis(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].align_template_index 
        != adesc1->info[i].align_template_index) return false;
  }

  return true;
}

static _Bool is_same_offset(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].align_subscript
        != adesc1->info[i].align_subscript) return false;
  }

  return true;
}

static _Bool is_same_alignmanner(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  _XMP_template_t *t0 = (_XMP_template_t *)adesc0->align_template;
  _XMP_template_t *t1 = (_XMP_template_t *)adesc1->align_template;
  int taxis0, taxis1, naxis0, naxis1, nsize0, nsize1;

  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    int idim0 = adesc0->info[i].align_template_index;
    int idim1 = adesc1->info[i].align_template_index;
    if ((adesc0->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)
        || (adesc1->info[i].align_manner == _XMP_N_ALIGN_DUPLICATION)){
       return false;
    }else if (adesc0->info[i].align_manner != adesc1->info[i].align_manner){
       return false;
    }else{
       if(adesc0->info[i].align_manner==_XMP_N_ALIGN_BLOCK_CYCLIC
         && t0->chunk[idim0].par_width != t1->chunk[idim1].par_width){
         return false;
       }else if(adesc0->info[i].align_manner==_XMP_N_ALIGN_GBLOCK){
         xmp_align_axis(adesc0, i+1, &taxis0);
         xmp_align_axis(adesc1, i+1, &taxis1);
         xmp_dist_axis(t0, taxis0, &naxis0);
         xmp_dist_axis(t1, taxis1, &naxis1);
         xmp_nodes_size(adesc0->array_nodes, naxis0, &nsize0);
         xmp_nodes_size(adesc1->array_nodes, naxis1, &nsize1);
         int map0[nsize0], map1[nsize1];
         xmp_dist_gblockmap(t0, naxis0, map0);
         xmp_dist_gblockmap(t1, naxis1, map1);
         if (nsize0 == nsize1){
           for(int ii=0; ii<nsize0; ii++){
             if (map0[ii] != map1[ii]){
                return false;
             }
           }
         }else{
           return false;
         }
       }
    }
  }

  return true;

}


/*static _Bool is_same_elmts(int dst_dim, int *dst_l, int *dst_u, int *dst_s, int src_dim, int *src_l, int *src_u, int *src_s)
{
  if (dst_dim != src_dim) return false;

  for (int i = 0; i < dst_dim; i++) {
    if (dst_l[i] != src_l[i] || dst_u[i] != src_u[i] || dst_s[i] != src_s[i]) return false;
  }

  return true;
}*/

static _Bool is_same_array_shape(_XMP_array_t *adesc0, _XMP_array_t *adesc1)
{
  if (adesc0->dim != adesc1->dim) return false;

  for (int i = 0; i < adesc0->dim; i++) {
    if (adesc0->info[i].ser_lower != adesc1->info[i].ser_lower ||
	adesc0->info[i].ser_upper != adesc1->info[i].ser_upper) return false;
  }

  return true;
}

static _Bool is_same_template_shape(_XMP_template_t *tdesc0, _XMP_template_t *tdesc1)
{
  if (tdesc0->dim != tdesc1->dim) return false;

  for (int i = 0; i < tdesc0->dim; i++) {
    if (tdesc0->info[i].ser_lower != tdesc1->info[i].ser_lower ||
        tdesc0->info[i].ser_upper != tdesc1->info[i].ser_upper) return false;
  }

  return true;
}

static _Bool is_whole(_XMP_gmv_desc_t *gmv_desc)
{
  _XMP_array_t *adesc = gmv_desc->a_desc;

  for (int i = 0; i < adesc->dim; i++){
    if (gmv_desc->lb[i] == 0 && gmv_desc->ub[i] == 0 && gmv_desc->st[i] == 0) continue;
    if (adesc->info[i].ser_lower != gmv_desc->lb[i] ||
	adesc->info[i].ser_upper != gmv_desc->ub[i] ||
	gmv_desc->st[i] != 1) return false;
  }

  return true;
}

static _Bool is_one_block(_XMP_array_t *adesc)
{
  int cnt = 0;

  for (int i = 0; i < adesc->dim; i++) {
    if (adesc->info[i].align_manner == _XMP_N_ALIGN_BLOCK) cnt++;
    else if (adesc->info[i].align_manner != _XMP_N_ALIGN_NOT_ALIGNED) return false;
  }
  
  if (cnt != 1) return false;
  else return true;
}

//#define DBG 1

#ifdef DBG
#include <stdio.h>
static void xmpf_dbg_printf(char *fmt, ...)
{
  char buf[512];
  va_list args;

  va_start(args, fmt);
  vsprintf(buf, fmt, args);
  va_end(args);

  printf("[%d] %s", _XMP_world_rank, buf);
  fflush(stdout);
}
#endif

void _XMP_gmove_calc_unit_size(_XMP_array_t *dst_array, _XMP_array_t *src_array, 
                               unsigned long long *alltoall_unit_size, 
                               unsigned long long *dst_pack_unit_size, 
                               unsigned long long *src_pack_unit_size, 
                               unsigned long long *dst_ser_size, 
                               unsigned long long *src_ser_size, 
                               int dst_block_dim, int src_block_dim){

  int dst_dim=dst_array->dim, src_dim=src_array->dim;
  int dst_chunk_size[_XMP_N_MAX_DIM], src_chunk_size[_XMP_N_MAX_DIM];

  *alltoall_unit_size=1;
  *dst_pack_unit_size=1;
  *src_pack_unit_size=1;
  *dst_ser_size=1;
  *src_ser_size=1;

  if ((_XMPF_running == 1) && (_XMPC_running == 0)){
    for(int i=0; i<dst_dim; i++){
      if(i==dst_block_dim){
        dst_chunk_size[i]=dst_array->info[i].par_size;
      }else if(i==src_block_dim){
        dst_chunk_size[i]=src_array->info[i].par_size;
      }else{
        dst_chunk_size[i]=dst_array->info[i].ser_size;
      }
    }
    for(int i=0; i<src_dim; i++){
      if(i==dst_block_dim){
        src_chunk_size[i]=dst_array->info[i].par_size;
      }else if(i==src_block_dim){
        src_chunk_size[i]=src_array->info[i].par_size;
      }else{
        src_chunk_size[i]=src_array->info[i].ser_size;
      }
      *alltoall_unit_size *= src_chunk_size[i];
    }
    for(int i=0; i<dst_block_dim+1; i++){
      *src_pack_unit_size *= src_chunk_size[i];
      *src_ser_size *= src_array->info[i].par_size;
    }
    for(int i=0; i<src_block_dim+1; i++){
      *dst_pack_unit_size *= dst_chunk_size[i];
      *dst_ser_size *= dst_array->info[i].par_size;
    }
  }

  if ((_XMPF_running == 0) && (_XMPC_running == 1)){
    for(int i=dst_dim-1; i> -1; i--){
      if(i==dst_block_dim){
        dst_chunk_size[i]=dst_array->info[i].par_size;
      }else if(i==src_block_dim){
        dst_chunk_size[i]=src_array->info[i].par_size;
      }else{
        dst_chunk_size[i]=dst_array->info[i].ser_size;
      }
    }
    for(int i=src_dim-1; i>-1; i--){
      if(i==dst_block_dim){
        src_chunk_size[i]=dst_array->info[i].par_size;
      }else if(i==src_block_dim){
        src_chunk_size[i]=src_array->info[i].par_size;
      }else{
        src_chunk_size[i]=src_array->info[i].ser_size;
      }
      *alltoall_unit_size *= src_chunk_size[i];
    }
    for(int i=src_array->dim-1; i>dst_block_dim-1; i--){
      *src_pack_unit_size *= src_chunk_size[i];
      *src_ser_size *= src_array->info[i].par_size;
    }
    for(int i=src_array->dim-1; i>src_block_dim-1; i--){
      *dst_pack_unit_size *= dst_chunk_size[i];
      *dst_ser_size *= dst_array->info[i].par_size;
    }
  }

}

static _Bool _XMP_gmove_transpose(_XMP_gmv_desc_t *gmv_desc_leftp,
				  _XMP_gmv_desc_t *gmv_desc_rightp)
{
  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

  int nnodes, type_size;

  int dst_block_dim=-1, src_block_dim=-1;
  int dst_block_dim_count=0, src_block_dim_count=0;
  int dst_dim=dst_array->dim, src_dim=src_array->dim;

  void *sendbuf, *recvbuf;
  unsigned long long bufsize, dst_div, src_div;
  unsigned long long dst_pack_unit_size, src_pack_unit_size, alltoall_unit_size=1;
  unsigned long long dst_ser_size, src_ser_size;

#ifdef DBG
  xmpf_dbg_printf("_XMPF_gmove_transpose\n");
#endif

  nnodes = dst_array->align_template->onto_nodes->comm_size;

  // No Shadow
  if (dst_array->info[0].shadow_size_lo != 0 ||
      dst_array->info[0].shadow_size_hi != 0 ||
      src_array->info[0].shadow_size_lo != 0 ||
      src_array->info[0].shadow_size_hi != 0) return false;

  // Dividable by the number of nodes
  if (dst_array->info[0].ser_size % nnodes != 0) return false;

  for(int i=0; i<dst_array->dim; i++){
    if (dst_array->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
       dst_block_dim=i;
       dst_block_dim_count++;
       if (dst_block_dim_count > 1){
         return false;
       }
    }
  }

  for(int i=0; i<src_array->dim; i++){
    if (src_array->info[i].align_manner == _XMP_N_ALIGN_BLOCK){
       src_block_dim=i;
       src_block_dim_count++;
       if (src_block_dim_count > 1){
         return false;
       }
    }
  }

  _XMP_gmove_calc_unit_size(dst_array, src_array, &alltoall_unit_size, 
                           &dst_pack_unit_size, &src_pack_unit_size, 
                           &dst_ser_size, &src_ser_size, 
                           dst_block_dim, src_block_dim);

  type_size = dst_array->type_size;
  bufsize=alltoall_unit_size*type_size*nnodes;

  dst_div = alltoall_unit_size/dst_pack_unit_size;
  src_div = alltoall_unit_size/src_pack_unit_size;

  if (dst_block_dim == src_block_dim){
    memcpy((char *)dst_array->array_addr_p, (char *)src_array->array_addr_p, bufsize);
    return true;
  }

  if (((_XMPF_running == 1) && (_XMPC_running == 0) && (dst_block_dim < src_dim-1))
      || ((_XMPF_running == 0) && (_XMPC_running == 1) && (dst_block_dim > 0))){
    sendbuf = _XMP_alloc(bufsize);
    // src_array -> sendbuf
    _XMP_pack_vector2((char *)sendbuf, (char *)src_array->array_addr_p, src_div, 
                      (int)src_pack_unit_size, nnodes, type_size, 1);
  }
  else {
    sendbuf = src_array->array_addr_p;
  }

  if (((_XMPF_running == 1) && (_XMPC_running == 0) && (src_block_dim < dst_dim - 1))
      || ((_XMPF_running == 0) && (_XMPC_running == 1) && (src_block_dim > 0))){
    recvbuf = _XMP_alloc(bufsize);
  }
  else {
    recvbuf = dst_array->array_addr_p;
  }

  MPI_Alltoall(sendbuf, alltoall_unit_size*type_size, MPI_BYTE, recvbuf, 
               alltoall_unit_size*type_size, MPI_BYTE,
               *((MPI_Comm *)dst_array->align_template->onto_nodes->comm));

  if (((_XMPF_running == 1) && (_XMPC_running == 0) && (src_block_dim < dst_dim-1))
      || ((_XMPF_running == 0) && (_XMPC_running == 1) && (src_block_dim > 0))){
    // dst_array <- recvbuf
    _XMP_pack_vector2((char *)dst_array->array_addr_p, (char *)recvbuf, nnodes, 
                      (int)dst_pack_unit_size, dst_div, type_size, 1);

    _XMP_free(recvbuf);
  }

  if (((_XMPF_running == 1) && (_XMPC_running == 0) && (src_block_dim == src_dim-1))
      || ((_XMPF_running == 0) && (_XMPC_running == 1) && (src_block_dim == 0))){
    _XMP_free(sendbuf);
  }

  return true;

}

void _XMP_align_local_idx(long long int global_idx, int *local_idx,
              _XMP_array_t *array, int array_axis, int *rank)
{
  _XMP_template_t *template = array->align_template;
  int template_index = array->info[array_axis].align_template_index;
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_nodes_info_t *n_info = chunk->onto_nodes_info;
  long long base = array->info[array_axis].ser_lower;
  long long tbase = template->info[template_index].ser_lower;
  int offset = array->info[array_axis].align_subscript + (base - tbase);
  int irank, idiv, imod;

  switch(array->info[array_axis].align_manner){
  case _XMP_N_ALIGN_DUPLICATION:
    {
      *rank=0;
      *local_idx = global_idx + offset - base;
    }
    break;
  case _XMP_N_ALIGN_BLOCK:
    {
      *rank = (global_idx + offset - base) / chunk->par_chunk_width;
      *local_idx = (global_idx + offset - base ) - *rank * chunk->par_chunk_width + array->info[array_axis].shadow_size_lo;
      idiv = offset / (chunk->par_chunk_width);
      if (*rank == idiv){
        *local_idx = *local_idx - offset%(chunk->par_chunk_width);
      }
    }
    break;
  case _XMP_N_ALIGN_CYCLIC:
    {
      idiv = offset/n_info->size;
      imod = offset%n_info->size;
      *rank = (global_idx + offset - base) % n_info->size;
      *local_idx = (global_idx + offset - base) / n_info->size;
      if (imod > *rank){
        *local_idx = *local_idx - (idiv + 1);
      } else {
        *local_idx = *local_idx - idiv;
      }
    }
    break;
  case _XMP_N_ALIGN_BLOCK_CYCLIC:
    {
      int w = chunk->par_width;
      idiv = (offset/w)/n_info->size;
      int imod1 = (offset/w)%n_info->size;
      int imod2 = offset%w;
      int off = global_idx + offset - base;
      *local_idx = (off / (n_info->size*w)) * w + off%w;

      *rank=(off/w)% (n_info->size);
      if (imod1 > 0){
        if (imod1 == *rank ){
          *local_idx = *local_idx - idiv*w-imod2;
        }else if (imod1 > *rank){
          *local_idx = *local_idx - (idiv+1)*w;
        }
      }else if (imod1 == 0){
        if (imod1 == *rank ){
          *local_idx = *local_idx - idiv*w -imod2;
        }else{
          *local_idx = *local_idx - idiv*w;
        }
      }
    }
    break;
  case _XMP_N_ALIGN_GBLOCK:
    {
      for(int i=1;i<(n_info->size+1);i++){
        if(global_idx + offset < chunk->mapping_array[i]+ (base - tbase)){
          *rank = i - 1;
          break;
        }
      }
      *local_idx = global_idx + offset - chunk->mapping_array[*rank]- (base - tbase) + array->info[array_axis].shadow_size_lo;
      for(int i=1;i<n_info->size+1;i++){
        if(offset < chunk->mapping_array[i]+(base-tbase)){
          irank = i - 1;
          idiv = offset - (chunk->mapping_array[i-1] + (base - tbase) - base);
          break;
        }
      }
      if (*rank == irank){
        *local_idx = *local_idx - idiv;
      }
    }
    break;
  case _XMP_N_ALIGN_NOT_ALIGNED:
    {
      *rank=0;
      *local_idx=global_idx - base;
    }
    break;
  default:
    _XMP_fatal("_XMP_: unknown chunk dist_manner");
  }

}

void _XMP_gmove_align_local_idx(long long int global_idx, int *local_idx,
              _XMP_gmv_desc_t *gmv_desc, int array_axis, int *rank)
{
  if (gmv_desc->is_global == true){
    _XMP_array_t *array = gmv_desc->a_desc;
    _XMP_align_local_idx(global_idx, local_idx, array, array_axis, rank);

  }else{
    *local_idx=global_idx - gmv_desc->a_lb[array_axis];
    *rank=0;
  }

}

int _XMP_gmove_calc_seq_rank(int *count, int *rank_acc, int **irank, int dim, int *ref){

  int rank_ref = 0;

  for(int i=0;i<dim;i++) rank_ref += rank_acc[i]*irank[i][count[i]];

  int exec_rank_ref = ref[rank_ref];

  return exec_rank_ref;

}

int _XMP_gmove_calc_seq_loc(int *count, unsigned long long *array_acc, int **local_idx, int dim){

  int loc_ref = 0;

  for(int i=0;i<dim;i++) loc_ref += array_acc[i]*local_idx[i][count[i]];

  return loc_ref;

}

void _XMP_gmove_calc_elmts_myrank(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, 
                                  int src_sub_dim, int src_dim, int dst_dim, 
                                  int *src_sub_num_ref, int *dst_sub_num_ref, int *src_count, int *dst_count, 
                                  int *num_triplet, 
                                  int **src_irank, int **dst_irank, int *src_rank_acc, int *dst_rank_acc, 
                                  int *src_num_myrank, int *dst_num_myrank, int *s2e, int *d2e, 
                                  int *send_size_ref, int *recv_size_ref, int *recv_size_ref2, 
                                  int *create_subcomm_flag, int *dst_color_ref){

  int i0,i1,i2,i3,i4,i5,i6;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int myrank = exec_nodes->comm_rank;
  int comm_size = exec_nodes->comm_size;
  int src_seq_rank, dst_seq_rank;
  int *unpack_loc;
  (*src_num_myrank)=0;
  (*dst_num_myrank)=0;

  switch(src_sub_dim){
  case 1:
    for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
      dst_count[dst_sub_num_ref[0]]=i0;
      src_count[src_sub_num_ref[0]]=i0;
      dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
      src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
      if(src_seq_rank==myrank){
        (*src_num_myrank)++;
        send_size_ref[dst_seq_rank]++;
      }
      if(dst_seq_rank==myrank){
        (*dst_num_myrank)++;
        recv_size_ref[src_seq_rank]++;
      }
      if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
          (*create_subcomm_flag == 1)) {
        if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
           recv_size_ref2[src_seq_rank]++;
        }
      }
      if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
        recv_size_ref2[src_seq_rank]++;
      }
    }
    break;
  case 2:
    for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
      for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
        dst_count[dst_sub_num_ref[1]]=i1;
        src_count[src_sub_num_ref[1]]=i1;
        dst_count[dst_sub_num_ref[0]]=i0;
        src_count[src_sub_num_ref[0]]=i0;
        dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
        src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
        if(src_seq_rank==myrank){
          (*src_num_myrank)++;
          send_size_ref[dst_seq_rank]++;
        }
        if(dst_seq_rank==myrank){
          (*dst_num_myrank)++;
          recv_size_ref[src_seq_rank]++;
        }
        if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
           (*create_subcomm_flag == 1)) {
          if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
            recv_size_ref2[src_seq_rank]++;
          }
        }
        if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
          recv_size_ref2[src_seq_rank]++;
        }
      }
    }
    break;
  case 3:
    for(i2=0;i2<num_triplet[src_sub_num_ref[2]];i2++){
      for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
        for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
          dst_count[dst_sub_num_ref[2]]=i2;
          src_count[src_sub_num_ref[2]]=i2;
          dst_count[dst_sub_num_ref[1]]=i1;
          src_count[src_sub_num_ref[1]]=i1;
          dst_count[dst_sub_num_ref[0]]=i0;
          src_count[src_sub_num_ref[0]]=i0;
          dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
          src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
          if(src_seq_rank==myrank){
            (*src_num_myrank)++;
            send_size_ref[dst_seq_rank]++;
          }
          if(dst_seq_rank==myrank){
            (*dst_num_myrank)++;
            recv_size_ref[src_seq_rank]++;
          }
          if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
              (*create_subcomm_flag == 1)) {
            if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
              recv_size_ref2[src_seq_rank]++;
            }
          }
          if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
            recv_size_ref2[src_seq_rank]++;
          }
        }
      }
    }
    break;
  case 4:
    for(i3=0;i3<num_triplet[src_sub_num_ref[3]];i3++){
      for(i2=0;i2<num_triplet[src_sub_num_ref[2]];i2++){
        for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
          for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
            dst_count[dst_sub_num_ref[3]]=i3;
            src_count[src_sub_num_ref[3]]=i3;
            dst_count[dst_sub_num_ref[2]]=i2;
            src_count[src_sub_num_ref[2]]=i2;
            dst_count[dst_sub_num_ref[1]]=i1;
            src_count[src_sub_num_ref[1]]=i1;
            dst_count[dst_sub_num_ref[0]]=i0;
            src_count[src_sub_num_ref[0]]=i0;
            dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
            src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
            if(src_seq_rank==myrank){
              (*src_num_myrank)++;
              send_size_ref[dst_seq_rank]++;
            }
            if(dst_seq_rank==myrank){
              (*dst_num_myrank)++;
              recv_size_ref[src_seq_rank]++;
            }
            if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
                (*create_subcomm_flag == 1)) {
              if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
                recv_size_ref2[src_seq_rank]++;
              }
            }
            if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
              recv_size_ref2[src_seq_rank]++;
            }
          }
        }
      }
    }
    break;
  case 5:
    for(i4=0;i4<num_triplet[src_sub_num_ref[4]];i4++){
      for(i3=0;i3<num_triplet[src_sub_num_ref[3]];i3++){
        for(i2=0;i2<num_triplet[src_sub_num_ref[2]];i2++){
          for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
            for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
              dst_count[dst_sub_num_ref[4]]=i4;
              src_count[src_sub_num_ref[4]]=i4;
              dst_count[dst_sub_num_ref[3]]=i3;
              src_count[src_sub_num_ref[3]]=i3;
              dst_count[dst_sub_num_ref[2]]=i2;
              src_count[src_sub_num_ref[2]]=i2;
              dst_count[dst_sub_num_ref[1]]=i1;
              src_count[src_sub_num_ref[1]]=i1;
              dst_count[dst_sub_num_ref[0]]=i0;
              src_count[src_sub_num_ref[0]]=i0;
              dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
              src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
              if(src_seq_rank==myrank){
                (*src_num_myrank)++;
                send_size_ref[dst_seq_rank]++;
              }
              if(dst_seq_rank==myrank){
                (*dst_num_myrank)++;
                recv_size_ref[src_seq_rank]++;
              }
              if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
                  (*create_subcomm_flag == 1)) {
                if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
                  recv_size_ref2[src_seq_rank]++;
                }
              }
              if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
                recv_size_ref2[src_seq_rank]++;
              }
            }
          }
        }
      }
    }
    break;
  case 6:
    for(i5=0;i5<num_triplet[src_sub_num_ref[5]];i5++){
      for(i4=0;i4<num_triplet[src_sub_num_ref[4]];i4++){
        for(i3=0;i3<num_triplet[src_sub_num_ref[3]];i3++){
          for(i2=0;i2<num_triplet[src_sub_num_ref[2]];i2++){
            for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
              for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
                dst_count[dst_sub_num_ref[5]]=i5;
                src_count[src_sub_num_ref[5]]=i5;
                dst_count[dst_sub_num_ref[4]]=i4;
                src_count[src_sub_num_ref[4]]=i4;
                dst_count[dst_sub_num_ref[3]]=i3;
                src_count[src_sub_num_ref[3]]=i3;
                dst_count[dst_sub_num_ref[2]]=i2;
                src_count[src_sub_num_ref[2]]=i2;
                dst_count[dst_sub_num_ref[1]]=i1;
                src_count[src_sub_num_ref[1]]=i1;
                dst_count[dst_sub_num_ref[0]]=i0;
                src_count[src_sub_num_ref[0]]=i0;
                dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                if(src_seq_rank==myrank){
                  (*src_num_myrank)++;
                  send_size_ref[dst_seq_rank]++;
                }
                if(dst_seq_rank==myrank){
                  (*dst_num_myrank)++;
                  recv_size_ref[src_seq_rank]++;
                }
                if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
                    (*create_subcomm_flag == 1)) {
                  if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
                    recv_size_ref2[src_seq_rank]++;
                  }
                }
                if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
                  recv_size_ref2[src_seq_rank]++;
                }
              }
            }
          }
        }
      }
    }
    break;
  case 7:
    for(i6=0;i6<num_triplet[src_sub_num_ref[6]];i6++){
      for(i5=0;i5<num_triplet[src_sub_num_ref[5]];i5++){
        for(i4=0;i4<num_triplet[src_sub_num_ref[4]];i4++){
          for(i3=0;i3<num_triplet[src_sub_num_ref[3]];i3++){
            for(i2=0;i2<num_triplet[src_sub_num_ref[2]];i2++){
              for(i1=0;i1<num_triplet[src_sub_num_ref[1]];i1++){
                for(i0=0;i0<num_triplet[src_sub_num_ref[0]];i0++){
                  dst_count[dst_sub_num_ref[6]]=i6;
                  src_count[src_sub_num_ref[6]]=i6;
                  dst_count[dst_sub_num_ref[5]]=i5;
                  src_count[src_sub_num_ref[5]]=i5;
                  dst_count[dst_sub_num_ref[4]]=i4;
                  src_count[src_sub_num_ref[4]]=i4;
                  dst_count[dst_sub_num_ref[3]]=i3;
                  src_count[src_sub_num_ref[3]]=i3;
                  dst_count[dst_sub_num_ref[2]]=i2;
                  src_count[src_sub_num_ref[2]]=i2;
                  dst_count[dst_sub_num_ref[1]]=i1;
                  src_count[src_sub_num_ref[1]]=i1;
                  dst_count[dst_sub_num_ref[0]]=i0;
                  src_count[src_sub_num_ref[0]]=i0;
                  dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                  src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                  if(src_seq_rank==myrank){
                    (*src_num_myrank)++;
                    send_size_ref[dst_seq_rank]++;
                  }
                  if(dst_seq_rank==myrank){
                    (*dst_num_myrank)++;
                    recv_size_ref[src_seq_rank]++;
                  }
                  if ((gmv_desc_leftp->is_global== true) && (gmv_desc_rightp->is_global==true) && 
                      (*create_subcomm_flag == 1)) {
                    if(dst_color_ref[dst_seq_rank]==dst_color_ref[myrank]){
                      recv_size_ref2[src_seq_rank]++;
                    }
                  }
                  if((gmv_desc_leftp->is_global==false) && (gmv_desc_rightp->is_global==true)){
                    recv_size_ref2[src_seq_rank]++;
                  }
                }
              }
            }
          }
        }
      }
    }
    break;
  default:
    _XMP_fatal("bad dimension of distributed array.");
  }

}

void _XMP_gmove_calc_color(int *color, int *div, int *mod, int *mult, int *num_term, int *target_rank){

    *color=0;
    for(int j=0;j<(*num_term);j++){
      *color=*color+(((*target_rank)/div[j])%mod[j])*mult[j];
    }
}

void _XMP_gmove_create_subcomm(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, MPI_Comm *newcomm, int *create_subcomm_flag, int *d2e, int *dst_color_ref){

  _XMP_array_t *dst_array;
  _XMP_template_t *dst_template;
  _XMP_nodes_t *dst_n;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int myrank = exec_nodes->comm_rank;
  int i, j, temp_index, nodes_index;
  int dst_dim, dst_n_dim, dst_myrank, dst_align_flag[_XMP_N_MAX_DIM];
  int dst_align_count=0;

  if (gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == true){
    dst_dim = gmv_desc_leftp->ndims;
    dst_array = gmv_desc_leftp->a_desc;
    dst_template = dst_array->align_template;
    dst_n = dst_template->onto_nodes;
    dst_n_dim = dst_n->dim;
    dst_myrank = dst_n->comm_rank;

    for(i=0;i<dst_n_dim;i++){
      dst_align_flag[i]=0;
    }

    for(i=0;i<dst_dim;i++){
      if ((dst_array->info[i].align_manner == _XMP_N_ALIGN_BLOCK)
         || (dst_array->info[i].align_manner == _XMP_N_ALIGN_CYCLIC)
         || (dst_array->info[i].align_manner == _XMP_N_ALIGN_BLOCK_CYCLIC)
         || (dst_array->info[i].align_manner == _XMP_N_ALIGN_GBLOCK)) {
        temp_index = dst_array->info[i].align_template_index;
        nodes_index = dst_template->chunk[temp_index].onto_nodes_index;
        dst_align_flag[nodes_index]=1;
        dst_align_count++;
      }
    }

    if (dst_align_count < dst_n_dim){
      *create_subcomm_flag=1;
    }else{
      *create_subcomm_flag=0;
    }

  }

  if (gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == true && *create_subcomm_flag == 1) {
    int dst_psize[_XMP_N_MAX_DIM];
    int noalign_flag;
    int div[4], mod[4], mult[4];

    if (dst_n_dim >= dst_dim ){

      dst_n_dim = dst_n->dim;

      for(i=0;i<4;i++){
        div[i]=1;
        mod[i]=1;
        mult[i]=1;
      }

      for(i=0;i<dst_n_dim;i++){
        dst_psize[i]=0;
      }

      for(i=0;i<dst_dim;i++){
        temp_index = dst_array->info[i].align_template_index;
        nodes_index = dst_template->chunk[temp_index].onto_nodes_index;
        if (temp_index==-1){
          dst_psize[i]=dst_n->info[i].size;
        }else{
          dst_psize[nodes_index]=dst_n->info[nodes_index].size;
        }
      }

      for(i=0;i<dst_n_dim;i++){
        if (dst_psize[i]==0){
          dst_psize[i]=dst_n->info[i].size;
        }
      }

      /* Calculate the number of terms of a color polynomial*/
      int dst_term_num_loc[4];
      noalign_flag=1;
      int num_term=0;
      for(i=0;i<dst_n_dim;i++){
        if (dst_align_flag[i] == 1){
          if (noalign_flag==1){
            num_term++;
            noalign_flag=0;
            dst_term_num_loc[num_term-1]=i-1;
          }
        }else{
          noalign_flag=1;
        }
      }

      /* Calculate the coefficient of a color polynomial and create new communicator.*/
      for(j=0;j<num_term;j++){
        for(i=0;i<dst_term_num_loc[j]+1;i++){
          div[j]=div[j]*dst_psize[i];
          if(dst_align_flag[i]==1){
            mult[j]=mult[j]*dst_psize[i];
          }
        }
        for(i=dst_term_num_loc[j]+1;i<dst_n_dim;i++){
          if(dst_align_flag[i]==1){
            mod[j]=mod[j]*dst_psize[i];
          }else{
            break;
          }
        }
      }

      int color;
      _XMP_gmove_calc_color(&color, div, mod, mult, &num_term, &dst_myrank);

      MPI_Comm newcomm0;
      //MPI_Comm_split( *(MPI_Comm *)dst_n->comm, color, myrank, &newcomm0);
      MPI_Comm_split( *(MPI_Comm *)dst_n->comm, color, dst_myrank, &newcomm0);
      *newcomm=newcomm0;

      int dst_color_tmp;
      for(int i=0;i<dst_n->comm_size;i++){
        //_XMP_gmove_calc_color(&dst_color_ref[i], div, mod, mult, &num_term, &i);
        _XMP_gmove_calc_color(&dst_color_tmp, div, mod, mult, &num_term, &i);
        dst_color_ref[d2e[i]]=dst_color_tmp;
      }

    }
  }
}

void _XMP_gmove_1to1(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long  *src_d){

  _XMP_array_t *dst_array;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;
  _XMP_template_t *dst_template;
  _XMP_template_t *src_template = src_array->align_template;
  _XMP_nodes_t *dst_n;
  _XMP_nodes_t *src_n = src_template->onto_nodes;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  MPI_Comm *exec_comm = exec_nodes->comm;
  int myrank = exec_nodes->comm_rank;
  int exec_comm_size = exec_nodes->comm_size;
  int dst_comm_size, dst_comm_rank, src_comm_size = src_n->comm_size;

  void *dst_addr, *src_addr;
  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;
  int dst_sub_dim=0, src_sub_dim=0;
  int *drank, *srank,*d2e, *s2e,*dst_color_ref;

  int type;
  size_t type_size;
  MPI_Group dst_grp, src_grp, exec_grp;

  MPI_Comm_group(*exec_comm, &exec_grp);

  if (gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == true){
    dst_array = gmv_desc_leftp->a_desc;
    dst_template = dst_array->align_template;
    dst_n = dst_template->onto_nodes;
    dst_comm_size = dst_n->comm_size;
    dst_comm_rank = dst_n->comm_rank;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = src_array->array_addr_p;
    MPI_Comm dst_comm = *(MPI_Comm *)dst_n->comm;
    MPI_Comm src_comm = *(MPI_Comm *)src_n->comm;
    MPI_Comm_group(dst_comm, &dst_grp);
    MPI_Comm_group(src_comm, &src_grp);
    drank = (int *)malloc(dst_comm_size*sizeof(int));
    srank = (int *)malloc(src_comm_size*sizeof(int));
    d2e = (int *)malloc(dst_comm_size*sizeof(int));
    s2e = (int *)malloc(src_comm_size*sizeof(int));
    dst_color_ref = (int *)malloc(exec_comm_size*sizeof(int));
    for(int i=0;i<dst_comm_size;i++) drank[i]=i;
    for(int i=0;i<src_comm_size;i++) srank[i]=i;
    for(int i=0;i<dst_comm_size;i++) d2e[i]=MPI_PROC_NULL;
    for(int i=0;i<src_comm_size;i++) s2e[i]=MPI_PROC_NULL;
    for(int i=0;i<dst_comm_size;i++) dst_color_ref[i]=MPI_PROC_NULL;
    if(dst_array->is_allocated){
      MPI_Group_translate_ranks(dst_grp, dst_comm_size, drank, exec_grp, d2e);
    }
    MPI_Allreduce(MPI_IN_PLACE, d2e, dst_comm_size, MPI_INT, MPI_MAX, *exec_comm);
    if(src_array->is_allocated){
      MPI_Group_translate_ranks(src_grp, src_comm_size, srank, exec_grp, s2e);
    }
    MPI_Allreduce(MPI_IN_PLACE, s2e, src_comm_size, MPI_INT, MPI_MAX, *exec_comm);
    free(drank);
    free(srank);
  }else if(gmv_desc_leftp->is_global == false && gmv_desc_rightp->is_global == true){
    dst_addr = gmv_desc_leftp->local_data;
    src_addr = src_array->array_addr_p;
    type = src_array->type;
    type_size = src_array->type_size;
    MPI_Comm src_comm = *(MPI_Comm *)src_n->comm;
    MPI_Comm_group(src_comm, &src_grp);
    srank = (int *)malloc(src_comm_size*sizeof(int));
    d2e = (int *)malloc(exec_comm_size*sizeof(int));
    s2e = (int *)malloc(src_comm_size*sizeof(int));
    for(int i=0;i<src_comm_size;i++) srank[i]=i;
    for(int i=0;i<exec_comm_size;i++) d2e[i]=i;
    for(int i=0;i<src_comm_size;i++) s2e[i]=MPI_PROC_NULL;
    if(src_array->is_allocated){
      MPI_Group_translate_ranks(src_grp, src_comm_size, srank, exec_grp, s2e);
    }
    MPI_Allreduce(MPI_IN_PLACE, s2e, exec_comm_size, MPI_INT, MPI_MAX, *exec_comm);
    free(srank);
  }else if(gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == false){
    dst_array = gmv_desc_leftp->a_desc;
    dst_template = dst_array->align_template;
    dst_n = dst_template->onto_nodes;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = gmv_desc_rightp->local_data;
  }

  int **src_local_idx, **dst_local_idx;
  int **src_irank,**dst_irank;
  int src_rank_acc[_XMP_N_MAX_DIM], dst_rank_acc[_XMP_N_MAX_DIM];
  int src_num[_XMP_N_MAX_DIM], dst_num[_XMP_N_MAX_DIM];
  int dst_num_myrank_total, src_num_myrank_total;
  int root_rank;
  unsigned long long dst_array_acc[_XMP_N_MAX_DIM];
  unsigned long long src_array_acc[_XMP_N_MAX_DIM];
  int i,j,jj,i0,i1,i2,i3,i4,i5,i6;
  int temp_index, nodes_index, create_subcomm_flag=0;
  int src_template_index[_XMP_N_MAX_DIM], dst_template_index[_XMP_N_MAX_DIM];
  int src_nodes_index[_XMP_N_MAX_DIM], dst_nodes_index[_XMP_N_MAX_DIM];

  int dst_sub_num_ref[_XMP_N_MAX_DIM], src_sub_num_ref[_XMP_N_MAX_DIM];
  int dst_count[_XMP_N_MAX_DIM], src_count[_XMP_N_MAX_DIM];

  src_local_idx=(int **)malloc(_XMP_N_MAX_DIM*sizeof(int *));
  src_irank=(int **)malloc(_XMP_N_MAX_DIM*sizeof(int *));
  dst_local_idx=(int **)malloc(_XMP_N_MAX_DIM*sizeof(int *));
  dst_irank=(int **)malloc(_XMP_N_MAX_DIM*sizeof(int *));

  for (i=0;i<_XMP_N_MAX_DIM;i++){
    src_num[i]=1;
    dst_num[i]=1;
    src_rank_acc[i]=0;
    dst_rank_acc[i]=0;
    src_sub_num_ref[i]=0;
    dst_sub_num_ref[i]=0;
    src_count[i]=0;
    dst_count[i]=0;
  }

  for (i=0;i<src_dim;i++){
    src_num[i]=_XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    if (src_num[i] > 1) {
      src_sub_num_ref[src_sub_dim]=i;
      src_sub_dim++;
    }
  }
  if (src_sub_dim == 0) src_sub_dim=src_dim;

  for (i=0;i<_XMP_N_MAX_DIM;i++){
    src_local_idx[i]=(int *)malloc(src_num[i]*sizeof(int));
    src_irank[i]=(int *)malloc(src_num[i]*sizeof(int));
    if (i>=src_dim){
      src_local_idx[i][0]=0;
      src_irank[i][0]=0;
    }
  }

  for (i=0;i<src_dim;i++){
    temp_index = src_array->info[i].align_template_index;
    nodes_index = src_template->chunk[temp_index].onto_nodes_index;
    src_array_acc[i] = src_array->info[i].dim_acc;
    if (temp_index == -1){
      src_rank_acc[i]=0;
    }else{
      src_template_index[i] = temp_index;
      src_nodes_index[i] = nodes_index;
      src_rank_acc[i]=src_n->info[nodes_index].multiplier;
    }

    jj=0;
    for (j = src_l[i]; j<=src_u[i]; j+=src_s[i]){

      _XMP_gmove_align_local_idx(j, &(src_local_idx[i][jj]),
                    gmv_desc_rightp, i, &(src_irank[i][jj]));
      jj++;

    }
  }

  unsigned long long dst_total_elmts=1;

  int dst_num_myindex, dst_num_myindex_total=1, dst_myindex[_XMP_N_MAX_DIM];

  for (i=0;i<dst_dim;i++){
    dst_num[i]=_XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_num[i] > 1) {
      dst_sub_num_ref[dst_sub_dim]=i;
      dst_sub_dim++;
    }
  }

  if (dst_sub_dim == 0) dst_sub_dim=dst_dim;

  for (i=0;i<_XMP_N_MAX_DIM;i++){
    dst_local_idx[i]=(int *)malloc(dst_num[i]*sizeof(int));
    dst_irank[i]=(int *)malloc(dst_num[i]*sizeof(int));
    if (i>=dst_dim){
      dst_local_idx[i][0]=0;
      dst_irank[i][0]=0;
    }
  }

  if (gmv_desc_leftp->is_global == true){
    for (i=0;i<dst_dim;i++){
      dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      temp_index = dst_array->info[i].align_template_index;
      nodes_index = dst_template->chunk[temp_index].onto_nodes_index;
      dst_array_acc[i] = dst_array->info[i].dim_acc;
      if (temp_index == -1){
        dst_myindex[i]=0;
        dst_rank_acc[i]=0;
      }else{
        dst_template_index[i] = temp_index;
        dst_nodes_index[i] = nodes_index;
        dst_myindex[i]=dst_n->info[dst_nodes_index[i]].rank;
        dst_rank_acc[i]=dst_n->info[dst_nodes_index[i]].multiplier;
      }

      jj=0;
      dst_num_myindex=0;
      for (j = dst_l[i]; j<=dst_u[i]; j+=dst_s[i]){

        _XMP_gmove_align_local_idx(j, &dst_local_idx[i][jj],
                      gmv_desc_leftp, i, &dst_irank[i][jj]);
        if (dst_irank[i][jj] == dst_myindex[i]) dst_num_myindex++;
        jj++;
      }
      dst_num_myindex_total *=dst_num_myindex;
    }
  }else{
    for (i=0;i<dst_dim;i++){
      dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
      dst_myindex[i]=0;
      dst_rank_acc[i]=0;
      dst_array_acc[i]=dst_d[i];

      jj=0;
      dst_num_myindex=0;
      for (j = dst_l[i]; j<=dst_u[i]; j+=dst_s[i]){
        _XMP_gmove_align_local_idx(j, &dst_local_idx[i][jj],
                      gmv_desc_leftp, i, &dst_irank[i][jj]);
        if (dst_irank[i][jj] == dst_myindex[i]) dst_num_myindex++;
        jj++;
      }
      dst_num_myindex_total *=dst_num_myindex;
    }
  }

  int *send_size_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_size_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_size_ref2=(int *)malloc(exec_comm_size*sizeof(int));
  int *send_addr_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_addr_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_addr_ref2=(int *)malloc(exec_comm_size*sizeof(int));
  int *send_count_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_count_ref=(int *)malloc(exec_comm_size*sizeof(int));
  int *recv_count_ref2=(int *)malloc(exec_comm_size*sizeof(int));
  
  size_t dst_type_size=type_size;
  size_t src_type_size=src_array->type_size;

  for(int i=0;i<exec_comm_size;i++) send_size_ref[i] = 0;
  for(int i=0;i<exec_comm_size;i++) recv_size_ref[i] = 0;
  for(int i=0;i<exec_comm_size;i++) recv_size_ref2[i] = 0;
  for(int i=0;i<exec_comm_size;i++) send_addr_ref[i] = 0;
  for(int i=0;i<exec_comm_size;i++) recv_addr_ref[i] = 0;
  for(int i=0;i<exec_comm_size;i++) recv_addr_ref2[i] = 0;

  int newcomm_rank, num_term=0;
  MPI_Comm newcomm;
  _XMP_gmove_create_subcomm(gmv_desc_leftp, gmv_desc_rightp, &newcomm, &create_subcomm_flag, d2e, dst_color_ref);

  _XMP_gmove_calc_elmts_myrank(gmv_desc_leftp, gmv_desc_rightp, src_sub_dim, src_dim, dst_dim, src_sub_num_ref, dst_sub_num_ref, src_count, dst_count, src_num, src_irank, dst_irank, src_rank_acc, dst_rank_acc, &src_num_myrank_total, &dst_num_myrank_total, s2e, d2e, send_size_ref, recv_size_ref, recv_size_ref2, &create_subcomm_flag, dst_color_ref);

  int isend_count=0, irecv_count=0, irecv_count2=0;
  int count_acc=0, count_acc_tmp=0;
  for(int i=0;i<exec_comm_size;i++) {
    if(send_size_ref[i] > 0){
      count_acc_tmp += send_size_ref[i];
      send_addr_ref[i]=count_acc;
      count_acc=count_acc_tmp;
      isend_count++;
    }
  }

  count_acc=0, count_acc_tmp=0;
  for(int i=0;i<exec_comm_size;i++) {
    if(recv_size_ref[i] > 0){
      count_acc_tmp += recv_size_ref[i];
      recv_addr_ref[i]=count_acc;
      count_acc=count_acc_tmp;
      irecv_count++;
    }
  }

  count_acc=0, count_acc_tmp=0;
  for(int i=0;i<exec_comm_size;i++) {
    if(recv_size_ref2[i] > 0){
      count_acc_tmp += recv_size_ref2[i];
      recv_addr_ref2[i]=count_acc;
      count_acc=count_acc_tmp;
      irecv_count2++;
    }
  }

  for(int i=0;i<exec_comm_size;i++) send_count_ref[i] = send_addr_ref[i];
  for(int i=0;i<exec_comm_size;i++) recv_count_ref[i] = recv_addr_ref[i];
  for(int i=0;i<exec_comm_size;i++) recv_count_ref2[i] = recv_addr_ref2[i];

  char *send_buf, *recv_buf;
  MPI_Request *dst_request;
  MPI_Request *src_request;
  MPI_Status *dst_status;
  MPI_Status *src_status;
  dst_request=(MPI_Request *)malloc(irecv_count*sizeof(MPI_Request));
  dst_status=(MPI_Status *)malloc(irecv_count*sizeof(MPI_Status));
  src_request=(MPI_Request *)malloc(isend_count*sizeof(MPI_Request));
  src_status=(MPI_Status *)malloc(isend_count*sizeof(MPI_Status));
  send_buf=(char *)malloc(src_num_myrank_total*src_type_size*sizeof(char));
  recv_buf=(char *)malloc(dst_num_myindex_total*dst_type_size*sizeof(char));
  int dst_seq_rank, src_seq_rank, dst_seq_loc, src_seq_loc;


  switch(src_sub_dim){
  case 1:
    for(i=0;i<dst_num[dst_sub_num_ref[0]];i++){
      dst_count[dst_sub_num_ref[0]]=i;
      src_count[src_sub_num_ref[0]]=i;
      src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
      src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
      dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
      if (src_seq_rank == myrank){
           memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                  (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
           send_count_ref[dst_seq_rank]++;
      }
    }
    break;
  case 2:
    for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
      for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
        dst_count[dst_sub_num_ref[1]]=i1;
        src_count[src_sub_num_ref[1]]=i1;
        dst_count[dst_sub_num_ref[0]]=i0;
        src_count[src_sub_num_ref[0]]=i0;
        src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
        src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
        dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
        if (src_seq_rank == myrank){
          memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                 (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
          send_count_ref[dst_seq_rank]++;
        }
      }
    }
    break;
  case 3:
    for(i2=0;i2<dst_num[dst_sub_num_ref[2]];i2++){
      for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
        for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
          dst_count[dst_sub_num_ref[2]]=i2;
          src_count[src_sub_num_ref[2]]=i2;
          dst_count[dst_sub_num_ref[1]]=i1;
          src_count[src_sub_num_ref[1]]=i1;
          dst_count[dst_sub_num_ref[0]]=i0;
          src_count[src_sub_num_ref[0]]=i0;
          src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
          src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
          dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
          if (src_seq_rank == myrank){
            memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                   (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
            send_count_ref[dst_seq_rank]++;
          }
        }
      }
    }
    break;
  case 4:
    for(i3=0;i3<dst_num[dst_sub_num_ref[3]];i3++){
      for(i2=0;i2<dst_num[dst_sub_num_ref[2]];i2++){
        for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
          for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
            dst_count[dst_sub_num_ref[3]]=i3;
            src_count[src_sub_num_ref[3]]=i3;
            dst_count[dst_sub_num_ref[2]]=i2;
            src_count[src_sub_num_ref[2]]=i2;
            dst_count[dst_sub_num_ref[1]]=i1;
            src_count[src_sub_num_ref[1]]=i1;
            dst_count[dst_sub_num_ref[0]]=i0;
            src_count[src_sub_num_ref[0]]=i0;
            src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
            src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
            dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
            if (src_seq_rank == myrank){
              memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                     (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
              send_count_ref[dst_seq_rank]++;
            }
          }
        }
      }
    }
    break;
  case 5:
    for(i4=0;i4<dst_num[dst_sub_num_ref[4]];i4++){
      for(i3=0;i3<dst_num[dst_sub_num_ref[3]];i3++){
        for(i2=0;i2<dst_num[dst_sub_num_ref[2]];i2++){
          for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
            for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
              dst_count[dst_sub_num_ref[4]]=i4;
              src_count[src_sub_num_ref[4]]=i4;
              dst_count[dst_sub_num_ref[3]]=i3;
              src_count[src_sub_num_ref[3]]=i3;
              dst_count[dst_sub_num_ref[2]]=i2;
              src_count[src_sub_num_ref[2]]=i2;
              dst_count[dst_sub_num_ref[1]]=i1;
              src_count[src_sub_num_ref[1]]=i1;
              dst_count[dst_sub_num_ref[0]]=i0;
              src_count[src_sub_num_ref[0]]=i0;
              src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
              src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
              dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
              if (src_seq_rank == myrank){
                memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                       (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
                send_count_ref[dst_seq_rank]++;
              }
            }
          }
        }
      }
    }
    break;
  case 6:
    for(i5=0;i5<dst_num[dst_sub_num_ref[5]];i5++){
      for(i4=0;i4<dst_num[dst_sub_num_ref[4]];i4++){
        for(i3=0;i3<dst_num[dst_sub_num_ref[3]];i3++){
          for(i2=0;i2<dst_num[dst_sub_num_ref[2]];i2++){
            for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
              for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
                dst_count[dst_sub_num_ref[5]]=i5;
                src_count[src_sub_num_ref[5]]=i5;
                dst_count[dst_sub_num_ref[4]]=i4;
                src_count[src_sub_num_ref[4]]=i4;
                dst_count[dst_sub_num_ref[3]]=i3;
                src_count[src_sub_num_ref[3]]=i3;
                dst_count[dst_sub_num_ref[2]]=i2;
                src_count[src_sub_num_ref[2]]=i2;
                dst_count[dst_sub_num_ref[1]]=i1;
                src_count[src_sub_num_ref[1]]=i1;
                dst_count[dst_sub_num_ref[0]]=i0;
                src_count[src_sub_num_ref[0]]=i0;
                src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
                dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                if (src_seq_rank == myrank){
                  memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                         (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
                  send_count_ref[dst_seq_rank]++;
                }
              }
            }
          }
        }
      }
    }
    break;
  case 7:
    for(i6=0;i6<dst_num[dst_sub_num_ref[6]];i6++){
      for(i5=0;i5<dst_num[dst_sub_num_ref[5]];i5++){
        for(i4=0;i4<dst_num[dst_sub_num_ref[4]];i4++){
          for(i3=0;i3<dst_num[dst_sub_num_ref[3]];i3++){
            for(i2=0;i2<dst_num[dst_sub_num_ref[2]];i2++){
              for(i1=0;i1<dst_num[dst_sub_num_ref[1]];i1++){
                for(i0=0;i0<dst_num[dst_sub_num_ref[0]];i0++){
                  dst_count[dst_sub_num_ref[6]]=i6;
                  src_count[src_sub_num_ref[6]]=i6;
                  dst_count[dst_sub_num_ref[5]]=i5;
                  src_count[src_sub_num_ref[5]]=i5;
                  dst_count[dst_sub_num_ref[4]]=i4;
                  src_count[src_sub_num_ref[4]]=i4;
                  dst_count[dst_sub_num_ref[3]]=i3;
                  src_count[src_sub_num_ref[3]]=i3;
                  dst_count[dst_sub_num_ref[2]]=i2;
                  src_count[src_sub_num_ref[2]]=i2;
                  dst_count[dst_sub_num_ref[1]]=i1;
                  src_count[src_sub_num_ref[1]]=i1;
                  dst_count[dst_sub_num_ref[0]]=i0;
                  src_count[src_sub_num_ref[0]]=i0;
                  src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                  src_seq_loc=_XMP_gmove_calc_seq_loc(src_count, src_array_acc, src_local_idx, src_dim);
                  dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                  if (src_seq_rank == myrank){
                    memcpy((char *)send_buf+send_count_ref[dst_seq_rank]*src_type_size, 
                           (char *)src_addr+src_seq_loc*src_type_size, src_type_size);
                    send_count_ref[dst_seq_rank]++;
                  }
                }
              }
            }
          }
        }
      }
    }

    break;
  default:
    _XMP_fatal("_XMP_: src_sub_dim unknown a dimension number");
  }

  int icount=0;
  for(int i=0;i<exec_comm_size;i++){
    if(send_size_ref[i] > 0){
      MPI_Isend((char *)send_buf+send_addr_ref[i]*src_type_size, 
                send_size_ref[i]*src_type_size, MPI_BYTE, i,
                _XMP_N_MPI_TAG_GMOVE, *exec_comm, &src_request[icount]);
      icount++;
    }
  }
  icount=0;
  for(int i=0;i<exec_comm_size;i++){
    if(recv_size_ref[i] > 0){
      MPI_Irecv((char *)recv_buf+recv_addr_ref[i]*dst_type_size, 
                recv_size_ref[i]*dst_type_size, MPI_BYTE, i,
                _XMP_N_MPI_TAG_GMOVE, *exec_comm, &dst_request[icount]);
      icount++;
    }
  }
  MPI_Waitall(isend_count, src_request, src_status);
  MPI_Waitall(irecv_count, dst_request, dst_status);

  int iroot=0;
  /* bcast & unpack */
  if (gmv_desc_rightp->is_global == true){
    if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
      if(dst_array->is_allocated){
        MPI_Bcast(recv_buf, dst_num_myindex_total*dst_type_size, MPI_BYTE, iroot, newcomm);
      }
    }else if (gmv_desc_leftp->is_global == false){
      MPI_Bcast(recv_buf, dst_num_myindex_total*dst_type_size, MPI_BYTE, iroot, *exec_comm);
    }
  }

  int mycolor, dst_color;
  switch(dst_sub_dim){
  case 1:
    for(i=0;i<src_num[src_sub_num_ref[0]];i++){
      dst_count[dst_sub_num_ref[0]]=i;
      src_count[src_sub_num_ref[0]]=i;
      dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
      dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
      src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
      if (gmv_desc_rightp->is_global == true){
        if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
          if (dst_seq_rank == myrank){
            memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                   (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
            recv_count_ref[src_seq_rank]++;
          }
        }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
          if (dst_color_ref[dst_seq_rank] == dst_color_ref[d2e[dst_comm_rank]]){
            memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                   (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
            recv_count_ref2[src_seq_rank]++;
          }
        }else if(gmv_desc_leftp->is_global == false){
          memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                 (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
          recv_count_ref2[src_seq_rank]++;
        }
      }
    }
    break;
  case 2:
    for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
      for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
        dst_count[dst_sub_num_ref[1]]=i1;
        src_count[src_sub_num_ref[1]]=i1;
        dst_count[dst_sub_num_ref[0]]=i0;
        src_count[src_sub_num_ref[0]]=i0;
        dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
        dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
        src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
        if (gmv_desc_rightp->is_global == true){
          if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
            if (dst_seq_rank == myrank){
              memcpy((char *)dst_addr+dst_seq_loc*dst_type_size, 
                     (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
              recv_count_ref[src_seq_rank]++;
            }
          }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
            if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
              memcpy((char *)dst_addr+dst_seq_loc*dst_type_size, 
                     (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
              recv_count_ref2[src_seq_rank]++;
            }
          }else if(gmv_desc_leftp->is_global == false){
            memcpy((char *)dst_addr+dst_seq_loc*dst_type_size, 
                   (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
            recv_count_ref2[src_seq_rank]++;
          }
        }
      }
    }
    break;
  case 3:
    for(i2=0;i2<src_num[src_sub_num_ref[2]];i2++){
      for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
        for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
          dst_count[dst_sub_num_ref[2]]=i2;
          src_count[src_sub_num_ref[2]]=i2;
          dst_count[dst_sub_num_ref[1]]=i1;
          src_count[src_sub_num_ref[1]]=i1;
          dst_count[dst_sub_num_ref[0]]=i0;
          src_count[src_sub_num_ref[0]]=i0;
          dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
          dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
          src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
          if (gmv_desc_rightp->is_global == true){
            if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
              if (dst_seq_rank == myrank){
                memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                       (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
                recv_count_ref[src_seq_rank]++;
              }
            }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
              if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
                memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                       (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                recv_count_ref2[src_seq_rank]++;
              }
            }else if(gmv_desc_leftp->is_global == false){
              memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                     (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
              recv_count_ref2[src_seq_rank]++;
            }
          }
        }
      }
    }

    break;
  case 4:
    for(i3=0;i3<src_num[src_sub_num_ref[3]];i3++){
      for(i2=0;i2<src_num[src_sub_num_ref[2]];i2++){
        for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
          for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
            dst_count[dst_sub_num_ref[3]]=i3;
            src_count[src_sub_num_ref[3]]=i3;
            dst_count[dst_sub_num_ref[2]]=i2;
            src_count[src_sub_num_ref[2]]=i2;
            dst_count[dst_sub_num_ref[1]]=i1;
            src_count[src_sub_num_ref[1]]=i1;
            dst_count[dst_sub_num_ref[0]]=i0;
            src_count[src_sub_num_ref[0]]=i0;
            dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
            dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
            src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
            if (gmv_desc_rightp->is_global == true){
              if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
                if (dst_seq_rank == myrank){
                  memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                         (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
                  recv_count_ref[src_seq_rank]++;
                }
              }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
                if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
                  memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                         (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                  recv_count_ref2[src_seq_rank]++;
                }
              }else if(gmv_desc_leftp->is_global == false){
                memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                       (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                recv_count_ref2[src_seq_rank]++;
              }
            }
          }
        }
      }
    }

    break;
  case 5:
    icount=0;
    for(i4=0;i4<src_num[src_sub_num_ref[4]];i4++){
      for(i3=0;i3<src_num[src_sub_num_ref[3]];i3++){
        for(i2=0;i2<src_num[src_sub_num_ref[2]];i2++){
          for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
            for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
              dst_count[dst_sub_num_ref[4]]=i4;
              src_count[src_sub_num_ref[4]]=i4;
              dst_count[dst_sub_num_ref[3]]=i3;
              src_count[src_sub_num_ref[3]]=i3;
              dst_count[dst_sub_num_ref[2]]=i2;
              src_count[src_sub_num_ref[2]]=i2;
              dst_count[dst_sub_num_ref[1]]=i1;
              src_count[src_sub_num_ref[1]]=i1;
              dst_count[dst_sub_num_ref[0]]=i0;
              src_count[src_sub_num_ref[0]]=i0;
              dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
              dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
              src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
              if (gmv_desc_rightp->is_global == true){
                if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
                  if (dst_seq_rank == myrank){
                    memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                           (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
                    recv_count_ref[src_seq_rank]++;
                  }
                }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
                  if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
                    memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                           (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                    recv_count_ref2[src_seq_rank]++;
                  }
                }else if(gmv_desc_leftp->is_global == false){
                  memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                         (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                  recv_count_ref2[src_seq_rank]++;
                }
              }
            }
          }
        }
      }
    }

    break;
  case 6:
    icount=0;
    for(i5=0;i5<src_num[src_sub_num_ref[5]];i5++){
      for(i4=0;i4<src_num[src_sub_num_ref[4]];i4++){
        for(i3=0;i3<src_num[src_sub_num_ref[3]];i3++){
          for(i2=0;i2<src_num[src_sub_num_ref[2]];i2++){
            for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
              for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
                dst_count[dst_sub_num_ref[5]]=i5;
                src_count[src_sub_num_ref[5]]=i5;
                dst_count[dst_sub_num_ref[4]]=i4;
                src_count[src_sub_num_ref[4]]=i4;
                dst_count[dst_sub_num_ref[3]]=i3;
                src_count[src_sub_num_ref[3]]=i3;
                dst_count[dst_sub_num_ref[2]]=i2;
                src_count[src_sub_num_ref[2]]=i2;
                dst_count[dst_sub_num_ref[1]]=i1;
                src_count[src_sub_num_ref[1]]=i1;
                dst_count[dst_sub_num_ref[0]]=i0;
                src_count[src_sub_num_ref[0]]=i0;
                dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
                src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                if (gmv_desc_rightp->is_global == true){
                  if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
                    if (dst_seq_rank == myrank){
                      memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                             (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
                      recv_count_ref[src_seq_rank]++;
                    }
                  }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
                    if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
                      memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                             (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                      recv_count_ref2[src_seq_rank]++;
                    }
                  }else if(gmv_desc_leftp->is_global == false){
                    memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                           (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                    recv_count_ref2[src_seq_rank]++;
                  }
                }
              }
            }
          }
        }
      }
    }

    break;
  case 7:
    icount=0;
    for(i6=0;i6<src_num[src_sub_num_ref[6]];i6++){
      for(i5=0;i5<src_num[src_sub_num_ref[5]];i5++){
        for(i4=0;i4<src_num[src_sub_num_ref[4]];i4++){
          for(i3=0;i3<src_num[src_sub_num_ref[3]];i3++){
            for(i2=0;i2<src_num[src_sub_num_ref[2]];i2++){
              for(i1=0;i1<src_num[src_sub_num_ref[1]];i1++){
                for(i0=0;i0<src_num[src_sub_num_ref[0]];i0++){
                  dst_count[dst_sub_num_ref[6]]=i6;
                  src_count[src_sub_num_ref[6]]=i6;
                  dst_count[dst_sub_num_ref[5]]=i5;
                  src_count[src_sub_num_ref[5]]=i5;
                  dst_count[dst_sub_num_ref[4]]=i4;
                  src_count[src_sub_num_ref[4]]=i4;
                  dst_count[dst_sub_num_ref[3]]=i3;
                  src_count[src_sub_num_ref[3]]=i3;
                  dst_count[dst_sub_num_ref[2]]=i2;
                  src_count[src_sub_num_ref[2]]=i2;
                  dst_count[dst_sub_num_ref[1]]=i1;
                  src_count[src_sub_num_ref[1]]=i1;
                  dst_count[dst_sub_num_ref[0]]=i0;
                  src_count[src_sub_num_ref[0]]=i0;
                  dst_seq_rank=_XMP_gmove_calc_seq_rank(dst_count, dst_rank_acc, dst_irank, dst_dim, d2e);
                  dst_seq_loc=_XMP_gmove_calc_seq_loc(dst_count, dst_array_acc, dst_local_idx, dst_dim);
                  src_seq_rank=_XMP_gmove_calc_seq_rank(src_count, src_rank_acc, src_irank, src_dim, s2e);
                  if (gmv_desc_rightp->is_global == true){
                    if (gmv_desc_leftp->is_global == true && create_subcomm_flag == 0){
                      if (dst_seq_rank == myrank){
                        memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                               (char *)recv_buf+recv_count_ref[src_seq_rank]*dst_type_size, dst_type_size);
                        recv_count_ref[src_seq_rank]++;
                      }
                    }else if(gmv_desc_leftp->is_global == true && create_subcomm_flag == 1){
                      if (dst_color_ref[dst_seq_rank] == dst_color_ref[myrank]){
                        memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                               (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                        recv_count_ref2[src_seq_rank]++;
                      }
                    }else if(gmv_desc_leftp->is_global == false){
                      memcpy((char *)dst_addr+dst_seq_loc*dst_type_size,
                             (char *)recv_buf+recv_count_ref2[src_seq_rank]*dst_type_size, dst_type_size);
                      recv_count_ref2[src_seq_rank]++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    break;
  default:
    _XMP_fatal("_XMP_: dst_sub_dim unknown a dimension number");
  }

  if (gmv_desc_rightp->is_global == true){
    free(d2e);
    free(s2e);
  }
  if (gmv_desc_rightp->is_global == true && gmv_desc_leftp->is_global == true){
    free(dst_color_ref);
  }

  free(dst_request);
  free(src_request);
  free(dst_status);
  free(src_status);

  free(send_size_ref);
  free(recv_size_ref);
  free(recv_size_ref2);
  free(send_count_ref);
  free(recv_count_ref);
  free(recv_count_ref2);
  free(send_addr_ref);
  free(recv_addr_ref);
  free(recv_addr_ref2);
  free(send_buf);
  free(recv_buf);

  for (int i=0;i<_XMP_N_MAX_DIM;i++){
    free(src_local_idx[i]);
    free(src_irank[i]);
  }
  for (int i=0;i<_XMP_N_MAX_DIM;i++){
    free(dst_local_idx[i]);
    free(dst_irank[i]);
  }
  free(src_local_idx);
  free(src_irank);
  free(dst_local_idx);
  free(dst_irank);

}

void _XMP_gmove_array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long *src_d){

  _XMP_array_t *dst_array;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;
  _XMP_template_t *dst_template, *src_template;
  _XMP_nodes_t *dst_nodes, *src_nodes;

  void *dst_addr, *src_addr;
  int type;
  size_t type_size;
  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;
  int dst_num, src_num, dst_sub_dim=0, src_sub_dim=0;
  int dst_comm_size, src_comm_size;
  int exec_comm_size = _XMP_get_execution_nodes()->comm_size;

  if (gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == true){
    dst_array = gmv_desc_leftp->a_desc;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = src_array->array_addr_p;
    dst_template = dst_array->align_template;
    src_template = src_array->align_template;
    dst_nodes = dst_template->onto_nodes;
    src_nodes = src_template->onto_nodes;
    dst_comm_size = dst_nodes->comm_size;
    src_comm_size = src_nodes->comm_size;
  }else if(gmv_desc_leftp->is_global == false && gmv_desc_rightp->is_global == true){
    dst_addr = gmv_desc_leftp->local_data;
    src_addr = src_array->array_addr_p;
    type = src_array->type;
    type_size = src_array->type_size;
  }else if(gmv_desc_leftp->is_global == true && gmv_desc_rightp->is_global == false){
    dst_array = gmv_desc_leftp->a_desc;
    type = dst_array->type;
    type_size = dst_array->type_size;
    dst_addr = dst_array->array_addr_p;
    src_addr = gmv_desc_rightp->local_data;
  }

  for (int i=0;i<dst_dim;i++){
    dst_num=_XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    if (dst_num > 1) {
      dst_sub_dim++;
    }
  }

  for (int i=0;i<src_dim;i++){
    src_num=_XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    if (src_num > 1) {
      src_sub_dim++;
    }
  }

  if ((gmv_desc_leftp->is_global == true) && (gmv_desc_rightp->is_global == true)){
    if ((exec_comm_size == dst_comm_size) && (exec_comm_size == src_comm_size)){
      if ((_XMP_IS_SINGLE) ||
          (is_same_array_shape(dst_array, src_array) &&
           is_same_template_shape(dst_array->align_template, src_array->align_template) &&
           is_same_axis(dst_array, src_array) &&
           is_same_offset(dst_array, src_array) &&
           is_same_alignmanner(dst_array, src_array) &&
           is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp))) {


        for (int i = 0; i < dst_dim; i++) {
          dst_l[i]=dst_array->info[i].local_lower;
          dst_u[i]=dst_array->info[i].local_upper;
          dst_s[i]=dst_array->info[i].local_stride;
        }

        for (int i = 0; i < src_dim; i++) {
          src_l[i]=src_array->info[i].local_lower;
          src_u[i]=src_array->info[i].local_upper;
          src_s[i]=src_array->info[i].local_stride;
        }

        _XMP_gmove_localcopy_ARRAY(type, type_size,
                                   (void *)dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                                   (void *)src_addr, src_dim, src_l, src_u, src_s, src_d);
        return;
      }

      if (is_same_array_shape(dst_array, src_array) &&
          is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp) &&
          is_one_block(dst_array) && is_one_block(src_array) &&
          (dst_array->dim >= dst_array->align_template->dim) &&
          (src_array->dim >= src_array->align_template->dim)){
        if (_XMP_gmove_transpose(gmv_desc_leftp, gmv_desc_rightp)) return;
      }
    }
  }

// temporary check flag : chk_flag

  int chk_flag=0, dst_chk_flag, src_chk_flag;

  if (gmv_desc_leftp->is_global == true
     && gmv_desc_rightp->is_global == true){

     for(int i=0; i<dst_template->dim;i++){
       if (dst_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK){
          dst_chk_flag=0;
          break;
       }else{
          dst_chk_flag=1;
       }
     }

     for(int i=0; i<src_template->dim;i++){
       if (src_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK){
          src_chk_flag=0;
          break;
       }else{
          src_chk_flag=1;
       }
     }

     if (dst_dim==dst_nodes->dim){
        if(src_dim==src_nodes->dim){

          for(int i=0; i<dst_dim;i++){
            if ((dst_array->info[i].align_subscript != 0 )
               || (dst_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK)){
              dst_chk_flag=0;
              break;
            }
          }

          for(int i=0; i<src_dim;i++){
            if ((src_array->info[i].align_subscript != 0 )
               || (src_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK)){
                src_chk_flag=0;
                break;
             }
          }

        }

        if (is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp)){
        }else{
          dst_chk_flag=0;
          src_chk_flag=0;
        }

     }else if (dst_dim < dst_nodes->dim){
        if(src_dim < src_nodes->dim){
          if((dst_nodes->dim != 2) 
            || (src_nodes->dim != 2)
            || (dst_array->info[0].align_subscript != 0)
            || (src_array->info[0].align_subscript != 0)){
            dst_chk_flag=0;
            src_chk_flag=0;
          }
        }else{
          dst_chk_flag=0;
          src_chk_flag=0;
        }
     }else if (dst_dim > dst_nodes->dim){
        if (src_dim > src_nodes->dim){
           if (_XMPF_running == 1
               && _XMPC_running == 0){
              dst_chk_flag=0;
              src_chk_flag=0;
           }
        }else{
           dst_chk_flag=0;
           src_chk_flag=0;
        }
     }

     if (dst_chk_flag==1 && src_chk_flag==1) chk_flag=1;

     if ((exec_comm_size != dst_comm_size) || (exec_comm_size != src_comm_size)) chk_flag=0;

  }

  if (chk_flag == 1) {

    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);

    _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
    int dst_array_nodes_dim = dst_array_nodes->dim;
    int dst_array_nodes_ref[dst_array_nodes_dim];
    for (int i = 0; i < dst_array_nodes_dim; i++) {
      dst_array_nodes_ref[i] = 0;
    }


    for(int i=0;i<dst_dim;i++){
      if (dst_array->info[i].align_manner ==_XMP_N_ALIGN_BLOCK_CYCLIC){
        break;
      }
    }

    for(int i=0;i<src_dim;i++){
      if (src_array->info[i].align_manner ==_XMP_N_ALIGN_BLOCK_CYCLIC){
        break;
      }
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
            _XMP_sendrecv_ARRAY(type, type_size, &mpi_datatype,
                              dst_array, dst_array_nodes_ref,
                              recv_lower, recv_upper, recv_stride, dst_d,
                              src_array, src_array_nodes_ref,
                              send_lower, send_upper, send_stride, src_d);
          }
        } while (_XMP_get_next_rank(src_array_nodes, src_array_nodes_ref));
      }
    } while (_XMP_get_next_rank(dst_array_nodes, dst_array_nodes_ref));

    MPI_Type_free(&mpi_datatype);

  }else {

    _XMP_gmove_1to1(gmv_desc_leftp, gmv_desc_rightp, dst_l, dst_u, dst_s, dst_d, src_l, src_u, src_s, src_d);

  }
}

void _XMP_gmove_BCAST_ARRAY(_XMP_array_t *src_array, int type, size_t type_size, ...) {
  unsigned long long gmove_total_elmts = 0;

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  int dummy[7] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = va_arg(args, void *);
  int dst_dim = va_arg(args, int);
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  int dst_a_lb[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + va_arg(args, int)-1;
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    dst_a_lb[i]=0;
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = src_array->array_addr_p;
  int src_dim = src_array->dim;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = src_l[i] + va_arg(args, int)-1;
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    gmove_total_elmts = dst_total_elmts;
  }

  gmv_desc_leftp.is_global = false;       gmv_desc_rightp.is_global = true;
  gmv_desc_leftp.ndims = dst_dim;         gmv_desc_rightp.ndims = src_dim;

  gmv_desc_leftp.a_desc = NULL;          gmv_desc_rightp.a_desc = src_array;

  gmv_desc_leftp.local_data = dst_addr;      gmv_desc_rightp.local_data = NULL;
  gmv_desc_leftp.a_lb = dst_a_lb;            gmv_desc_rightp.a_lb = NULL;
  gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

  gmv_desc_leftp.kind = dummy;           gmv_desc_rightp.kind = dummy; // always triplet
  gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_l;
  gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_u;
  gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = src_s;

  _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp, dst_l, dst_u, dst_s, dst_d, src_l, src_u, src_s, src_d);

  int iflag =0;
  if (iflag==1){
    if (_XMP_IS_SINGLE) {
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
    } while (_XMP_get_next_rank(array_nodes, array_nodes_ref));
  }
}

void _XMP_gmove_HOMECOPY_ARRAY(_XMP_array_t *dst_array, int type, size_t type_size, ...) {

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  if (!dst_array->is_allocated) {
    return;
  }
  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = dst_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = va_arg(args, void *);
  int src_dim = va_arg(args, int);
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_u[i] = src_l[i] + (size - 1) * src_s[i];
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  if (_XMP_IS_SINGLE) {
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

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  unsigned long long gmove_total_elmts = 0;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = dst_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + va_arg(args, int)-1;
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = src_array->array_addr_p;
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    src_u[i] = src_l[i] + va_arg(args, int)-1;
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }
  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    gmove_total_elmts = dst_total_elmts;
  }

  // do transpose
  int dummy[7] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

  gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = true;
  gmv_desc_leftp.ndims = dst_array->dim; gmv_desc_rightp.ndims = src_array->dim;

  gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = src_array;

  gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = NULL;
  gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = NULL;
  gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

  gmv_desc_leftp.kind = dummy;           gmv_desc_rightp.kind = dummy; // always triplet
  gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_l;
  gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_u;
  gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = src_s;

  _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp, dst_l, dst_u, dst_s, dst_d, src_l, src_u, src_s, src_d);

}

// Test commicator cache mechanism for _XMP_gmove_BCAST_TO_NOTALIGNED_ARRAY
#define GMOVE_COMM_CACHE_SIZE 10
static int num_of_gmove_cache_comm = 0;
static MPI_Comm gmove_cache_comm[GMOVE_COMM_CACHE_SIZE];
static int save_key[GMOVE_COMM_CACHE_SIZE];

static bool is_cache_comm(int key){
  bool flag = false;

  if(num_of_gmove_cache_comm == 0)
    return false;
  else{
    for(int i=num_of_gmove_cache_comm-1;i!=-1;i--){
      if(save_key[i] == key){
	flag = true;
      }
    }
  }
  return flag;
}

static void delete_first_cache_comm(){
  for(int i=0;i<GMOVE_COMM_CACHE_SIZE-1;i++){
    save_key[i] = save_key[i+1];
    gmove_cache_comm[i] = gmove_cache_comm[i+1];
  }
  num_of_gmove_cache_comm--;
}

static void insert_cache_comm(int key, MPI_Comm comm){
  if(num_of_gmove_cache_comm == GMOVE_COMM_CACHE_SIZE-1)
    delete_first_cache_comm();

  save_key[num_of_gmove_cache_comm] = key;
  gmove_cache_comm[num_of_gmove_cache_comm] = comm;
  num_of_gmove_cache_comm++;
}

static MPI_Comm get_cache_comm(int key){
  MPI_Comm newcomm;
  
  for(int i=num_of_gmove_cache_comm-1;i!=-1;i--){
    if(save_key[i] == key){
      newcomm = gmove_cache_comm[i];
    }
  }

  return newcomm;
}

// Fix me, support only 2 dimentional array
void _XMP_gmove_BCAST_TO_NOTALIGNED_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array, int type, size_t type_size, ...){
  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;
  va_list args;
  va_start(args, type_size);

  // get dst info
  void *dst_addr = dst_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim];
  int tmp_dst_u[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    tmp_dst_u[i] = va_arg(args, int);
    dst_u[i] = tmp_dst_u[i] + dst_l[i];
    dst_s[i] = va_arg(args, int);
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
  }

  // get src info
  void *src_addr = src_array->array_addr_p;
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim];
  int tmp_src_u[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int); 
    tmp_src_u[i] = va_arg(args, int);
    src_u[i] = tmp_src_u[i] + src_l[i];
    src_s[i] = va_arg(args, int);
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
  }
  va_end(args);

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  _XMP_template_t *dst_template = dst_array->align_template;  // Note: dst_template and src_template are the same.
  _XMP_template_t *src_template = src_array->align_template;

  int dst_local_start[2], dst_local_end[2], dst_local_size[2];
  int src_local_start[2], src_local_end[2], src_local_size[2]; 

  int not_aligned_dim = -1;
  for(int i=0;i<dst_array->dim;i++){
    if(_XMP_N_ALIGN_NOT_ALIGNED == dst_array->info[i].align_manner)
      not_aligned_dim = i;
  }

  // Support number of root is 1. Fix me.

  // Must be the same shape of both arrays
  if(not_aligned_dim == 0){
    if(!(dst_l[1] == src_l[1] && dst_s[1] == src_s[1])){
      _XMP_gmove_SENDRECV_ARRAY(dst_array, src_array, type, type_size, 
				dst_l[0], tmp_dst_u[0], dst_s[0], dst_d[0],
				dst_l[1], tmp_dst_u[1], dst_s[1], dst_d[1],
				src_l[0], tmp_src_u[0], src_s[0], src_d[0],
				src_l[1], tmp_src_u[1], src_s[1], src_d[1]);
      return;
    }
    else if(not_aligned_dim == 1){
      if(!(dst_l[0] == src_l[0] && dst_s[0] == src_s[0])){
	_XMP_gmove_SENDRECV_ARRAY(dst_array, src_array, type, type_size,
				  dst_l[0], tmp_dst_u[0], dst_s[0], dst_d[0],
				  dst_l[1], tmp_dst_u[1], dst_s[1], dst_d[1],
				  src_l[0], tmp_src_u[0], src_s[0], src_d[0],
				  src_l[1], tmp_src_u[1], src_s[1], src_d[1]);
	return;
      }
    }
  }


  if(not_aligned_dim == -1)
    _XMP_fatal("All dimensions are aligned");
  else if(not_aligned_dim >= 2)
    _XMP_fatal("Not implemented");

  if(not_aligned_dim == 0){
    xmp_sched_template_index(&dst_local_start[0], &dst_local_end[0],
			     dst_l[1], dst_u[1], dst_s[1], dst_template, 0);  // Not aligned     
    dst_local_start[1] = dst_l[0]; dst_local_end[1] = dst_u[0];
  }
  else if(not_aligned_dim == 1){
    xmp_sched_template_index(&dst_local_start[1], &dst_local_end[1],
			     dst_l[0], dst_u[0], dst_s[0], dst_template, 1);
    dst_local_start[0] = dst_l[1]; dst_local_end[0] = dst_u[1];
  }

  dst_local_size[0] = dst_local_end[0] - dst_local_start[0];
  dst_local_size[1] = dst_local_end[1] - dst_local_start[1];
  unsigned long long dst_local_length = dst_local_size[0] * dst_local_size[1];

  xmp_sched_template_index(&src_local_start[0], &src_local_end[0],
			   src_l[1], src_u[1], src_s[1], src_template, 0);
  xmp_sched_template_index(&src_local_start[1], &src_local_end[1],
			   src_l[0], src_u[0], src_s[0], src_template, 1);
  src_local_size[0] = src_local_end[0] - src_local_start[0];
  src_local_size[1] = src_local_end[1] - src_local_start[1];
  // Memo: src_local_start[1] is y axis, src_local_start[0] is x axis, 

  unsigned long long src_local_length = src_local_size[0] * src_local_size[1];
  void *buf = malloc(type_size * dst_local_length);

  if(src_local_length != 0){ // packing  
    unsigned long long x_dim_elmts = src_array->info[0].dim_elmts;
    for(int i=0;i<src_local_size[1];i++){
      memcpy((char *)buf + src_local_size[0]*type_size*i, 
	     (char *)src_addr+(x_dim_elmts*(i+src_local_start[1])+src_local_start[0])*type_size, 
	     src_local_size[0]*type_size);
    }
  } // end packing

  int process_size = src_array_nodes->info[0].size;
  int root=0, color=0, key=0;
  if(not_aligned_dim == 0){
    root = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(src_array, src_l) / process_size;
    color = src_array_nodes->comm_rank % src_array_nodes->info[0].size;
    key   = src_array_nodes->comm_rank / src_array_nodes->info[0].size;
  }
  else if(not_aligned_dim == 1){
    root = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(src_array, src_l) % process_size;  
    color = src_array_nodes->comm_rank / src_array_nodes->info[0].size;
    key   = src_array_nodes->comm_rank % src_array_nodes->info[0].size;
  }

  MPI_Comm newcomm;
  if(is_cache_comm(not_aligned_dim)){   // Communicator has been cached
    newcomm = get_cache_comm(not_aligned_dim);
  }
  else{
    MPI_Comm_split(*((MPI_Comm*)src_array->align_comm), color, key, &newcomm);
    insert_cache_comm(not_aligned_dim, newcomm);
  }

  if(dst_local_length != 0){
    MPI_Bcast(buf, dst_local_length, mpi_datatype, root, newcomm);

    // unpacking
    for(int i=0;i<dst_local_size[1];i++){
      unsigned long long x_dim_elmts = dst_array->info[0].dim_elmts;
      memcpy((char *)dst_addr+(x_dim_elmts*(i+dst_local_start[1])+dst_local_start[0])*type_size,
	     (char *)buf + dst_local_size[0]*type_size*i,
	     dst_local_size[0]*type_size);
    } // end unpaking
  }
  free(buf);
}
