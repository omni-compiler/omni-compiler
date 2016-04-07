#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

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


void (*_XMP_pack_comm_set)(void *sendbuf, int sendbuf_size,
			   _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
void (*_XMP_unpack_comm_set)(void *recvbuf, int recvbuf_size,
			     _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

static void _XMPC_pack_comm_set(void *sendbuf, int sendbuf_size,
				_XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);
static void _XMPC_unpack_comm_set(void *recvbuf, int recvbuf_size,
				  _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]);

static void _XMP_gmove_1to1(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int mode);

#ifdef _XMP_MPI3_ONESIDED
static void _XMP_gmove_inout(_XMP_gmv_desc_t *gmv_desc_org, _XMP_gmv_desc_t *gmv_desc_tgt,
			     _XMP_comm_set_t *org_comm_set[][_XMP_N_MAX_DIM],
			     int rdma_type);
#endif

#define XMP_DBG 0
#define DBG_RANK 3
#define XMP_DBG_OWNER_REGION 0

_XMP_nodes_t *gmv_nodes;
int n_gmv_nodes;

int (*_alloc_size)[_XMP_N_MAX_DIM];
int _dim_alloc_size;


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
      if (onto_nodes_index == -1) continue;
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

void _XMP_gmove_bcast_SCALAR(void *dst_addr, void *src_addr, size_t type_size, int root_rank) {
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
      _XMP_fatal("wrong assign statement for gmove");
    } else {
      (*_xmp_pack_array)(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
    }
  }

  _XMP_gmove_bcast(buffer, type_size, dst_buffer_elmts, root_rank);

  (*_xmp_unpack_array)(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
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
    _XMP_fatal("wrong assign statement for gmove");
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);
  (*_xmp_pack_array)(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
  (*_xmp_unpack_array)(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
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
          _XMP_fatal("wrong assign statement for gmove");
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
          _XMP_fatal("wrong assign statement for gmove");
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
    _XMP_fatal("wrong assign statement for gmove");
  }

  // recv phase
  void *recv_buffer = NULL;
  void *recv_alloc = NULL;
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

      for (int i = 0; i < dst_dim; i++) {
	_XMP_gtol_array_ref_triplet(dst_array, i, &(dst_lower[i]), &(dst_upper[i]), &(dst_stride[i]));
      }
      if(dst_dim == 1 && dst_stride[0] == 1){
	recv_buffer = (char*)dst_addr + type_size * dst_lower[0];
      }else{
	recv_alloc = _XMP_alloc(total_elmts * type_size);
	recv_buffer = recv_alloc;
      }
      MPI_Irecv(recv_buffer, total_elmts, *mpi_datatype, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
      //      fprintf(stderr, "DEBUG: Proc(%d), Irecv(src=%d, total_elmnt=%llu)\n", exec_rank, src_rank, total_elmts);
    }
  }

  // send phase
  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      void *send_buffer = NULL;
      void *send_alloc = NULL;
      for (int j = 0; j < src_dim; j++) {
        _XMP_gtol_array_ref_triplet(src_array, j, &(src_lower[j]), &(src_upper[j]), &(src_stride[j]));
      }
      if(src_dim == 1 && src_stride[0] == 1){
	send_buffer = (char*)src_addr + type_size * src_lower[0];
      }else{
	send_alloc = _XMP_alloc(total_elmts * type_size);
	send_buffer = send_alloc;
	(*_xmp_pack_array)(send_buffer, src_addr, type, type_size, src_dim, src_lower, src_upper, src_stride, src_dim_acc);
      }
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        if (i < dst_shrink_nodes_size) {
          MPI_Send(send_buffer, total_elmts, *mpi_datatype, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Send(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[i], total_elmts);
        }
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size - 1, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(send_buffer, total_elmts, *mpi_datatype, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm,
                    requests + request_count);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Isend(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[j], total_elmts);
          request_count++;
        }

        MPI_Waitall(request_size, requests, MPI_STATUSES_IGNORE);
        _XMP_free(requests);
      }

      _XMP_free(send_alloc);
    }
  }

  // wait recv phase
  if (wait_recv) {
    MPI_Wait(&gmove_request, MPI_STATUS_IGNORE);
    if(! (dst_dim == 1 && dst_stride[0] == 1)){
      (*_xmp_unpack_array)(dst_addr, recv_buffer, type, type_size, dst_dim, dst_lower, dst_upper, dst_stride, dst_dim_acc);
    }
    _XMP_free(recv_alloc);
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
  int array_dim = array->dim;
  int ref_index[array_dim];

  for (int i = 0; i < array_dim; i++) {
    ref_index[i] = va_arg(args, int);
  }
  //root_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, ref_index);

  int mode = va_arg(args, int);
  va_end(args);

  if (mode == _XMP_N_GMOVE_NORMAL){
    root_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, ref_index);
    _XMP_gmove_bcast_SCALAR(dst_addr, src_addr, type_size, root_rank);
  }
  else if (mode == _XMP_N_GMOVE_IN){
#ifdef _XMP_MPI3_ONESIDED
    
    int dummy0[7] = { 0, 0, 0, 0, 0, 0, 0 }; /* temporarily assuming maximum 7-dimensional */
    int dummy1[7] = { 1, 1, 1, 1, 1, 1, 1 }; /* temporarily assuming maximum 7-dimensional */

    _XMP_gmv_desc_t gmv_desc;

    gmv_desc.is_global = true;
    gmv_desc.ndims = array_dim;
    gmv_desc.a_desc = array;

    gmv_desc.local_data = NULL;
    gmv_desc.a_lb = NULL;
    gmv_desc.a_ub = NULL;

    gmv_desc.kind = dummy1; // always index
    gmv_desc.lb = ref_index;
    gmv_desc.ub = ref_index;
    gmv_desc.st = dummy0;

    _XMP_gmove_inout_scalar(dst_addr, &gmv_desc, _XMP_N_COARRAY_GET);
#else
    _XMP_fatal("Not supported gmove in/out on non-MPI3 environments");
#endif
  }
  else {
    _XMP_fatal("_XMP_gmove_BCAST_SCALAR: wrong gmove mode");
  }

}

void _XMP_gmove_BCAST_GSCALAR(void *dst_addr, void *src_addr, _XMP_array_t *array, int ref_index[]){
  int type_size = array->type_size;

  if(_XMP_IS_SINGLE) {
    memcpy(dst_addr, src_addr, type_size);
    return;
  }

  int root_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, ref_index);

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
  //int mode = va_arg(args, int);
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
  int dst_array_dim = dst_array->dim;
  int dst_ref_index[dst_array_dim];

  for (int i = 0; i < dst_array_dim; i++) {
    dst_ref_index[i] = va_arg(args, int);
  }
  //dst_ref = _XMP_create_gmove_nodes_ref_SCALAR(dst_array, dst_ref_index);

  _XMP_nodes_ref_t *src_ref;
  int src_array_dim = src_array->dim;
  int src_ref_index[src_array_dim];

  for (int i = 0; i < src_array_dim; i++) {
    src_ref_index[i] = va_arg(args, int);
  }
  //src_ref = _XMP_create_gmove_nodes_ref_SCALAR(src_array, src_ref_index);

  int mode = va_arg(args, int);
  va_end(args);

  if (mode == _XMP_N_GMOVE_NORMAL){

    dst_ref = _XMP_create_gmove_nodes_ref_SCALAR(dst_array, dst_ref_index);
    src_ref = _XMP_create_gmove_nodes_ref_SCALAR(src_array, src_ref_index);

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
  else {
    int dummy0[7] = { 0, 0, 0, 0, 0, 0, 0 }; /* temporarily assuming maximum 7-dimensional */
    int dummy1[7] = { 1, 1, 1, 1, 1, 1, 1 }; /* temporarily assuming maximum 7-dimensional */

    _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

    gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = true;
    gmv_desc_leftp.ndims = dst_array_dim;  gmv_desc_rightp.ndims = src_array_dim;

    gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = src_array;

    gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = NULL;
    gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = NULL;
    gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

    gmv_desc_leftp.kind = dummy1;          gmv_desc_rightp.kind = dummy1; // always index
    gmv_desc_leftp.lb = dst_ref_index;     gmv_desc_rightp.lb = src_ref_index;
    gmv_desc_leftp.ub = dst_ref_index;     gmv_desc_rightp.ub = src_ref_index;
    gmv_desc_leftp.st = dummy0;            gmv_desc_rightp.st = dummy0;

    unsigned long long src_d[src_array_dim];
    for (int i = 0; i < src_array_dim; i++) {
      src_d[i] = src_array->info[i].dim_acc;
    }

    unsigned long long dst_d[dst_array_dim];
    for (int i = 0; i < dst_array_dim; i++) {
      dst_d[i] = dst_array->info[i].dim_acc;
    }

    _XMP_pack_comm_set = _XMPC_pack_comm_set;
    _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

    _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
    				  dst_ref_index, dst_ref_index, dummy0, dst_d,
    				  src_ref_index, src_ref_index, dummy0, src_d,
    				  mode);
  }

}


void _XMP_gmove_SENDRECV_GSCALAR(void *dst_addr, void *src_addr,
				 _XMP_array_t *dst_array, _XMP_array_t *src_array,
				 int dst_ref_index[], int src_ref_index[]) {
  _XMP_ASSERT(dst_array->type_size == src_array->type_size);
  size_t type_size = dst_array->type_size;

  if(_XMP_IS_SINGLE) {
    memcpy(dst_addr, src_addr, type_size);
    return;
  }

  _XMP_nodes_ref_t *dst_ref = _XMP_create_gmove_nodes_ref_SCALAR(dst_array, dst_ref_index);
  _XMP_nodes_ref_t *src_ref = _XMP_create_gmove_nodes_ref_SCALAR(src_array, src_ref_index);

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
/* void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...) { */
/*   // skip counting elmts: _XMP_gmove_localcopy_ARRAY() counts elmts */

/*   _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp; */
/*   va_list args; */
/*   va_start(args, type_size); */

/*   // get dst info */
/*   void *dst_addr = va_arg(args, void *); */
/*   int dst_dim = va_arg(args, int); */
/*   int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim]; unsigned long long dst_d[dst_dim]; */
/*   for (int i = 0; i < dst_dim; i++) { */
/*     dst_l[i] = va_arg(args, int); */
/*     int size = va_arg(args, int); */
/*     dst_s[i] = va_arg(args, int); */
/*     dst_u[i] = dst_l[i] + (size - 1) * dst_s[i]; */
/*     dst_d[i] = va_arg(args, unsigned long long); */
/*     _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i])); */
/*   } */

/*   // get src info */
/*   void *src_addr = va_arg(args, void *); */
/*   int src_dim = va_arg(args, int); */
/*   int src_l[src_dim], src_u[src_dim], src_s[src_dim]; unsigned long long src_d[src_dim]; */
/*   for (int i = 0; i < src_dim; i++) { */
/*     src_l[i] = va_arg(args, int); */
/*     int size = va_arg(args, int); */
/*     src_s[i] = va_arg(args, int); */
/*     src_u[i] = src_l[i] + (size - 1) * src_s[i]; */
/*     src_d[i] = va_arg(args, unsigned long long); */
/*     _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i])); */
/*   } */

/*   //int mode = va_arg(args, int); */

/*   va_end(args); */

/*   _XMP_gmove_localcopy_ARRAY(type, type_size, */
/*                              dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d, */
/*                              src_addr, src_dim, src_l, src_u, src_s, src_d); */
/* } */

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
        dst_chunk_size[dst_block_dim]=dst_array->info[dst_block_dim].par_size;
      }else if(i==src_block_dim){
        dst_chunk_size[src_block_dim]=src_array->info[src_block_dim].par_size;
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
      irank=-1;
      idiv=0;
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


static void _XMP_gmove_garray_garray_block_cyclic(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
						  int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
						  int *src_l, int *src_u, int *src_s, unsigned long long *src_d){

  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

  int type = dst_array->type;
  int type_size = dst_array->type_size;

  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;

  //

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

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

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

}


static int _XMP_gmove_garray_garray_opt(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
					int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
					int *src_l, int *src_u, int *src_s, unsigned long long *src_d){

  _XMP_ASSERT(gmv_desc_leftp->is_global);
  _XMP_ASSERT(gmv_desc_rightp->is_global);

  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

  int type = dst_array->type;
  int type_size = dst_array->type_size;

  void *dst_addr = dst_array->array_addr_p;
  void *src_addr = src_array->array_addr_p;

  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;

  _XMP_template_t *dst_template = dst_array->align_template;
  _XMP_template_t *src_template = src_array->align_template;

  _XMP_nodes_t *dst_nodes = dst_template->onto_nodes;
  _XMP_nodes_t *src_nodes = src_template->onto_nodes;

  int dst_comm_size = dst_nodes->comm_size;
  int src_comm_size = src_nodes->comm_size;

  int exec_comm_size = _XMP_get_execution_nodes()->comm_size;

  if (exec_comm_size != dst_comm_size || exec_comm_size != src_comm_size) return 0;

  //
  // First, try optimized comms.
  //

  if (is_same_array_shape(dst_array, src_array) &&
      is_whole(gmv_desc_leftp) && is_whole(gmv_desc_rightp)){

    if (is_same_template_shape(dst_template, src_template) &&
	is_same_axis(dst_array, src_array) &&
	is_same_offset(dst_array, src_array) &&
	is_same_alignmanner(dst_array, src_array)){

      //
      // just local copy (no comms.)
      //

      for (int i = 0; i < dst_dim; i++) {
	dst_l[i] = dst_array->info[i].local_lower;
	dst_u[i] = dst_array->info[i].local_upper;
	dst_s[i] = dst_array->info[i].local_stride;
      }

      if (dst_array->is_allocated){ // src_array->is_allocated
	_XMP_gmove_localcopy_ARRAY(type, type_size,
				   dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
				   src_addr, src_dim, dst_l, dst_u, dst_s, dst_d);
      }

      return 1;

    }
    else if (is_one_block(dst_array) && is_one_block(src_array) &&
	     dst_array->dim >= dst_array->align_template->dim &&
	     src_array->dim >= src_array->align_template->dim){

      //
      // transpose
      //
	
      if (_XMP_gmove_transpose(gmv_desc_leftp, gmv_desc_rightp)) return 1;
    }

  }

  //
  // Next, do the general algorithm
  //
    
  // check if _XMP_sendrecv_ARRAY can be applied.

  for (int i = 0; i < dst_template->dim; i++){
    if (dst_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK &&
	dst_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC){
      return 0;
    }
  }

  for (int i = 0; i < src_template->dim; i++){
    if (src_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK &&
	src_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC){
      return 0;
    }
  }

  if (dst_dim == dst_nodes->dim){

    if (src_dim == src_nodes->dim){

      for (int i = 0; i < dst_dim; i++){
	if (dst_array->info[i].align_subscript != 0 ||
	    (dst_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK &&
	     dst_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC)){
	  return 0;
	}
      }

      for (int i = 0; i < src_dim; i++){
	if (src_array->info[i].align_subscript != 0 ||
	    (src_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK &&
	     src_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC)){
	  return 0;
	}
      }

    }
    else return 0;

    if (!is_whole(gmv_desc_leftp) || !is_whole(gmv_desc_rightp)) return 0;

  }
  else if (dst_dim < dst_nodes->dim){
    if (src_dim < src_nodes->dim){
      if (dst_nodes->dim != 2 || src_nodes->dim != 2 ||
	  dst_array->info[0].align_subscript != 0 ||
	  src_array->info[0].align_subscript != 0){
	return 0;
      }
    }
    else return 0;
  }
  else if (dst_dim > dst_nodes->dim){
    if (src_dim > src_nodes->dim){
      if (_XMPF_running == 1 && _XMPC_running == 0) return 0;
    }
    else return 0;
  }

  _XMP_gmove_garray_garray_block_cyclic(gmv_desc_leftp, gmv_desc_rightp,
					dst_l, dst_u, dst_s, dst_d,
					src_l, src_u, src_s, src_d);

  return 1;

}


void _XMP_gmove_array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
				   int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
				   int *src_l, int *src_u, int *src_s, unsigned long long *src_d,
				   int mode){

  // NOTE: asynchronous gmove aloways done by _XMP_gmove_1to1
  if (!xmp_is_async() && gmv_desc_leftp->is_global && gmv_desc_rightp->is_global && mode == _XMP_N_GMOVE_NORMAL){
    if (_XMP_gmove_garray_garray_opt(gmv_desc_leftp, gmv_desc_rightp,
  				     dst_l, dst_u, dst_s, dst_d,
  				     src_l, src_u, src_s, src_d)) return;
    // fall through
  }

  _XMP_gmove_1to1(gmv_desc_leftp, gmv_desc_rightp, mode);

  return;

}


void _XMP_gmove_BCAST_ARRAY(_XMP_array_t *src_array, int type, size_t type_size, ...) {
  //unsigned long long gmove_total_elmts = 0;

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
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    dst_a_lb[i]=0;
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  //void *src_addr = src_array->array_addr_p;
  int src_dim = src_array->dim;
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

  int mode = va_arg(args, int);

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("wrong assign statement for gmove");
  } else {
    //gmove_total_elmts = dst_total_elmts;
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

  _XMP_ASSERT(gmv_desc_rightp->a_desc);

  // create a temporal descriptor for the "non-distributed" LHS array (to be possibly used
  // in _XMP_gmove_1to1)
  _XMP_array_t *a;
  _XMP_init_array_desc_NOT_ALIGNED(&a, src_array->align_template, dst_dim,
				   src_array->type, src_array->type_size, dst_d, dst_addr);
  gmv_desc_leftp.a_desc = a;

  _XMP_pack_comm_set = _XMPC_pack_comm_set;
  _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

  _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
				dst_l, dst_u, dst_s, dst_d,
				src_l, src_u, src_s, src_d,
				mode);

  _XMP_finalize_array_desc(a);

  /* int iflag =0; */
  /* if (iflag==1){ */
  /*   if (_XMP_IS_SINGLE) { */
  /*     _XMP_gmove_localcopy_ARRAY(type, type_size, */
  /*                                dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d, */
  /*                                src_addr, src_dim, src_l, src_u, src_s, src_d); */
  /*     return; */
  /*   } */

  /*   _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes(); */
  /*   _XMP_ASSERT(exec_nodes->is_member); */

  /*   _XMP_nodes_t *array_nodes = src_array->array_nodes; */
  /*   int array_nodes_dim = array_nodes->dim; */
  /*   int array_nodes_ref[array_nodes_dim]; */
  /*   for (int i = 0; i < array_nodes_dim; i++) { */
  /*     array_nodes_ref[i] = 0; */
  /*   } */

  /*   int dst_lower[dst_dim], dst_upper[dst_dim], dst_stride[dst_dim]; */
  /*   int src_lower[src_dim], src_upper[src_dim], src_stride[src_dim]; */
  /*   do { */
  /*     for (int i = 0; i < dst_dim; i++) { */
  /*       dst_lower[i] = dst_l[i]; dst_upper[i] = dst_u[i]; dst_stride[i] = dst_s[i]; */
  /*     } */

  /*     for (int i = 0; i < src_dim; i++) { */
  /*       src_lower[i] = src_l[i]; src_upper[i] = src_u[i]; src_stride[i] = src_s[i]; */
  /*     } */

  /*     if (_XMP_calc_global_index_BCAST(dst_dim, dst_lower, dst_upper, dst_stride, */
  /*                                    src_array, array_nodes_ref, src_lower, src_upper, src_stride)) { */
  /*       int root_rank = _XMP_calc_linear_rank_on_target_nodes(array_nodes, array_nodes_ref, exec_nodes); */
  /*       if (root_rank == (exec_nodes->comm_rank)) { */
  /*         for (int i = 0; i < src_dim; i++) { */
  /*           _XMP_gtol_array_ref_triplet(src_array, i, &(src_lower[i]), &(src_upper[i]), &(src_stride[i])); */
  /*         } */
  /*       } */

  /*       gmove_total_elmts -= _XMP_gmove_bcast_ARRAY(dst_addr, dst_dim, dst_lower, dst_upper, dst_stride, dst_d, */
  /*                                                   src_addr, src_dim, src_lower, src_upper, src_stride, src_d, */
  /*                                                   type, type_size, root_rank); */

  /*       _XMP_ASSERT(gmove_total_elmts >= 0); */
  /*       if (gmove_total_elmts == 0) { */
  /*         return; */
  /*       } */
  /*     } */
  /*   } while (_XMP_get_next_rank(array_nodes, array_nodes_ref)); */
  /* } */
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

  int mode = va_arg(args, int);

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("wrong assign statement for gmove");
  }

  if (_XMP_IS_SINGLE) {
    _XMP_gmove_localcopy_ARRAY(type, type_size,
                               dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                               src_addr, src_dim, src_l, src_u, src_s, src_d);
    return;
  }

  if (mode == _XMP_N_GMOVE_OUT){

#ifdef _XMP_MPI3_ONESIDED

    // create a temporal descriptor for the "non-distributed" LHS array (to be possibly used
    // in _XMP_gmove_1to1)
    _XMP_array_t *a;
    _XMP_init_array_desc_NOT_ALIGNED(&a, dst_array->align_template, src_dim,
				     dst_array->type, dst_array->type_size, src_d, src_addr);

    int dummy0[_XMP_N_MAX_DIM] = { 0, 0, 0, 0, 0, 0, 0 }; /* temporarily assuming maximum 7-dimensional */
    int dummy2[_XMP_N_MAX_DIM] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

    _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

    gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = false;
    gmv_desc_leftp.ndims = dst_array->dim; gmv_desc_rightp.ndims = src_dim;

    gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = a;

    gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = src_addr;
    gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = dummy0;
    gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

    gmv_desc_leftp.kind = dummy2;          gmv_desc_rightp.kind = dummy2; // always triplet
    gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_l;
    gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_u;
    gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = src_s;

    _XMP_pack_comm_set = _XMPC_pack_comm_set;
    _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

    _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
				  dst_l, dst_u, dst_s, dst_d,
				  src_l, src_u, src_s, src_d,
				  mode);

    _XMP_finalize_array_desc(a);

    return;
#else
    _XMP_fatal("Not supported gmove in/out on non-MPI3 environments");
#endif
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
          _XMP_fatal("wrong assign statement for gmove");
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
    _XMP_fatal("wrong assign statement for gmove");
  }

  void *buffer = _XMP_alloc(dst_buffer_elmts * type_size);
  (*_xmp_pack_array)(buffer, src_addr, type, type_size, src_dim, src_l, src_u, src_s, src_d);
  (*_xmp_unpack_array)(dst_addr, buffer, type, type_size, dst_dim, dst_l, dst_u, dst_s, dst_d);
  _XMP_free(buffer);
}


void _XMP_gmove_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
                               int type, size_t type_size, ...) {

  _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

  va_list args;
  va_start(args, type_size);

  // get dst info
  unsigned long long dst_total_elmts = 1;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    if (dst_s[i] != 0) dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim];
  unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    src_s[i] = va_arg(args, int);
    src_u[i] = src_l[i] + (size - 1) * src_s[i];
    src_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    if (src_s[i] != 0) src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  int mode = va_arg(args, int);

  va_end(args);

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("wrong assign statement for gmove");
  } else {
    //gmove_total_elmts = dst_total_elmts;
  }

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

  _XMP_pack_comm_set = _XMPC_pack_comm_set;
  _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

  _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
				dst_l, dst_u, dst_s, dst_d,
				src_l, src_u, src_s, src_d,
				mode); 
}


#if 0
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
  MPI_Comm newcomm = NULL;
  
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

  int mode = va_arg(args, int);

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
#endif


static _XMP_csd_t*
get_owner_csd(_XMP_array_t *a, int adim, int ncoord[]){

  _XMP_array_info_t *ainfo = &(a->info[adim]);
  int tdim = ainfo->align_template_index;

  _XMP_template_info_t *tinfo = NULL;
  _XMP_template_chunk_t *tchunk = NULL;
  int ndim = 0;
  int nidx = 0;

#if XMP_DBG_OWNER_REGION
  int myrank = gmv_nodes->comm_rank;
#endif

  if (tdim != -1){
    // neither _XMP_N_ALIGN_DUPLICATION nor _XMP_N_ALIGN_NOT_ALIGNED:
    tinfo = &(a->align_template->info[tdim]);
    tchunk = &(a->align_template->chunk[tdim]);
    ndim = tchunk->onto_nodes_index;
    nidx = ncoord[ndim];
  }

  _XMP_bsd_t bsd = { 0, 0, 0, 0 };

  switch (ainfo->align_manner){

  case _XMP_N_ALIGN_DUPLICATION:
  case _XMP_N_ALIGN_NOT_ALIGNED:
    bsd.l = ainfo->ser_lower;
    bsd.u = ainfo->ser_upper;
    bsd.b = 1;
    bsd.c = 1;
    _dim_alloc_size = ainfo->ser_upper - ainfo->ser_lower + 1;
    return bsd2csd(&bsd);
    break;
	
  case _XMP_N_ALIGN_BLOCK:
    {
      /* printf("[%d] ser_lower = %d, ser_upper = %d, par_lower = %d, par_upper = %d, subscript = %d\n",*/
      /* 	   nidx, ainfo->ser_lower, ainfo->ser_upper, ainfo->par_lower, ainfo->par_upper, */
      /* 	   ainfo->align_subscript); */
      /* bsd.l = tinfo->ser_lower + tchunk->par_chunk_width * nidx - ainfo->align_subscript; */
      /* bsd.u = MIN(tinfo->ser_lower + tchunk->par_chunk_width * (nidx + 1) - 1 - ainfo->align_subscript, */
      /* 		  tinfo->ser_upper - ainfo->align_subscript); */

      long long align_lower = ainfo->ser_lower + ainfo->align_subscript;
      long long align_upper = ainfo->ser_upper + ainfo->align_subscript;

      long long template_lower = tinfo->ser_lower + tchunk->par_chunk_width * nidx;
      long long template_upper = MIN(tinfo->ser_lower + tchunk->par_chunk_width * (nidx + 1) - 1,
				     tinfo->ser_upper);

      if (align_lower < template_lower) {
      	bsd.l = template_lower - ainfo->align_subscript;
      }
      else if (template_upper < align_lower) {
      	return NULL;
      }
      else {
      	bsd.l = ainfo->ser_lower;
      }

      if (align_upper < template_lower) {
      	return NULL;
      }
      else if (template_upper < align_upper) {
      	bsd.u = template_upper - ainfo->align_subscript;
      }
      else {
      	bsd.u = ainfo->ser_upper;
      }

      bsd.b = 1;
      bsd.c = 1;

      _dim_alloc_size = bsd.u - bsd.l + 1 + ainfo->shadow_size_lo + ainfo->shadow_size_hi;

      return bsd2csd(&bsd);
    }

    break;

  case _XMP_N_ALIGN_CYCLIC:
  case _XMP_N_ALIGN_BLOCK_CYCLIC:
    {
      bsd.l = tinfo->ser_lower + (nidx * tchunk->par_width) - ainfo->align_subscript;
      bsd.u = tinfo->ser_upper - ainfo->align_subscript;
      bsd.b = tchunk->par_width;
      bsd.c = tchunk->par_stride;

      _XMP_bsd_t bsd_declare = { ainfo->ser_lower, ainfo->ser_upper, 1, 1 };

      _XMP_csd_t *csd = intersection_csds(bsd2csd(&bsd), bsd2csd(&bsd_declare));

      _dim_alloc_size = get_csd_size(csd);

      return csd;

      /* bsd.b = tchunk->par_width; */
      /* bsd.c = tchunk->par_stride; */

      /* int rank_dist = (ainfo->ser_lower + ainfo->align_subscript - tinfo->ser_lower) / tchunk->par_width; */
      /* int v_lower = tinfo->ser_lower + tchunk->par_width * rank_dist - ainfo->align_subscript; */
      /* int nsize = tchunk->onto_nodes_info->size; */
      /* int rank_lb = rank_dist % nsize; */
      /* int mod = _XMP_modi_ll_i(nidx - rank_lb, nsize); */

      /* bsd.l = MAX(ainfo->ser_lower, (long long)(mod * tchunk->par_width) + v_lower); */

      /* int dist = (ainfo->ser_upper - (mod * tchunk->par_width + v_lower)) / bsd.c; */
      /* int diff = ainfo->ser_upper - (bsd.l + (dist * bsd.c) + (tchunk->par_width - 1)); */
      /* if (diff > 0){ */
      /* 	bsd.u = bsd.l + (dist * bsd.c) + (tchunk->par_width - 1); */
      /* 	_dim_alloc_size = (dist + 1) * tchunk->par_width; */
      /* } */
      /* else { */
      /* 	bsd.u = ainfo->ser_upper; */
      /* 	_dim_alloc_size = (dist + 1) * tchunk->par_width + diff; */
      /* } */
    }

    break;

  case _XMP_N_ALIGN_GBLOCK:
    {

      long long align_lower = ainfo->ser_lower + ainfo->align_subscript;
      long long align_upper = ainfo->ser_upper + ainfo->align_subscript;

      long long template_lower = tchunk->mapping_array[nidx];
      long long template_upper = tchunk->mapping_array[nidx + 1] - 1;
				 
      if (align_lower < template_lower) {
      	bsd.l = template_lower - ainfo->align_subscript;
      }
      else if (template_upper < align_lower) {
      	return NULL;
      }
      else {
      	bsd.l = ainfo->ser_lower;
      }

      if (align_upper < template_lower) {
      	return NULL;
      }
      else if (template_upper < align_upper) {
      	bsd.u = template_upper - ainfo->align_subscript;
      }
      else {
      	bsd.u = ainfo->ser_upper;
      }

      bsd.b = 1;
      bsd.c = 1;

      _dim_alloc_size = bsd.u - bsd.l + 1 + ainfo->shadow_size_lo + ainfo->shadow_size_hi;

      return bsd2csd(&bsd);
    }

    break;
      
  default:
    _XMP_fatal("_XMP_gmove_1to1: unknown distribution format");

  }

  return NULL;

}


static void
get_owner_ref_csd(_XMP_array_t *adesc, int *lb, int *ub, int *st,
		  _XMP_csd_t *owner_ref_csd[][_XMP_N_MAX_DIM],
		  int is_remote){

  int n_adims = adesc->dim;

#if XMP_DBG_OWNER_REGION
  int myrank = gmv_nodes->comm_rank;
#endif

  //
  // referenced region
  //

  _XMP_rsd_t rsd_ref[n_adims];
  _XMP_csd_t *csd_ref[n_adims];
  for (int i = 0; i < n_adims; i++){
    if (st[i] != 0){
      rsd_ref[i].l = lb[i];
      rsd_ref[i].u = ub[i];
      rsd_ref[i].s = st[i];
    }
    else {
      rsd_ref[i].l = lb[i];
      rsd_ref[i].u = lb[i];
      rsd_ref[i].s = 1;
    }
    csd_ref[i] = rsd2csd(&rsd_ref[i]);
  }

  // local (non-distributed) data
  if (adesc->total_elmts == -1){

    for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
      for (int i = 0; i < n_adims; i++){
  	owner_ref_csd[gmv_rank][i] = copy_csd(csd_ref[i]);
	if (is_remote)
	  _alloc_size[gmv_rank][i] = adesc->info[i].ser_size;
      }
    }

    for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
      reduce_csd(owner_ref_csd[gmv_rank], n_adims);
    }
    
    for (int adim = 0; adim < n_adims; adim++){
      free_csd(csd_ref[adim]);
    }

#if XMP_DBG_OWNER_REGION
    if (myrank == DBG_RANK){
      for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){

	printf("\n");
	printf("[%d]\n", gmv_rank);

	printf("owner_ref\n");
	for (int adim = 0; adim < n_adims; adim++){
	  printf("  %d: ", adim); print_csd(owner_ref_csd[gmv_rank][adim]);
	}

      }
    }
#endif

    return;

  }

  //
  // owner region
  //

  _XMP_csd_t *owner_csd[n_gmv_nodes][n_adims];

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){

    //if (myrank == 0) printf("gmv_rank = %d\n", gmv_rank);

    int gmv_ncoord[_XMP_N_MAX_DIM];
    _XMP_calc_rank_array(gmv_nodes, gmv_ncoord, gmv_rank);

    _XMP_nodes_t *target_nodes = adesc->align_template->onto_nodes;
    int target_ncoord[_XMP_N_MAX_DIM];

    if (_XMP_calc_coord_on_target_nodes(gmv_nodes, gmv_ncoord, target_nodes, target_ncoord)){

      /* if (myrank == 0) printf("gmv_rank = %d, n[0] = %d, n[1] = %d\n", gmv_rank, */
      /* 			      target_ncoord[0], target_ncoord[1]); */

      for (int i = 0; i < target_nodes->dim; i++){
	/* if (myrank == 0) printf("(%d) nidx = %d, size = %d\n", */
	/* 			i, target_ncoord[i], target_nodes->info[i].size); */
	if (target_ncoord[i] < 0 || target_ncoord[i] >= target_nodes->info[i].size){
	  for (int adim = 0; adim < n_adims; adim++){
	    owner_csd[gmv_rank][adim] = NULL;
	  }
	  goto next;
	}
      }

      for (int adim = 0; adim < n_adims; adim++){
  	owner_csd[gmv_rank][adim] = get_owner_csd(adesc, adim, target_ncoord);
	if (is_remote)
	  _alloc_size[gmv_rank][adim] = _dim_alloc_size;
      }

    }
    else {
      _XMP_fatal("_XMP_gmove_1to1: array not allocated on the executing node array");
    }

  next:;

  }

  //
  // intersection of reference and owner region
  //

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    for (int adim = 0; adim < n_adims; adim++){
      owner_ref_csd[gmv_rank][adim] = intersection_csds(owner_csd[gmv_rank][adim], csd_ref[adim]);
    }
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    reduce_csd(owner_ref_csd[gmv_rank], n_adims);
  }

#if XMP_DBG_OWNER_REGION
  fflush(stdout);
  xmp_barrier();
  if (myrank == DBG_RANK){
    for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){

      printf("\n");
      printf("[%d]\n", gmv_rank);

      printf("owner\n");
      for (int adim = 0; adim < n_adims; adim++){
  	printf("  %d: ", adim); print_csd(owner_csd[gmv_rank][adim]);
      }

      printf("ref\n");
      for (int adim = 0; adim < n_adims; adim++){
  	printf("  %d: ", adim); print_csd(csd_ref[adim]);
      }

      printf("owner_ref\n");
      for (int adim = 0; adim < n_adims; adim++){
      	printf("  %d: ", adim); print_csd(owner_ref_csd[gmv_rank][adim]);
      }

    }
  }
#endif

  for (int adim = 0; adim < n_adims; adim++){
    free_csd(csd_ref[adim]);
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    for (int adim = 0; adim < n_adims; adim++){
      free_csd(owner_csd[gmv_rank][adim]);
    }
  }

}


static int
get_commbuf_size(_XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM], int ndims, int counts[]){

  //int myrank = gmv_nodes->comm_rank;

  int buf_size = 0;

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    int size = 1;
    for (int i = 0; i < ndims; i++){
      int dim_size = 0;
      _XMP_comm_set_t *c = comm_set[gmv_rank][i];
      while (c){
	dim_size += (c->u - c->l + 1);
	c = c->next;
      }
      size *= dim_size;
    }
    counts[gmv_rank] = size;
    //xmp_dbg_printf("buf_size[%d] = %d\n", gmv_rank, size);
    buf_size += size;
  }

  return buf_size;

}


unsigned long long _XMP_gtol_calc_offset(_XMP_array_t *a, int g_idx[]){

  int l_idx[a->dim];
  xmp_array_gtol(a, g_idx, l_idx);

  //xmp_dbg_printf("g0 = %d, g1 = %d, l0 = %d, l1 = %d\n", g_idx[0], g_idx[1], l_idx[0], l_idx[1]);

  unsigned long long offset = 0;

  for (int i = 0; i < a->dim; i++){
    offset += (l_idx[i] * a->info[i].dim_acc * a->type_size);
    //xmp_dbg_printf("(%d) acc = %llu, type_size = %d\n", i, a->info[i].dim_acc, a->type_size);
  }

  return offset;

}


static void
_XMPC_pack_comm_set(void *sendbuf, int sendbuf_size,
		    _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  int ndims = a->dim;

  char *buf = (char *)sendbuf;
  char *src = (char *)a->array_addr_p;

  for (int dst_node = 0; dst_node < n_gmv_nodes; dst_node++){

    _XMP_comm_set_t *c[ndims];

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
    	i[0] = c[0]->l;
    	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      break;

    case 2:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
    	i[1] = c[1]->l;
    	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      break;

    case 3:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
    	i[2] = c[2]->l;
    	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
    	i[3] = c[3]->l;
    	int size = (c[3]->u - c[3]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
    	i[4] = c[4]->l;
    	int size = (c[4]->u - c[4]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
    	i[5] = c[5]->l;
    	int size = (c[5]->u - c[5]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      break;

    case 7:
      for (c[0] = comm_set[dst_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[dst_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[dst_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[dst_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[dst_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[dst_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[6] = comm_set[dst_node][6]; c[6]; c[6] = c[6]->next){
    	i[6] = c[6]->l;
    	int size = (c[6]->u - c[6]->l + 1) * a->type_size;
    	memcpy(buf, src + _XMP_gtol_calc_offset(a, i), size);
    	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      }}
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;

  if (myrank == 0){
    printf("\n");
    printf("Send buffer -------------------------------------\n");
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    if (myrank == gmv_rank){
      printf("\n");
      printf("[%d]\n", myrank);
      for (int i = 0; i < sendbuf_size; i++){
  	printf("%.0f ", ((double *)sendbuf)[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    xmp_barrier();
  }
#endif

}


static void
_XMPC_unpack_comm_set(void *recvbuf, int recvbuf_size,
		      _XMP_array_t *a, _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  //int myrank = gmv_nodes->comm_rank;

  int ndims = a->dim;

  char *buf = (char *)recvbuf;
  char *dst = (char *)a->array_addr_p;

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;

  if (myrank == 0){
    printf("\n");
    printf("Recv buffer -------------------------------------\n");
  }

  for (int gmv_rank = 0; gmv_rank < n_gmv_nodes; gmv_rank++){
    if (myrank == gmv_rank){
      printf("\n");
      printf("[%d]\n", myrank);
      for (int i = 0; i < recvbuf_size; i++){
  	printf("%.0f ", ((double *)recvbuf)[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    xmp_barrier();
  }
#endif

  for (int src_node = 0; src_node < n_gmv_nodes; src_node++){

    _XMP_comm_set_t *c[ndims];

    int i[_XMP_N_MAX_DIM];

    switch (ndims){

    case 1:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	i[0] = c[0]->l;
	int size = (c[0]->u - c[0]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      break;

    case 2:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	i[1] = c[1]->l;
	int size = (c[1]->u - c[1]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	//xmp_dbg_printf("(%d, %d) offset = %03d, size = %d\n", i[0], i[1], o, size);
	buf += size;
      }
      }}
      break;

    case 3:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	i[2] = c[2]->l;
	int size = (c[2]->u - c[2]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      break;

    case 4:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	i[3] = c[3]->l;
	int size = (c[3]->u - c[3]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      break;

    case 5:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	i[4] = c[4]->l;
	int size = (c[4]->u - c[4]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      break;

    case 6:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	i[5] = c[5]->l;
	int size = (c[5]->u - c[5]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      break;

    case 7:
      for (c[0] = comm_set[src_node][0]; c[0]; c[0] = c[0]->next){
	for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
      for (c[1] = comm_set[src_node][1]; c[1]; c[1] = c[1]->next){
	for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
      for (c[2] = comm_set[src_node][2]; c[2]; c[2] = c[2]->next){
	for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
      for (c[3] = comm_set[src_node][3]; c[3]; c[3] = c[3]->next){
	for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
      for (c[4] = comm_set[src_node][4]; c[4]; c[4] = c[4]->next){
	for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
      for (c[5] = comm_set[src_node][5]; c[5]; c[5] = c[5]->next){
	for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
      for (c[6] = comm_set[src_node][6]; c[6]; c[6] = c[6]->next){
	i[6] = c[6]->l;
	int size = (c[6]->u - c[6]->l + 1) * a->type_size;
	memcpy(dst + _XMP_gtol_calc_offset(a, i), buf, size);
	buf += size;
      }
      }}
      }}
      }}
      }}
      }}
      }}
      break;

    default:
      _XMP_fatal("wrong array dimension");
    }

  }

}


//
// get the list of elements (comm. set) of array1 to be moved to/from elements of array0
//
// 0: target, 1: origin
static void
get_comm_list(_XMP_gmv_desc_t *gmv_desc0, _XMP_gmv_desc_t *gmv_desc1,
	      _XMP_csd_t *owner_ref_csd0[][_XMP_N_MAX_DIM], _XMP_csd_t *owner_ref_csd1[][_XMP_N_MAX_DIM],
	      _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM]){

  _XMP_array_t *array0 = gmv_desc0->a_desc;
  int *lb0 = gmv_desc0->lb;
  //int *ub0 = gmv_desc0->ub;
  int *st0 = gmv_desc0->st;
  int ndims0 = array0->dim;

  _XMP_array_t *array1 = gmv_desc1->a_desc;
  int *lb1 = gmv_desc1->lb;
  int *ub1 = gmv_desc1->ub;
  int *st1 = gmv_desc1->st;
  int ndims1 = array1->dim;

  //_XMP_nodes_t *gmv_nodes = _XMP_get_execution_nodes();
  int myrank = gmv_nodes->comm_rank;
  //int n_gmv_nodes = gmv_nodes->comm_size;

  _XMP_csd_t *comm_csd[n_gmv_nodes][_XMP_N_MAX_DIM];

  int scalar_flag0 = 1;
  for (int i0 = 0; i0 < ndims0; i0++) scalar_flag0 &= (st0[i0] == 0);

  /* int scalar_flag1 = 1; */
  /* for (int i1 = 0; i1 < ndims1; i1++) scalar_flag0 &= (st1[i1] == 0); */

  for (int r_rank = 0; r_rank < n_gmv_nodes; r_rank++){

    if (scalar_flag0){
      for (int i0 = 0; i0 < ndims0; i0++){
    	if (owner_ref_csd0[r_rank][i0] == NULL){
	  for (int i1 = 0; i1 < ndims1; i1++){
	    comm_csd[r_rank][i1] = NULL;
	    comm_set[r_rank][i1] = NULL;
	  }
	  goto next;
	}
      }
    }

    _XMP_csd_t *r;

    int i0 = 0;
    for (int i1 = 0; i1 < ndims1; i1++){

      if (st1[i1] == 0){
  	r = alloc_csd(1);
  	r->l[0] = lb1[i1];
  	r->u[0] = lb1[i1];
  	r->s = 1;
      }
      else if (scalar_flag0){
      	r = alloc_csd(1);
      	r->l[0] = lb1[i1];
      	r->u[0] = ub1[i1];
      	r->s = st1[i1];
      }
      else {
      	while (st0[i0] == 0 && i0 < ndims0) i0++;
      	if (i0 == ndims0) _XMP_fatal("_XMP_gmove_1to1: lhs and rhs not conformable");
      	_XMP_csd_t *l = owner_ref_csd0[r_rank][i0];
	if (l){
	  r = alloc_csd(l->n);
	  for (int i = 0; i < l->n; i++){
	    r->l[i] = (l->l[i] - lb0[i0]) * st1[i1] / st0[i0] + lb1[i1];
	    r->u[i] = (l->u[i] - lb0[i0]) * st1[i1] / st0[i0] + lb1[i1];
	  }
	  r->s = l->s * st1[i1] / st0[i0];
	}
	else {
	  r = NULL;
	}
      	i0++;
      }

      comm_csd[r_rank][i1] = intersection_csds(r, owner_ref_csd1[myrank][i1]);

      if (r) free_csd(r);

    }

    reduce_csd(comm_csd[r_rank], ndims1);

    for (int i1 = 0; i1 < ndims1; i1++){
      comm_set[r_rank][i1] = csd2comm_set(comm_csd[r_rank][i1]);
    }

  next:
    ;
  }

#if XMP_DBG
  for (int l_rank = 0; l_rank < n_gmv_nodes; l_rank++){
    xmp_barrier();
    if (myrank == l_rank){
      for (int r_rank = 0; r_rank < n_gmv_nodes; r_rank++){

  	printf("\n");
  	printf("me[%d] - [%d]\n", myrank, r_rank);

  	for (int i1 = 0; i1 < ndims1; i1++){
  	  printf("  %d: ", i1); print_comm_set(comm_set[r_rank][i1]);
  	}

      }
    }
    xmp_barrier();
    fflush(stdout);
    xmp_barrier();
  }
#endif

  for (int rank = 0; rank < n_gmv_nodes; rank++){
    for (int i1 = 0; i1 < ndims1; i1++){
      if (comm_csd[rank][i1]) free_csd(comm_csd[rank][i1]);
    }
  }

}


//_XMP_nodes_t *get_common_ancestor_nodes(_XMP_nodes_t *n0, _XMP_nodes_t *n1);

static void
_XMP_gmove_1to1(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int mode){

  _XMP_array_t *lhs_array = gmv_desc_leftp->a_desc;
  int *lhs_lb = gmv_desc_leftp->lb;
  int *lhs_ub = gmv_desc_leftp->ub;
  int *lhs_st = gmv_desc_leftp->st;
  int n_lhs_dims = gmv_desc_leftp->ndims;

  _XMP_array_t *rhs_array = gmv_desc_rightp->a_desc;
  int *rhs_lb = gmv_desc_rightp->lb;
  int *rhs_ub = gmv_desc_rightp->ub;
  int *rhs_st = gmv_desc_rightp->st;
  int n_rhs_dims = gmv_desc_rightp->ndims;

  if (mode == _XMP_N_GMOVE_NORMAL){
    gmv_nodes = _XMP_get_execution_nodes();
    n_gmv_nodes = gmv_nodes->comm_size;
  }
  else if (mode == _XMP_N_GMOVE_IN){
    //gmv_nodes = get_common_ancestor_nodes(exec_nodes, rhs_array->align_template->onto_nodes);
    gmv_nodes = _XMP_world_nodes;
    if (!gmv_nodes) _XMP_fatal("_XMP_gmove_1to1: RHS array declared on neither the executing"
				 "node array nor its ancestors");
    n_gmv_nodes = gmv_nodes->comm_size;
  }
  else { // mode == _XMP_N_GMOVE_OUT
    //gmv_nodes = get_common_ancestor_nodes(exec_nodes, lhs_array->align_template->onto_nodes);
    gmv_nodes = _XMP_world_nodes;
    if (!gmv_nodes) _XMP_fatal("_XMP_gmove_1to1: LHS array declared on neither the executing"
				 "node array nor its ancestors");
    n_gmv_nodes = gmv_nodes->comm_size;
  }

  if (mode == _XMP_N_GMOVE_IN || mode == _XMP_N_GMOVE_OUT){
    _alloc_size = _XMP_alloc(sizeof(int) * n_gmv_nodes *_XMP_N_MAX_DIM);
  }

  int rhs_is_scalar = 1;
  for (int i = 0; i < n_rhs_dims; i++) rhs_is_scalar &= (rhs_st[i] == 0);

  int myrank = gmv_nodes->comm_rank;
  MPI_Comm *gmv_comm = gmv_nodes->comm;

  //
  // Get referenced and owned section
  //

  // LHS

#if XMP_DBG_OWNER_REGION
  if (myrank == DBG_RANK){
    printf("\nLHS -------------------------------------\n");
  }
  fflush(stdout);
  xmp_barrier();
#endif

  _XMP_csd_t *lhs_owner_ref_csd[n_gmv_nodes][_XMP_N_MAX_DIM];
  get_owner_ref_csd(lhs_array, lhs_lb, lhs_ub, lhs_st, lhs_owner_ref_csd, mode == _XMP_N_GMOVE_OUT);

  // RHS

#if XMP_DBG_OWNER_REGION
  if (myrank == DBG_RANK){
    printf("\nRHS -------------------------------------\n");
  }
  fflush(stdout);
  xmp_barrier();
#endif

  _XMP_csd_t *rhs_owner_ref_csd[n_gmv_nodes][_XMP_N_MAX_DIM];
  get_owner_ref_csd(rhs_array, rhs_lb, rhs_ub, rhs_st, rhs_owner_ref_csd, mode == _XMP_N_GMOVE_IN);

  //
  // Get communication sets
  //

  // Send list

#if XMP_DBG
  if (myrank == DBG_RANK){
    printf("\nSend List -------------------------------------\n");
  }
#endif

  _XMP_comm_set_t *send_comm_set[n_gmv_nodes][_XMP_N_MAX_DIM];
  if (mode == _XMP_N_GMOVE_NORMAL || (mode == _XMP_N_GMOVE_OUT && !rhs_is_scalar)){
    get_comm_list(gmv_desc_leftp, gmv_desc_rightp, lhs_owner_ref_csd, rhs_owner_ref_csd, send_comm_set);
  }

  // Recv list

#if XMP_DBG
  if (myrank == DBG_RANK){
    printf("\nRecv List -------------------------------------\n");
  }
#endif

  _XMP_comm_set_t *(*recv_comm_set)[_XMP_N_MAX_DIM] = { NULL };
  if (mode == _XMP_N_GMOVE_NORMAL || mode == _XMP_N_GMOVE_IN){
    recv_comm_set = _XMP_alloc(sizeof(_XMP_comm_set_t*) * n_gmv_nodes *_XMP_N_MAX_DIM);
    get_comm_list(gmv_desc_rightp, gmv_desc_leftp, rhs_owner_ref_csd, lhs_owner_ref_csd, recv_comm_set);
  }
  else if (mode == _XMP_N_GMOVE_OUT && rhs_is_scalar){
    recv_comm_set = _XMP_alloc(sizeof(_XMP_comm_set_t*) * n_gmv_nodes *_XMP_N_MAX_DIM);
    for (int rank = 0; rank < n_gmv_nodes; rank++){
      for (int adim = 0; adim < n_lhs_dims; adim++){
	if (rhs_owner_ref_csd[myrank][adim])
	  recv_comm_set[rank][adim] = csd2comm_set(lhs_owner_ref_csd[rank][adim]);
	else
	  recv_comm_set[rank][adim] = NULL;
      }
    }
  }

  // free owner_ref_csd

  for (int rank = 0; rank < n_gmv_nodes; rank++){
    for (int adim = 0; adim < n_lhs_dims; adim++){
      free_csd(lhs_owner_ref_csd[rank][adim]);
    }
    for (int adim = 0; adim < n_rhs_dims; adim++){
      free_csd(rhs_owner_ref_csd[rank][adim]);
    }
  }

  if (mode == _XMP_N_GMOVE_NORMAL){

    //
    // Allocate buffers
    //

    // send buffer

    int sendcounts[n_gmv_nodes];
    int sendbuf_size = get_commbuf_size(send_comm_set, n_rhs_dims, sendcounts);
    void *sendbuf = _XMP_alloc(sendbuf_size * rhs_array->type_size);

    //xmp_dbg_printf("alloc_send = %d * %d\n", (int)sendbuf_size, (int)rhs_array->type_size);

    int sdispls[n_gmv_nodes];
    sdispls[0] = 0;
    for (int i = 1; i < n_gmv_nodes; i++){
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
    }

    // recv buffer

    int recvcounts[n_gmv_nodes];
    int recvbuf_size = get_commbuf_size(recv_comm_set, n_lhs_dims, recvcounts);
    void *recvbuf = _XMP_alloc(recvbuf_size * lhs_array->type_size);

    //xmp_dbg_printf("alloc_recv = %d * %d\n", (int)recvbuf_size, (int)lhs_array->type_size);
    
    int rdispls[n_gmv_nodes];
    rdispls[0] = 0;
    for (int i = 1; i < n_gmv_nodes; i++){
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }

    //
    // Packing
    //

    (*_XMP_pack_comm_set)(sendbuf, sendbuf_size, rhs_array, send_comm_set);

    //
    // communication
    //

#ifdef _XMP_MPI3
    if (xmp_is_async()){

      _XMP_async_comm_t *async = _XMP_get_current_async();

      MPI_Ialltoallv(sendbuf, sendcounts, sdispls, rhs_array->mpi_type,
		     recvbuf, recvcounts, rdispls, lhs_array->mpi_type,
		     *gmv_comm, &async->reqs[async->nreqs]);

      async->nreqs++;

      _XMP_async_gmove_t *gmove = _XMP_alloc(sizeof(_XMP_async_gmove_t));
      gmove->mode = mode;
      gmove->sendbuf = sendbuf;
      gmove->recvbuf = recvbuf;
      gmove->recvbuf_size = recvbuf_size;
      gmove->a = lhs_array;
      gmove->comm_set = recv_comm_set;
      async->gmove = gmove;

      for (int rank = 0; rank < n_gmv_nodes; rank++){
	for (int adim = 0; adim < n_rhs_dims; adim++){
	  free_comm_set(send_comm_set[rank][adim]);
	}
      }

      return;
    }
#endif

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, rhs_array->mpi_type,
    		  recvbuf, recvcounts, rdispls, lhs_array->mpi_type,
    		  *gmv_comm);

    /* int tag = 0; */
    /* int i = 0; */
    /* MPI_Request reqs[n_gmv_nodes * 2]; */

    /* for (int rank = 0; rank < n_gmv_nodes; rank++){ */

    /*   if (sendcounts[rank]){ */
    /* 	MPI_Isend((char*)sendbuf + sdispls[rank] * rhs_array->type_size, */
    /* 		  sendcounts[rank], rhs_array->mpi_type, rank, tag, *gmv_comm, &reqs[i++]); */
    /*   } */
    /*   if (recvcounts[rank]){ */
    /* 	MPI_Irecv((char*)recvbuf + rdispls[rank] * rhs_array->type_size, */
    /* 		  recvcounts[rank], lhs_array->mpi_type, rank, tag, *gmv_comm, &reqs[i++]); */
    /*   } */
    /* } */

    /* MPI_Waitall(i, reqs, MPI_STATUS_IGNORE); */

    //
    // Unpack
    //

    (*_XMP_unpack_comm_set)(recvbuf, recvbuf_size, lhs_array, recv_comm_set);

    //
    // Deallocate temporarls
    //

    _XMP_free(sendbuf);
    _XMP_free(recvbuf);

  }
#ifdef _XMP_MPI3_ONESIDED
  else if (mode == _XMP_N_GMOVE_IN){

    _XMP_gmove_inout(gmv_desc_leftp, gmv_desc_rightp, recv_comm_set, _XMP_N_COARRAY_GET);

    if (xmp_is_async()){
      _XMP_async_comm_t *async = _XMP_get_current_async();
      async->nreqs++;
      _XMP_async_gmove_t *gmove = _XMP_alloc(sizeof(_XMP_async_gmove_t));
      gmove->mode = mode;
      gmove->sendbuf = _XMP_get_execution_nodes()->comm; // NOTE: the sendbuf field is used for an improper purpose.
      async->gmove = gmove;
    }
    else {
      _XMP_sync_images_EXEC(NULL);
    }

  }
  else { // mode == _XMP_N_GMOVE_OUT

    if (!rhs_is_scalar)
      _XMP_gmove_inout(gmv_desc_rightp, gmv_desc_leftp, send_comm_set, _XMP_N_COARRAY_PUT);
    else
      _XMP_gmove_inout(gmv_desc_rightp, gmv_desc_leftp, recv_comm_set, _XMP_N_COARRAY_PUT);

    if (xmp_is_async()){
      _XMP_async_comm_t *async = _XMP_get_current_async();
      async->nreqs++;
      _XMP_async_gmove_t *gmove = _XMP_alloc(sizeof(_XMP_async_gmove_t));
      gmove->mode = mode;
      gmove->sendbuf = _XMP_get_execution_nodes()->comm; // NOTE: the sendbuf field is used for an improper purpose.
      async->gmove = gmove;
    }
    else {
      _XMP_sync_images_EXEC(NULL);
    }

  }
#else
  else {
    _XMP_fatal("Not supported gmove in/out on non-MPI3 environments");
  }
#endif

  if (mode == _XMP_N_GMOVE_NORMAL || (mode == _XMP_N_GMOVE_OUT && !rhs_is_scalar)){
    for (int rank = 0; rank < n_gmv_nodes; rank++){
      for (int adim = 0; adim < n_rhs_dims; adim++){
	free_comm_set(send_comm_set[rank][adim]);
      }
    }
  }

  if (mode == _XMP_N_GMOVE_NORMAL || mode == _XMP_N_GMOVE_IN || (mode == _XMP_N_GMOVE_OUT && rhs_is_scalar)){
    for (int rank = 0; rank < n_gmv_nodes; rank++){
      for (int adim = 0; adim < n_lhs_dims; adim++){
	free_comm_set(recv_comm_set[rank][adim]);
      }
    }
    _XMP_free(recv_comm_set);
    _XMP_free(_alloc_size);
  }

}


#ifdef _XMP_MPI3_ONESIDED

typedef struct _XMP_gmv_section_type {
  int ndims;
  long lb[_XMP_N_MAX_DIM];
  long len[_XMP_N_MAX_DIM];
  long st[_XMP_N_MAX_DIM];
} _XMP_gmv_section_t;

typedef struct _XMP_gmv_inout_list_type {
  _XMP_gmv_section_t org;
  _XMP_gmv_section_t tgt;
  struct _XMP_gmv_inout_list_type *next;
} _XMP_gmv_inout_list_t;


static void _XMP_set_comm_list(_XMP_array_t *a, _XMP_gmv_section_t *sec,
			       int gidx[], long len[], int st[]){

  int lidx[_XMP_N_MAX_DIM];

  int ndims = a->dim;

  xmp_array_gtol(a, gidx, lidx);
  sec->ndims = ndims;

  if (a->order == MPI_ORDER_FORTRAN){
    for (int i = 0; i < ndims; i++){
      sec->lb[i] = lidx[ndims - 1 - i];
      sec->len[i] = len[ndims - 1 - i];
      sec->st[i] = st[ndims - 1 - i];
    }
  }
  else {
    for (int i = 0; i < ndims; i++){
      sec->lb[i] = lidx[i];
      sec->len[i] = len[i];
      sec->st[i] = st[i];
    }
  }

}


#define SET_I_AND_LEN(k) \
  { if (!org_is_scalar){ \
      org_i[k] = c[k]->l; \
      org_len[k] = c[k]->u - c[k]->l + 1; \
      org_stride[k] = 1; \
      if (org_st[k] != 0 && tgt_dim[k] != -1){ \
	tgt_i[tgt_dim[k]] = (org_i[k] - org_lb[k]) \
                          * tgt_st[tgt_dim[k]] / org_st[k] + tgt_lb[tgt_dim[k]]; \
	tgt_len[tgt_dim[k]] = c[k]->u - c[k]->l + 1; \
        tgt_stride[tgt_dim[k]] = tgt_st[tgt_dim[k]]; \
      } \
    } \
    else if (!tgt_is_scalar){ \
      tgt_i[k] = c[k]->l; \
      tgt_len[k] = c[k]->u - c[k]->l + 1; \
      tgt_stride[k] = tgt_st[k]; \
    } \
  }

/* #define SET_I_AND_LEN(k) \ */
/*   { if (!org_is_scalar){ \ */
/*       org_i[k] = c[k]->l; \ */
/*       org_len[k] = c[k]->u - c[k]->l + 1; \ */
/*       org_stride[k] = 1; \ */
/*       if (org_st[k] != 0 && tgt_dim[k] != -1){ \ */
/* 	tgt_i[tgt_dim[k]] = (org_i[k] - org_lb[k]) \ */
/*                           * tgt_st[tgt_dim[k]] / org_st[k] + tgt_lb[tgt_dim[k]]; \ */
/* 	tgt_len[tgt_dim[k]] = c[k]->u - c[k]->l + 1; \ */
/*         tgt_stride[tgt_dim[k]] = tgt_st[tgt_dim[k]]; \ */
/*       } \ */
/*     } \ */
/*   } */


static void _XMP_conv_comm_set_to_list(_XMP_gmv_desc_t *gmv_desc_org,
				       _XMP_gmv_desc_t *gmv_desc_tgt,
				       _XMP_comm_set_t *comm_set[],
				       _XMP_gmv_inout_list_t *gmv_inout_listh){

  _XMP_array_t *org_array = gmv_desc_org->a_desc;
  int *org_lb = gmv_desc_org->lb;
  int *org_st = gmv_desc_org->st;
  int org_ndims = gmv_desc_org->ndims;

  _XMP_array_t *tgt_array = gmv_desc_tgt->a_desc;
  int *tgt_lb = gmv_desc_tgt->lb;
  int *tgt_st = gmv_desc_tgt->st;
  int tgt_ndims = gmv_desc_tgt->ndims;

  _XMP_comm_set_t *c[org_ndims];
  int org_i[org_ndims], tgt_i[tgt_ndims];
  long org_len[org_ndims], tgt_len[tgt_ndims];
  int org_stride[org_ndims], tgt_stride[tgt_ndims];
  int tgt_dim[org_ndims];

  int org_is_scalar = 1;
  for (int i = 0; i < org_ndims; i++) org_is_scalar &= (org_st[i] == 0);

  int tgt_is_scalar = 1;
  for (int i = 0; i < tgt_ndims; i++) tgt_is_scalar &= (tgt_st[i] == 0);

  _XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh;

  if (!org_is_scalar){

    int j = 0;
    for (int i = 0; i < org_ndims; i++){
      if (org_st[i] != 0){
	while (tgt_st[j] == 0 && j < tgt_ndims){
	  tgt_i[j] = tgt_lb[j];
	  tgt_len[j] = 1;
	  j++;
	}
	if (j < tgt_ndims) tgt_dim[i] = j++;
	else tgt_dim[i] = -1;
      }
    }

    for (; j < tgt_ndims; j++){
      tgt_i[j] = tgt_lb[j];
      tgt_len[j] = 1;
    }

  }
  else { // org_is_scalar

    for (int i = 0; i < org_ndims; i++){
      org_i[i] = org_lb[i];
      org_len[i] = 1;
      org_stride[i] = 1;
      tgt_dim[i] = -1;
    }

    if (tgt_is_scalar){
      for (int i = 0; i < tgt_ndims; i++){
	tgt_i[i] = tgt_lb[i];
	tgt_len[i] = 1;
	tgt_stride[i] = 1;
      }
    }

  }

  switch (org_ndims){

  case 1:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }

    break;

  case 2:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}

    break;

  case 3:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN(2);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}

    break;

  case 4:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN(3);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}

    break;

  case 5:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN(4);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}

    break;

  case 6:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN(4);
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      SET_I_AND_LEN(5);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}}

    break;

  case 7:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN(4);
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      SET_I_AND_LEN(5);
    for (c[6] = comm_set[6]; c[6]; c[6] = c[6]->next){
      SET_I_AND_LEN(6);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(org_array, &t->org, org_i, org_len, org_stride);
      _XMP_set_comm_list(tgt_array, &t->tgt, tgt_i, tgt_len, tgt_stride);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}}}

    break;

  default:
    _XMP_fatal("wrong array dimension");

  }

}


#define SET_I_AND_LEN_2(k) \
  { i[k] = c[k]->l; \
    len[k] = c[k]->u - c[k]->l + 1; \
    st[k] = 1; \
  }


static void _XMP_conv_comm_set_to_list_2(_XMP_array_t *a,
				       _XMP_comm_set_t *comm_set[],
				       _XMP_gmv_inout_list_t *gmv_inout_listh){

  int ndims = a->dim;

  _XMP_comm_set_t *c[ndims];
  int i[ndims];
  long len[ndims];
  int st[ndims];

  _XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh;

  switch (ndims){

  case 1:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }

    break;

  case 2:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}

    break;

  case 3:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN_2(2);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}

    break;

  case 4:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN_2(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN_2(3);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}

    break;

  case 5:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN_2(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN_2(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN_2(4);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}

    break;

  case 6:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN_2(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN_2(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN_2(4);
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      SET_I_AND_LEN_2(5);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}}

    break;

  case 7:

    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      SET_I_AND_LEN_2(0);
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      SET_I_AND_LEN_2(1);
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      SET_I_AND_LEN_2(2);
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      SET_I_AND_LEN_2(3);
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      SET_I_AND_LEN_2(4);
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      SET_I_AND_LEN_2(5);
    for (c[6] = comm_set[6]; c[6]; c[6] = c[6]->next){
      SET_I_AND_LEN_2(6);

      _XMP_gmv_inout_list_t *t = _XMP_alloc(sizeof(_XMP_gmv_inout_list_t));
      t->next = NULL;

      _XMP_set_comm_list(a, &t->tgt, i, len, st);

      gmv_inout_list->next = t;
      gmv_inout_list = t;

    }}}}}}}

    break;

  default:
    _XMP_fatal("wrong array dimension");

  }

}


#if XMP_DBG
static void print_gmv_inout_list(_XMP_gmv_inout_list_t *gmv_inout_listh){
  for (_XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh;
       gmv_inout_list; gmv_inout_list = gmv_inout_list->next){

    _XMP_gmv_section_t *org = &gmv_inout_list->org;

    printf(" ORG: (");
    for (int i = 0; i < org->ndims; i++){
      printf("%ld:%ld:%ld, ", org->lb[i], org->len[i], org->st[i]);
    } 
    printf(")\n");

    _XMP_gmv_section_t *tgt = &gmv_inout_list->tgt;
    
    printf(" TGT: (");
    for (int i = 0; i < tgt->ndims; i++){
      printf("%ld:%ld:%ld, ", tgt->lb[i], tgt->len[i], tgt->st[i]);
    }
    printf(")\n");

  }

}
#endif


static void _XMP_gmove_inout(_XMP_gmv_desc_t *gmv_desc_org, _XMP_gmv_desc_t *gmv_desc_tgt,
			     _XMP_comm_set_t *comm_set[][_XMP_N_MAX_DIM],
			     int rdma_type){

  _XMP_array_t *org_array = gmv_desc_org->a_desc;
  int org_ndims = gmv_desc_org->ndims;
  _XMP_coarray_t *org_coarray = org_array->coarray;
  void *org_addr = org_array->array_addr_p;

  _XMP_array_t *tgt_array = gmv_desc_tgt->a_desc;
  int tgt_ndims = gmv_desc_tgt->ndims;
  _XMP_coarray_t *tgt_coarray = tgt_array->coarray;

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;
#endif

  int ndims_gmv_nodes = gmv_nodes->dim;

  long elmts[org_ndims];
  long distance[org_ndims];

  if (org_array->order == MPI_ORDER_FORTRAN){
    for (int i = 0; i < org_ndims; i++){
      elmts[i] = org_array->info[org_ndims - 1 - i].alloc_size;
      distance[i] = org_array->info[org_ndims - 1 - i].dim_acc * org_array->type_size;
    }
  }
  else {
    for (int i = 0; i < org_ndims; i++){
      elmts[i] = org_array->info[i].alloc_size;
      distance[i] = org_array->info[i].dim_acc * org_array->type_size;
    }
  }

  for (int tgt_node = 0; tgt_node < n_gmv_nodes; tgt_node++){

    _XMP_gmv_inout_list_t gmv_inout_listh;
    gmv_inout_listh.next = NULL;
    _XMP_conv_comm_set_to_list(gmv_desc_org, gmv_desc_tgt,
			       comm_set[tgt_node], &gmv_inout_listh);

#if XMP_DBG
    if (myrank == DBG_RANK){
      if (rdma_type == _XMP_N_COARRAY_GET) printf("[%d gets from %d]\n", myrank, tgt_node);
      else printf("[%d puts to %d]\n", myrank, tgt_node);
      print_gmv_inout_list(gmv_inout_listh.next);
      fflush(stdout);
    }
#endif

    long coarray_elmts[tgt_ndims], coarray_distance[tgt_ndims];
    unsigned long long total_elmts = tgt_array->type_size;

    if (tgt_array->order == MPI_ORDER_FORTRAN){
      for (int i = tgt_ndims - 1; i >= 0; i--){
	coarray_elmts[i] = _alloc_size[tgt_node][tgt_ndims - 1 - i];
	coarray_distance[i] = total_elmts;
	total_elmts *= coarray_elmts[i];
      }
    }
    else {
      for (int i = tgt_ndims - 1; i >= 0; i--){
	coarray_elmts[i] = _alloc_size[tgt_node][i];
	coarray_distance[i] = total_elmts;
	total_elmts *= coarray_elmts[i];
      }
    }

    for (_XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh.next;
	 gmv_inout_list; gmv_inout_list = gmv_inout_list->next){

      int ncoord[_XMP_N_MAX_DIM];
      _XMP_calc_rank_array(gmv_nodes, ncoord, tgt_node);
      for (int i = 0; i < ndims_gmv_nodes; i++) ncoord[i]++; // to one-based.
      _XMP_coarray_rdma_image_set_n(ndims_gmv_nodes, ncoord);

      long *org_lbound = gmv_inout_list->org.lb;
      long *org_length = gmv_inout_list->org.len;
      long *org_stride = gmv_inout_list->org.st;
      long *tgt_lbound = gmv_inout_list->tgt.lb;
      long *tgt_length = gmv_inout_list->tgt.len;
      long *tgt_stride = gmv_inout_list->tgt.st;

      // set org (local array)
      _XMP_coarray_rdma_array_set_n(org_ndims, org_lbound, org_length, org_stride, elmts, distance);

      // set tgt (coarray)
      _XMP_coarray_rdma_coarray_set_n(tgt_ndims, tgt_lbound, tgt_length, tgt_stride);

      // do comms.
      _XMP_coarray_rdma_do2(rdma_type, tgt_coarray, org_addr, org_coarray,
      			    coarray_elmts, coarray_distance);

    }

    // free gmv_inout_list
    _XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh.next;
    while (gmv_inout_list){
      _XMP_gmv_inout_list_t *next = gmv_inout_list->next;
      _XMP_free(gmv_inout_list);
      gmv_inout_list = next;
    }

  }

}


void _XMP_gmove_inout_scalar(void *scalar, _XMP_gmv_desc_t *gmv_desc, int rdma_type){

  _XMP_array_t *a = gmv_desc->a_desc;
  int ndims = gmv_desc->ndims;
  int *lb = gmv_desc->lb;
  int *ub = gmv_desc->ub;
  int *st = gmv_desc->st;
  _XMP_coarray_t *coarray = a->coarray;

  gmv_nodes = _XMP_world_nodes;
  n_gmv_nodes = gmv_nodes->comm_size;
  _alloc_size = _XMP_alloc(sizeof(int) * n_gmv_nodes *_XMP_N_MAX_DIM);

#if XMP_DBG
  int myrank = gmv_nodes->comm_rank;
#endif

  int ndims_gmv_nodes = gmv_nodes->dim;

  _XMP_csd_t *owner_ref_csd[n_gmv_nodes][_XMP_N_MAX_DIM];
  get_owner_ref_csd(a, lb, ub, st, owner_ref_csd, 1);

  _XMP_comm_set_t *comm_set[n_gmv_nodes][_XMP_N_MAX_DIM];
  for (int i = 0; i < n_gmv_nodes; i++){
    for (int j = 0; j < ndims; j++){
      comm_set[i][j] = csd2comm_set(owner_ref_csd[i][j]);
    }
  }

  for (int tgt_node = 0; tgt_node < n_gmv_nodes; tgt_node++){

    _XMP_gmv_inout_list_t gmv_inout_listh;
    gmv_inout_listh.next = NULL;

    if (rdma_type == _XMP_N_COARRAY_GET){ // s = g(i)[j];
      _XMP_conv_comm_set_to_list_2(a, comm_set[tgt_node], &gmv_inout_listh);
    }
    else { // g(i)[j] = s
      _XMP_conv_comm_set_to_list_2(a, comm_set[tgt_node], &gmv_inout_listh);
    }

#if XMP_DBG
    if (myrank == DBG_RANK){
      if (rdma_type == _XMP_N_COARRAY_GET) printf("[%d gets from %d]\n", myrank, tgt_node);
      else printf("[%d puts to %d]\n", myrank, tgt_node);
      print_gmv_inout_list(gmv_inout_listh.next);
      fflush(stdout);
    }
#endif

    long coarray_elmts[ndims], coarray_distance[ndims];
    unsigned long long total_elmts = a->type_size;

    if (a->order == MPI_ORDER_FORTRAN){
      for (int i = ndims - 1; i >= 0; i--){
	coarray_elmts[i] = _alloc_size[tgt_node][ndims - 1 - i];
	coarray_distance[i] = total_elmts;
	total_elmts *= coarray_elmts[i];
      }
    }
    else {
      for (int i = ndims - 1; i >= 0; i--){
	coarray_elmts[i] = _alloc_size[tgt_node][i];
	coarray_distance[i] = total_elmts;
	total_elmts *= coarray_elmts[i];
      }
    }

    for (_XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh.next;
	 gmv_inout_list; gmv_inout_list = gmv_inout_list->next){

      int ncoord[_XMP_N_MAX_DIM];
      _XMP_calc_rank_array(gmv_nodes, ncoord, tgt_node);
      for (int i = 0; i < ndims_gmv_nodes; i++) ncoord[i]++; // to one-based.
      _XMP_coarray_rdma_image_set_n(ndims_gmv_nodes, ncoord);

      long *tgt_lbound = gmv_inout_list->tgt.lb;
      long *tgt_length = gmv_inout_list->tgt.len;
      long *tgt_stride = gmv_inout_list->tgt.st;

      // set org (scalar)
      _XMP_coarray_rdma_array_set_1(0, 1, 1, 1, a->type_size);

      // set tgt (coarray)
      _XMP_coarray_rdma_coarray_set_n(ndims, tgt_lbound, tgt_length, tgt_stride);

      // do comms.
      _XMP_coarray_rdma_do2(rdma_type, coarray, scalar, NULL,
      			    coarray_elmts, coarray_distance);

    }

    // free gmv_inout_list
    _XMP_gmv_inout_list_t *gmv_inout_list = gmv_inout_listh.next;
    while (gmv_inout_list){
      _XMP_gmv_inout_list_t *next = gmv_inout_list->next;
      _XMP_free(gmv_inout_list);
      gmv_inout_list = next;
    }

  }

  _XMP_free(_alloc_size);

}

#endif

void _XMP_copy_scalar_array(char *scalar, _XMP_array_t *a, _XMP_comm_set_t *comm_set[]){

  int ndims = a->dim;

  char *dst = (char *)a->array_addr_p;

  _XMP_comm_set_t *c[ndims];

  int i[_XMP_N_MAX_DIM];

  switch (ndims){

  case 1:
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    break;

  case 2:
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    break;

  case 3:
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    }}
    break;

  case 4:
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    }}
    }}
    break;

  case 5:
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    }}
    }}
    }}
    break;

  case 6:
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    }}
    }}
    }}
    }}
    break;

  case 7:
    for (c[6] = comm_set[6]; c[6]; c[6] = c[6]->next){
      for (i[6] = c[6]->l; i[6] <= c[6]->u; i[6]++){
    for (c[5] = comm_set[5]; c[5]; c[5] = c[5]->next){
      for (i[5] = c[5]->l; i[5] <= c[5]->u; i[5]++){
    for (c[4] = comm_set[4]; c[4]; c[4] = c[4]->next){
      for (i[4] = c[4]->l; i[4] <= c[4]->u; i[4]++){
    for (c[3] = comm_set[3]; c[3]; c[3] = c[3]->next){
      for (i[3] = c[3]->l; i[3] <= c[3]->u; i[3]++){
    for (c[2] = comm_set[2]; c[2]; c[2] = c[2]->next){
      for (i[2] = c[2]->l; i[2] <= c[2]->u; i[2]++){
    for (c[1] = comm_set[1]; c[1]; c[1] = c[1]->next){
      for (i[1] = c[1]->l; i[1] <= c[1]->u; i[1]++){
    for (c[0] = comm_set[0]; c[0]; c[0] = c[0]->next){
      for (i[0] = c[0]->l; i[0] <= c[0]->u; i[0]++){
	memcpy(dst + _XMP_gtol_calc_offset(a, i), scalar, a->type_size);
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    break;

  default:
    _XMP_fatal("wrong array dimension");
  }

}


void _XMP_gmove_gsection_scalar(_XMP_array_t *lhs_array, int *lhs_lb, int *lhs_ub, int *lhs_st, char *scalar){

  int n_lhs_dims = lhs_array->dim;;

  _XMP_nodes_t *nodes = lhs_array->align_template->onto_nodes;
  int ncoord[nodes->dim];
  for (int i = 0; i < nodes->dim; i++){
    ncoord[i] = nodes->info[i].rank;
  }

  _XMP_csd_t *csd_ref[n_lhs_dims];
  _XMP_csd_t *owner_csd[n_lhs_dims];
  _XMP_csd_t *owner_ref_csd[n_lhs_dims];
  _XMP_comm_set_t *comm_set[n_lhs_dims];

  for (int i = 0; i < n_lhs_dims; i++){

    _XMP_rsd_t rsd_ref;

    if (lhs_st[i] != 0){
      rsd_ref.l = lhs_lb[i];
      rsd_ref.u = lhs_ub[i];
      rsd_ref.s = lhs_st[i];
    }
    else {
      rsd_ref.l = lhs_lb[i];
      rsd_ref.u = lhs_lb[i];
      rsd_ref.s = 1;
    }

    csd_ref[i] = rsd2csd(&rsd_ref);
    owner_csd[i] = get_owner_csd(lhs_array, i, ncoord);
    owner_ref_csd[i] = intersection_csds(owner_csd[i], csd_ref[i]);

  }

  reduce_csd(owner_ref_csd, n_lhs_dims);

  for (int i = 0; i < n_lhs_dims; i++){
    comm_set[i] = csd2comm_set(owner_ref_csd[i]);
  }

  _XMP_copy_scalar_array(scalar, lhs_array, comm_set);

  for (int i = 0; i < n_lhs_dims; i++){
    free_csd(csd_ref[i]);
    free_csd(owner_csd[i]);
    free_csd(owner_ref_csd[i]);
    free_comm_set(comm_set[i]);
  }

}


void _XMP_gmove_GSECTION_GSCALAR(_XMP_array_t *dst_array, _XMP_array_t *src_array,
				 int type, size_t type_size, ...){

  _XMP_gmv_desc_t gmv_desc_leftp;

  va_list args;
  va_start(args, type_size);

  // get dst info
  //unsigned long long dst_total_elmts = 1;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    //if (dst_s[i] != 0) dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  int src_dim = src_array->dim;;
  int src_ref_index[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_ref_index[i] = va_arg(args, int);
  }

  int mode = va_arg(args, int);

  va_end(args);

  if (mode == _XMP_N_GMOVE_NORMAL){
    char *tmp = _XMP_alloc(src_array->type_size);
    char *src_addr = (char *)src_array->array_addr_p + _XMP_gtol_calc_offset(src_array, src_ref_index);
    _XMP_gmove_BCAST_GSCALAR(tmp, src_addr, src_array, src_ref_index);

    _XMP_gmove_gsection_scalar(dst_array, dst_l, dst_u, dst_s, tmp);

    _XMP_free(tmp);
  }
  else {
    int dummy0[7] = { 0, 0, 0, 0, 0, 0, 0 }; /* temporarily assuming maximum 7-dimensional */
    int dummy1[7] = { 1, 1, 1, 1, 1, 1, 1 }; /* temporarily assuming maximum 7-dimensional */
    int dummy2[7] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

    _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

    gmv_desc_leftp.is_global = true;       gmv_desc_rightp.is_global = true;
    gmv_desc_leftp.ndims = dst_dim;        gmv_desc_rightp.ndims = src_dim;

    gmv_desc_leftp.a_desc = dst_array;     gmv_desc_rightp.a_desc = src_array;

    gmv_desc_leftp.local_data = NULL;      gmv_desc_rightp.local_data = NULL;
    gmv_desc_leftp.a_lb = NULL;            gmv_desc_rightp.a_lb = NULL;
    gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

    gmv_desc_leftp.kind = dummy2;          gmv_desc_rightp.kind = dummy1; // always index
    gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_ref_index;
    gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_ref_index;
    gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = dummy0;

    unsigned long long src_d[src_dim];
    for (int i = 0; i < src_dim; i++) {
      src_d[i] = src_array->info[i].dim_acc;
    }

    /* unsigned long long dst_d[dst_dim]; */
    /* for (int i = 0; i < dst_dim; i++) { */
    /*   dst_d[i] = dst_array->info[i].dim_acc; */
    /* } */

    _XMP_pack_comm_set = _XMPC_pack_comm_set;
    _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

    _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
    				  dst_l, dst_u, dst_s, dst_d,
    				  src_ref_index, src_ref_index, dummy0, src_d,
    				  mode);
  }
}


void _XMP_gmove_lsection_scalar(char *dst, int ndims, int *lb, int *ub, int *st, unsigned long long *d,
				char *scalar, size_t type_size){

  unsigned long long i[ndims];

  switch (ndims){

  case 1:
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * i[0], scalar, type_size);
    }
    break;

  case 2:
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1]), scalar, type_size);
    }}
    break;

  case 3:
    for (i[2] = lb[2] * d[2]; i[2] <= ub[2] * d[2]; i[2] += st[2] * d[2]){
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1] + i[2]), scalar, type_size);
    }}}
    break;

  case 4:
    for (i[3] = lb[3] * d[3]; i[3] <= ub[3] * d[3]; i[3] += st[3] * d[3]){
    for (i[2] = lb[2] * d[2]; i[2] <= ub[2] * d[2]; i[2] += st[2] * d[2]){
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1] + i[2] + i[3]), scalar, type_size);
    }}}}
    break;

  case 5:
    for (i[4] = lb[4] * d[4]; i[4] <= ub[4] * d[4]; i[4] += st[4] * d[4]){
    for (i[3] = lb[3] * d[3]; i[3] <= ub[3] * d[3]; i[3] += st[3] * d[3]){
    for (i[2] = lb[2] * d[2]; i[2] <= ub[2] * d[2]; i[2] += st[2] * d[2]){
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1] + i[2] + i[3] + i[4]), scalar, type_size);
    }}}}}
    break;

  case 6:
    for (i[5] = lb[5] * d[5]; i[5] <= ub[5] * d[5]; i[5] += st[5] * d[5]){
    for (i[4] = lb[4] * d[4]; i[4] <= ub[4] * d[4]; i[4] += st[4] * d[4]){
    for (i[3] = lb[3] * d[3]; i[3] <= ub[3] * d[3]; i[3] += st[3] * d[3]){
    for (i[2] = lb[2] * d[2]; i[2] <= ub[2] * d[2]; i[2] += st[2] * d[2]){
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1] + i[2] + i[3] + i[4] + i[5]), scalar, type_size);
    }}}}}}
    break;

  case 7:
    for (i[6] = lb[6] * d[6]; i[6] <= ub[6] * d[6]; i[6] += st[6] * d[6]){
    for (i[5] = lb[5] * d[5]; i[5] <= ub[5] * d[5]; i[5] += st[5] * d[5]){
    for (i[4] = lb[4] * d[4]; i[4] <= ub[4] * d[4]; i[4] += st[4] * d[4]){
    for (i[3] = lb[3] * d[3]; i[3] <= ub[3] * d[3]; i[3] += st[3] * d[3]){
    for (i[2] = lb[2] * d[2]; i[2] <= ub[2] * d[2]; i[2] += st[2] * d[2]){
    for (i[1] = lb[1] * d[1]; i[1] <= ub[1] * d[1]; i[1] += st[1] * d[1]){
    for (i[0] = lb[0] * d[0]; i[0] <= ub[0] * d[0]; i[0] += st[0] * d[0]){
      memcpy(dst + type_size * (i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6]), scalar, type_size);
    }}}}}}}
    break;

  default:
    _XMP_fatal("wrong array dimension");
  }

}


void _XMP_gmove_LSECTION_GSCALAR(_XMP_array_t *src_array, int type, size_t type_size,
				 char *dst, int dst_dim, ...){

  va_list args;
  va_start(args, dst_dim);

  // get dst info
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
  }

  // get src info
  int src_dim = src_array->dim;;
  int src_ref_index[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_ref_index[i] = va_arg(args, int);
  }

  int mode = va_arg(args, int);

  va_end(args);

  if (mode == _XMP_N_GMOVE_NORMAL){
    char *tmp = _XMP_alloc(src_array->type_size);
    char *src_addr = (char *)src_array->array_addr_p + _XMP_gtol_calc_offset(src_array, src_ref_index);
    _XMP_gmove_BCAST_GSCALAR(tmp, src_addr, src_array, src_ref_index);

    _XMP_gmove_lsection_scalar(dst, dst_dim, dst_l, dst_u, dst_s, dst_d, tmp, type_size);

    _XMP_free(tmp);
  }
  else if (mode == _XMP_N_GMOVE_IN){
#ifdef _XMP_MPI3_ONESIDED
    
    // create a temporal descriptor for the "non-distributed" LHS array (to be possibly used
    // in _XMP_gmove_1to1)
    _XMP_array_t *a;
    _XMP_init_array_desc_NOT_ALIGNED(&a, src_array->align_template, dst_dim,
				     src_array->type, src_array->type_size, dst_d, dst);

    int dummy0[_XMP_N_MAX_DIM] = { 0, 0, 0, 0, 0, 0, 0 }; /* temporarily assuming maximum 7-dimensional */
    int dummy1[_XMP_N_MAX_DIM] = { 1, 1, 1, 1, 1, 1, 1 }; /* temporarily assuming maximum 7-dimensional */
    int dummy2[_XMP_N_MAX_DIM] = { 2, 2, 2, 2, 2, 2, 2 }; /* temporarily assuming maximum 7-dimensional */

    _XMP_gmv_desc_t gmv_desc_leftp, gmv_desc_rightp;

    gmv_desc_leftp.is_global = false;      gmv_desc_rightp.is_global = true;
    gmv_desc_leftp.ndims = dst_dim;        gmv_desc_rightp.ndims = src_dim;

    gmv_desc_leftp.a_desc = a;             gmv_desc_rightp.a_desc = src_array;

    gmv_desc_leftp.local_data = dst;       gmv_desc_rightp.local_data = NULL;
    gmv_desc_leftp.a_lb = dummy0;          gmv_desc_rightp.a_lb = NULL;
    gmv_desc_leftp.a_ub = NULL;            gmv_desc_rightp.a_ub = NULL;

    gmv_desc_leftp.kind = dummy2;          gmv_desc_rightp.kind = dummy1; // always index
    gmv_desc_leftp.lb = dst_l;             gmv_desc_rightp.lb = src_ref_index;
    gmv_desc_leftp.ub = dst_u;             gmv_desc_rightp.ub = src_ref_index;
    gmv_desc_leftp.st = dst_s;             gmv_desc_rightp.st = dummy0;

    _XMP_ASSERT(gmv_desc_rightp->a_desc);

    unsigned long long src_d[src_dim];
    for (int i = 0; i < src_dim; i++) {
      src_d[i] = src_array->info[i].dim_acc;
    }

    _XMP_pack_comm_set = _XMPC_pack_comm_set;
    _XMP_unpack_comm_set = _XMPC_unpack_comm_set;

    _XMP_gmove_array_array_common(&gmv_desc_leftp, &gmv_desc_rightp,
				  dst_l, dst_u, dst_s, dst_d,
				  src_ref_index, src_ref_index, dummy0, src_d,
				  mode);

    _XMP_finalize_array_desc(a);
#else
    _XMP_fatal("Not supported gmove in/out on non-MPI3 environments");
#endif
  }
  else {
    _XMP_fatal("_XMP_gmove_LSECTION_SCALAR: wrong gmove mode");
  }

}


void _XMP_gmove_INOUT_SCALAR(_XMP_array_t *dst_array, void *scalar, ...){

  _XMP_gmv_desc_t gmv_desc_leftp;

  va_list args;
  va_start(args, scalar);

  // get dst info
  //unsigned long long dst_total_elmts = 1;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = va_arg(args, int);
    int size = va_arg(args, int);
    dst_s[i] = va_arg(args, int);
    dst_u[i] = dst_l[i] + (size - 1) * dst_s[i];
    dst_d[i] = va_arg(args, unsigned long long);
    _XMP_normalize_array_section(&gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    //if (dst_s[i] != 0) dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  //int mode = va_arg(args, int);

  va_end(args);

  _XMP_ASSERT(mode != _XMP_N_GMOVE_NORMAL);

#ifdef _XMP_MPI3_ONESIDED
  _XMP_gmv_desc_t gmv_desc;
  int kind[_XMP_N_MAX_DIM];
  for (int i = 0; i < dst_dim; i++){
    kind[i] = (dst_s[i] == 0) ? XMP_N_GMOVE_INDEX : XMP_N_GMOVE_RANGE;
  }

  gmv_desc.is_global = true;
  gmv_desc.ndims = dst_dim;

  gmv_desc.a_desc = dst_array;

  gmv_desc.local_data = NULL;
  gmv_desc.a_lb = NULL;
  gmv_desc.a_ub = NULL;

  gmv_desc.kind = kind;
  gmv_desc.lb = dst_l;
  gmv_desc.ub = dst_u;
  gmv_desc.st = dst_s;

  _XMP_gmove_inout_scalar(scalar, &gmv_desc, _XMP_N_COARRAY_PUT);
#else
  _XMP_fatal("Not supported gmove in/out on non-MPI3 environments");
#endif

}
