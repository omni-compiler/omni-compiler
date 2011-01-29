/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include <string.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

static void _XMP_create_shadow_comm(_XMP_array_t *array, int array_index);

static void _XMP_create_shadow_comm(_XMP_array_t *array, int array_index) {
  _XMP_ASSERT(array != NULL);
  
  _XMP_nodes_t *onto_nodes = (array->align_template)->onto_nodes;
  if (!onto_nodes->is_member) {
    return;
  }

  _XMP_array_info_t *ai = &(array->info[array_index]);
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED); // checked by compiler
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_DUPLICATION); // checked by compiler

  _XMP_template_chunk_t *chunk = ai->align_template_chunk;
  _XMP_ASSERT(chunk->dist_manner != _XMP_N_DIST_DUPLICATION); // align_manner is not _XMP_N_ALIGN_DUPLICATION

  int onto_nodes_index = chunk->onto_nodes_index;

  int color = 1;
  int acc_nodes_size = 1;
  int nodes_dim = onto_nodes->dim;
  for (int i = 0; i < nodes_dim; i++) {
    _XMP_nodes_info_t *onto_nodes_info = &(onto_nodes->info[i]);
    int size = onto_nodes_info->size;
    int rank = onto_nodes_info->rank;

    if (i != onto_nodes_index) {
      color += (acc_nodes_size * rank);
    }

    acc_nodes_size *= size;
  }

  if (!array->is_allocated) {
    color = 0;
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*(onto_nodes->comm), color, onto_nodes->comm_rank, comm);

  // set members
  if (array->is_allocated) {
    ai->is_shadow_comm_member = true;

    ai->shadow_comm = comm;
    MPI_Comm_size(*comm, &(ai->shadow_comm_size));
    MPI_Comm_rank(*comm, &(ai->shadow_comm_rank));
  }
  else {
    _XMP_finalize_comm(comm);
  }
}

void _XMP_init_shadow(_XMP_array_t *array, ...) {
  _XMP_ASSERT(array != NULL);

  int dim = array->dim;
  va_list args;
  va_start(args, array);
  for (int i = 0; i < dim; i++) {
    _XMP_array_info_t *ai = &(array->info[i]);

    int type = va_arg(args, int);
    switch (type) {
      case _XMP_N_SHADOW_NONE:
        ai->shadow_type = _XMP_N_SHADOW_NONE;
        break;
      case _XMP_N_SHADOW_NORMAL:
        {
          int lo = va_arg(args, int);
          if (lo < 0) {
            _XMP_fatal("<shadow-width> should be a nonnegative integer");
          }

          int hi = va_arg(args, int);
          if (hi < 0) {
            _XMP_fatal("<shadow-width> should be a nonnegative integer");
          }

          if ((lo == 0) && (hi == 0)) {
            ai->shadow_type = _XMP_N_SHADOW_NONE;
          }
          else {
            ai->shadow_type = _XMP_N_SHADOW_NORMAL;
            ai->shadow_size_lo = lo;
            ai->shadow_size_hi = hi;

            if (array->is_allocated) {
              if (ai->align_manner == _XMP_N_ALIGN_BLOCK) {
                ai->local_lower += lo;
                ai->local_upper += lo;
             // ai->local_stride is not changed
                ai->alloc_size += lo + hi;

                *(ai->temp0) -= lo;
              }
              else {
                // FIXME implement for other dist manners
                _XMP_fatal("not implemented yet");
              }
            }

            _XMP_create_shadow_comm(array, i);
          }

          break;
        }
      case _XMP_N_SHADOW_FULL:
        {
          ai->shadow_type = _XMP_N_SHADOW_FULL;

          if (array->is_allocated) {
            ai->shadow_size_lo = ai->par_lower - ai->ser_lower;
            ai->shadow_size_hi = ai->ser_upper - ai->par_upper;

            ai->local_lower = ai->par_lower;
            ai->local_upper = ai->par_upper;
            ai->local_stride = ai->par_stride;
            ai->alloc_size = ai->ser_size;
          }

          _XMP_create_shadow_comm(array, i);
          break;
        }
      default:
        _XMP_fatal("unknown shadow type");
    }
  }
}

// FIXME consider full shadow in other dimensions
void _XMP_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr,
                             _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);

  if (!array_desc->is_allocated) {
    return;
  }

  int array_type = array_desc->type;
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // pack lo shadow
  if (rank != (size - 1)) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // alloc buffer
      *lo_buffer = _XMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts) * (array_desc->type_size));

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          // XXX shadow is allowed in BLOCK distribution
          lower[i] = array_desc->info[i].local_upper - array_desc->info[i].shadow_size_lo + 1;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_lo - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      // pack data
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(*lo_buffer, array_addr, array_desc->type_size,
                                       array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_pack_array_BASIC(*lo_buffer, array_addr, array_type, array_dim, lower, upper, stride, dim_acc);
      }
    }
  }

  // pack hi shadow
  if (rank != 0) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // alloc buffer
      *hi_buffer = _XMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts) * (array_desc->type_size));

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          // FIXME shadow is allowed in BLOCK distribution
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      // pack data
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(*hi_buffer, array_addr, array_desc->type_size,
                                       array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_pack_array_BASIC(*hi_buffer, array_addr, array_type, array_dim, lower, upper, stride, dim_acc);
      }
    }
  }
}

// FIXME not consider full shadow
void _XMP_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr,
                               _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(lo_buffer != NULL);
  _XMP_ASSERT(hi_buffer != NULL);
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);

  if (!array_desc->is_allocated) {
    return;
  }

  int array_type = array_desc->type;
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // unpack lo shadow
  if (rank != 0) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          // FIXME shadow is allowed in BLOCK distribution
          lower[i] = 0;
          upper[i] = array_desc->info[i].shadow_size_lo - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      // unpack data
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(array_addr, lo_buffer, array_desc->type_size,
                                         array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_unpack_array_BASIC(array_addr, lo_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
      }

      // free buffer
      _XMP_free(lo_buffer);
    }
  }

  // unpack hi shadow
  if (rank != (size - 1)) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          // FIXME shadow is allowed in BLOCK distribution
          lower[i] = array_desc->info[i].local_upper + 1;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      // unpack data
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(array_addr, hi_buffer, array_desc->type_size,
                                         array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_unpack_array_BASIC(array_addr, hi_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
      }

      // free buffer
      _XMP_free(hi_buffer);
    }
  }
}

// FIXME change tag
// FIXME not consider full shadow
void _XMP_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
                                 void *lo_send_buffer, void *hi_send_buffer,
                                 _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(lo_send_buffer != NULL);
  _XMP_ASSERT(hi_send_buffer != NULL);
  _XMP_ASSERT(array_desc != NULL);

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  // get communicator info
  MPI_Comm *comm = ai->shadow_comm;
  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(array_desc->type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // exchange shadow
  MPI_Request send_req[2];
  MPI_Request recv_req[2];

  if (ai->shadow_size_lo > 0) {
    if (rank != 0) {
      *lo_recv_buffer = _XMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts) * (array_desc->type_size));
      MPI_Irecv(*lo_recv_buffer, (ai->shadow_size_lo) * (ai->dim_elmts), mpi_datatype,
                rank - 1, _XMP_N_MPI_TAG_REFLECT_LO, *comm, &(recv_req[0]));
    }

    if (rank != (size - 1)) {
      MPI_Isend(lo_send_buffer, (ai->shadow_size_lo) * (ai->dim_elmts), mpi_datatype,
                rank + 1, _XMP_N_MPI_TAG_REFLECT_LO, *comm, &(send_req[0]));
    }
  }

  if (ai->shadow_size_hi > 0) {
    if (rank != (size - 1)) {
      *hi_recv_buffer = _XMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts) * (array_desc->type_size));
      MPI_Irecv(*hi_recv_buffer, (ai->shadow_size_hi) * (ai->dim_elmts), mpi_datatype,
                rank + 1, _XMP_N_MPI_TAG_REFLECT_HI, *comm, &(recv_req[1]));
    }

    if (rank != 0) {
      MPI_Isend(hi_send_buffer, (ai->shadow_size_hi) * (ai->dim_elmts), mpi_datatype,
                rank - 1, _XMP_N_MPI_TAG_REFLECT_HI, *comm, &(send_req[1]));
    }
  }

  // wait & free
  MPI_Status stat;

  if (ai->shadow_size_lo > 0) {
    if (rank != 0) {
      MPI_Wait(&(recv_req[0]), &stat);
    }

    if (rank != (size - 1)) {
      MPI_Wait(&(send_req[0]), &stat);
      _XMP_free(lo_send_buffer);
    }
  }

  if (ai->shadow_size_hi > 0) {
    if (rank != (size - 1)) {
      MPI_Wait(&(recv_req[1]), &stat);
    }

    if (rank != 0) {
      MPI_Wait(&(send_req[1]), &stat);
      _XMP_free(hi_send_buffer);
    }
  }
}

static void _XMP_reflect_shadow_FULL_ALLGATHER(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);
  _XMP_ASSERT(array_desc->is_allocated);
  _XMP_ASSERT(array_desc->dim == 1);

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  size_t type_size = array_desc->type_size;
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  int gather_count = ai->par_size;
  size_t gather_byte_size = type_size * gather_count;
  void *pack_buffer = _XMP_alloc(gather_byte_size);
  memcpy(pack_buffer, array_addr + (type_size * ai->local_lower), gather_byte_size);

  MPI_Allgather(pack_buffer, gather_count, mpi_datatype,
                array_addr, gather_count, mpi_datatype,
                *(ai->shadow_comm));

  _XMP_free(pack_buffer);
}

static void _XMP_reflect_shadow_FULL_ALLGATHERV(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);
  _XMP_ASSERT(array_desc->is_allocated);
  _XMP_ASSERT(array_desc->dim == 1);

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  size_t type_size = array_desc->type_size;
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  int gather_count = ai->par_size;
  size_t gather_byte_size = type_size * gather_count;
  void *pack_buffer = _XMP_alloc(gather_byte_size);
  memcpy(pack_buffer, array_addr + (type_size * ai->local_lower), gather_byte_size);

  MPI_Allgather(pack_buffer, gather_count, mpi_datatype,
                array_addr, gather_count, mpi_datatype,
                *(ai->shadow_comm));

  _XMP_free(pack_buffer);
}

// FIXME not implemented yet
static void _XMP_reflect_shadow_FULL_BCAST(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);
  _XMP_ASSERT(array_desc->is_allocated);

  int array_type = array_desc->type;
  size_t array_type_size = array_desc->type_size; 
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED); // checked by compiler
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_DUPLICATION); // checked by compiler
  _XMP_ERR_WHEN(!ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  _XMP_template_chunk_t *chunk = ai->align_template_chunk;
  unsigned long long chunk_width = chunk->par_chunk_width;

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // alloc buffer
  void *bcast_buffer = _XMP_alloc(chunk_width * (ai->dim_elmts) * array_type_size);

  // calc index
  int pack_lower[array_dim], pack_upper[array_dim];
  int unpack_lower[array_dim], unpack_upper[array_dim];
  int stride[array_dim];
  unsigned long long dim_acc[array_dim];
  for (int i = 0; i < array_dim; i++) {
    pack_lower[i] = array_desc->info[i].local_lower;
    pack_upper[i] = array_desc->info[i].local_upper;

    unpack_lower[i] = pack_lower[i];
    unpack_upper[i] = pack_upper[i];

    stride[i] = array_desc->info[i].local_stride;
    dim_acc[i] = array_desc->info[i].dim_acc;
  }

  for (int i = 0; i < size; i++) {
    int bcast_width = 0;
    if (i == rank) {
      // pack data
      _XMP_pack_array(bcast_buffer, array_addr, array_type, array_type_size,
                      array_dim, pack_lower, pack_upper, stride, dim_acc);

      bcast_width = _XMP_M_COUNT_TRIPLETi(pack_lower[array_index], pack_upper[array_index], stride[array_index]);
    }
    else {
      // calc unpack index
      switch (ai->align_manner) {
        case _XMP_N_ALIGN_BLOCK:
          {
            unpack_lower[array_index] = i * chunk_width;

            if (i == (size - 1)) {
              unpack_upper[array_index] = ai->ser_upper;
            }
            else {
              unpack_upper[array_index] = unpack_lower[array_index] + chunk_width - 1;
            }

            break;
          }
        case _XMP_N_ALIGN_CYCLIC:
          {
            unpack_lower[array_index] = i;

            int cycle = ai->par_stride;
            int mod = ai->ser_upper % cycle;
            if (i > mod) {
              unpack_upper[array_index] = i + (cycle * (chunk_width - 2));
            }
            else {
              unpack_upper[array_index] = i + (cycle * (chunk_width - 1));
            }

            break;
          }
        default:
          _XMP_fatal("unknown align manner");
      }

      bcast_width = _XMP_M_COUNT_TRIPLETi(unpack_lower[array_index], unpack_upper[array_index], stride[array_index]);
    }

    // bcast data
    MPI_Bcast(bcast_buffer, bcast_width * ai->dim_elmts, mpi_datatype, i, *(ai->shadow_comm));

    if (i != rank) {
      // unpack data
      _XMP_unpack_array(array_addr, bcast_buffer, array_type, array_type_size,
                        array_dim, unpack_lower, unpack_upper, stride, dim_acc);
    }
  }

  _XMP_free(bcast_buffer);
}

void _XMP_reflect_shadow_FULL(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_addr != NULL);
  _XMP_ASSERT(array_desc != NULL);

  if (!array_desc->is_allocated) {
    return;
  }

  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);

  // using allgather/allgatherv in special cases
  if ((array_dim == 1) && (ai->align_manner == _XMP_N_ALIGN_BLOCK)) {
    if (ai->is_regular_chunk) {
      _XMP_reflect_shadow_FULL_ALLGATHER(array_addr, array_desc, array_index);
    }
    else {
      _XMP_reflect_shadow_FULL_ALLGATHERV(array_addr, array_desc, array_index);
    }
  }
  else {
    _XMP_reflect_shadow_FULL_BCAST(array_addr, array_desc, array_index);
  }
}
