/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */
#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdarg.h>
#include <string.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void _XMP_create_shadow_comm(_XMP_array_t *array, int array_index) {
  _XMP_nodes_t *onto_nodes = (array->align_template)->onto_nodes;
  _XMP_array_info_t *ai = &(array->info[array_index]);

  _XMP_template_t *align_template = array->align_template;

  int color = 1;
  if (onto_nodes->is_member) {
    _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED);
    _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_DUPLICATION);

    _XMP_template_chunk_t *chunk = &(align_template->chunk[ai->align_template_index]);
    _XMP_ASSERT(chunk->dist_manner != _XMP_N_DIST_DUPLICATION);

    int onto_nodes_index = chunk->onto_nodes_index;

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
  } else {
    color = 0;
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  // set members
  if (array->is_allocated) {
    ai->is_shadow_comm_member = true;

    ai->shadow_comm = comm;
    MPI_Comm_size(*comm, &(ai->shadow_comm_size));
    MPI_Comm_rank(*comm, &(ai->shadow_comm_rank));
  } else {
    _XMP_finalize_comm(comm);
  }
}

static void _XMP_reflect_shadow_FULL_ALLGATHER(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_desc->is_allocated);
  _XMP_ASSERT(array_desc->dim == 1);

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  size_t type_size = array_desc->type_size;
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  int gather_count = ai->par_size;
  size_t gather_byte_size = type_size * gather_count;
  void *pack_buffer = _XMP_alloc(gather_byte_size);
  memcpy(pack_buffer, (char *)array_addr + (type_size * ai->local_lower), gather_byte_size);

  MPI_Allgather(pack_buffer, gather_count, mpi_datatype,
                array_addr, gather_count, mpi_datatype,
                *((MPI_Comm *)ai->shadow_comm));

  MPI_Type_free(&mpi_datatype);
  _XMP_free(pack_buffer);
}


static void _XMP_reflect_shadow_FULL_BCAST(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_ASSERT(array_desc->is_allocated);

  _XMP_template_t *align_template = array_desc->align_template;

  int array_type = array_desc->type;
  size_t array_type_size = array_desc->type_size;
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_NOT_ALIGNED);
  _XMP_ASSERT(ai->align_manner != _XMP_N_ALIGN_DUPLICATION);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  _XMP_template_chunk_t *chunk = &(align_template->chunk[ai->align_template_index]);;
  unsigned long long chunk_width = chunk->par_chunk_width;

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(array_type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // calc index
  int pack_lower[array_dim], pack_upper[array_dim];
  int unpack_lower[array_dim], unpack_upper[array_dim];
  int stride[array_dim];
  unsigned long long dim_acc[array_dim];
  int count = 1;
  for (int i = 0; i < array_dim; i++) {

    int width;

    if (i == array_index){
      pack_lower[i] = array_desc->info[i].local_lower;
      pack_upper[i] = array_desc->info[i].local_upper;
      stride[i] = array_desc->info[i].local_stride;
      width = _XMP_M_COUNT_TRIPLETi(pack_lower[i], pack_upper[i], stride[i]);
    }
    else {
      pack_lower[i] = 0;
      pack_upper[i] = array_desc->info[i].ser_upper - array_desc->info[i].ser_lower;
      stride[i] = 1;
      width = pack_upper[i] - pack_lower[i] + 1;
    }      

    unpack_lower[i] = pack_lower[i];
    unpack_upper[i] = pack_upper[i];

    dim_acc[i] = array_desc->info[i].dim_acc;

    count *= width;
  }

  // alloc buffer
  void *bcast_buffer = _XMP_alloc(count * array_type_size);

  for (int i = 0; i < size; i++) {
    //int bcast_width = 0;
    if (i == rank) {
      // pack data
      _xmp_pack_array(bcast_buffer, array_addr, array_type, array_type_size,
		       array_dim, pack_lower, pack_upper, stride, dim_acc);

      //bcast_width = _XMP_M_COUNT_TRIPLETi(pack_lower[array_index], pack_upper[array_index], stride[array_index]);
    }
    else {
      // calc unpack index
      switch (ai->align_manner) {
        case _XMP_N_ALIGN_BLOCK:
          {
            unpack_lower[array_index] = i * chunk_width;

            if (i == (size - 1)) {
              unpack_upper[array_index] = ai->ser_upper - ai->ser_lower;
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

      //bcast_width = _XMP_M_COUNT_TRIPLETi(unpack_lower[array_index], unpack_upper[array_index], stride[array_index]);
    }

    // bcast data
    //MPI_Bcast(bcast_buffer, bcast_width * ai->dim_elmts, mpi_datatype, i, *((MPI_Comm *)ai->shadow_comm));
    MPI_Bcast(bcast_buffer, count, mpi_datatype, i, *((MPI_Comm *)ai->shadow_comm));

    if (i != rank) {
      // unpack data
      _xmp_unpack_array(array_addr, bcast_buffer, array_type, array_type_size,
			 array_dim, unpack_lower, unpack_upper, stride, dim_acc);
    }

  }

  MPI_Type_free(&mpi_datatype);
  _XMP_free(bcast_buffer);
}

void _XMP_init_shadow(_XMP_array_t *array, ...) {
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
          _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);

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
              ai->local_lower += lo;
              ai->local_upper += lo;
           // ai->local_stride is not changed
              ai->alloc_size += lo + hi;

              *(ai->temp0) -= lo;
              ai->temp0_v -= lo;
            }

	    if (!ai->reflect_sched){
	      _XMP_reflect_sched_t *sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
	      sched->is_periodic = -1; /* not used yet */
	      sched->datatype_lo = MPI_DATATYPE_NULL;
	      sched->datatype_hi = MPI_DATATYPE_NULL;
	      for (int j = 0; j < 4; j++) sched->req[j] = MPI_REQUEST_NULL;
	      sched->lo_send_buf = NULL;
	      sched->lo_recv_buf = NULL;
	      sched->hi_send_buf = NULL;
	      sched->hi_recv_buf = NULL;
	      ai->reflect_sched = sched;
	    }
	    ai->reflect_acc_sched = NULL;

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

            ai->local_lower = ai->par_lower - ai->ser_lower;
            ai->local_upper = ai->par_upper - ai->ser_lower;
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


void _XMP_init_shadow_dim(_XMP_array_t *array, int i, int type, int lo, int hi){

  _XMP_array_info_t *ai = &(array->info[i]);

  switch (type) {
  case _XMP_N_SHADOW_NONE:
    ai->shadow_type = _XMP_N_SHADOW_NONE;
    break;
  case _XMP_N_SHADOW_NORMAL:
    {
      _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);

      if (lo < 0) {
	_XMP_fatal("<shadow-width> should be a nonnegative integer");
      }

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
	  ai->local_lower += lo;
	  ai->local_upper += lo;
	  // ai->local_stride is not changed
	  ai->alloc_size += lo + hi;

	  *(ai->temp0) -= lo;
	  ai->temp0_v -= lo;
	}

	if (!ai->reflect_sched){
	  _XMP_reflect_sched_t *sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
	  sched->is_periodic = -1; /* not used yet */
	  sched->datatype_lo = MPI_DATATYPE_NULL;
	  sched->datatype_hi = MPI_DATATYPE_NULL;
	  for (int j = 0; j < 4; j++) sched->req[j] = MPI_REQUEST_NULL;
	  sched->lo_send_buf = NULL;
	  sched->lo_recv_buf = NULL;
	  sched->hi_send_buf = NULL;
	  sched->hi_recv_buf = NULL;
	  ai->reflect_sched = sched;
	}
	ai->reflect_acc_sched = NULL;

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


/* void _XMP_init_shadow_noalloc(_XMP_array_t *a, int shadow_type, int lshadow, int ushadow) { */
/*   _XMP_ASSERT(a->dim == 1); */
/*   _XMP_array_info_t *ai = &(a->info[0]); */
/*   ai->shadow_type = shadow_type; */
/*   ai->shadow_size_lo = lshadow; */
/*   ai->shadow_size_hi = ushadow; */
/* } */

void _XMP_init_shadow_noalloc(_XMP_array_t *a, ...) {

  int dim = a->dim;

  va_list args;
  va_start(args, a);

  for (int i = 0; i < dim; i++) {
    _XMP_array_info_t *ai = &(a->info[i]);
    ai->shadow_type = va_arg(args, int);
    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      ai->shadow_size_lo = va_arg(args, int);
      ai->shadow_size_hi = va_arg(args, int);
    }
  }

  va_end(args);
}

void _XMP_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr,
                             _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  int array_type = array_desc->type;
  int array_dim = array_desc->dim;

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
      (*_xmp_pack_array)(*lo_buffer, array_addr, array_type, array_desc->type_size,
			 array_dim, lower, upper, stride, dim_acc);
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
      (*_xmp_pack_array)(*hi_buffer, array_addr, array_type, array_desc->type_size,
			 array_dim, lower, upper, stride, dim_acc);
    }
  }
}

void _XMP_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr,
                               _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  int array_type = array_desc->type;
  int array_dim = array_desc->dim;

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
      (*_xmp_unpack_array)(array_addr, lo_buffer, array_type, array_desc->type_size,
			   array_dim, lower, upper, stride, dim_acc);

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
      (*_xmp_unpack_array)(array_addr, hi_buffer, array_type, array_desc->type_size,
			   array_dim, lower, upper, stride, dim_acc);

      // free buffer
      _XMP_free(hi_buffer);
    }
  }
}

void _XMP_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
                                 void *lo_send_buffer, void *hi_send_buffer,
                                 _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  // get communicator info
  int size = ai->shadow_comm_size;
  if (size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  MPI_Comm *comm = ai->shadow_comm;

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

  MPI_Type_free(&mpi_datatype);
}

void _XMP_reflect_shadow_FULL(void *array_addr, _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  if ((ai->shadow_comm_size) == 1) {
    return;
  }

  int array_dim = array_desc->dim;

  // using allgather/allgatherv in special cases
  if ((array_dim == 1) && (ai->align_manner == _XMP_N_ALIGN_BLOCK) && (ai->is_regular_chunk)) {
    _XMP_reflect_shadow_FULL_ALLGATHER(array_addr, array_desc, array_index);
  }
  else {
    _XMP_reflect_shadow_FULL_BCAST(array_addr, array_desc, array_index);
  }
}
