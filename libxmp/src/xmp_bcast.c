/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void _XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, _XMP_N_DEFAULT_ROOT_RANK, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

void _XMP_bcast_NODES_ENTIRE_GLOBAL(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                    int from_lower, int from_upper, int from_stride) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  // check <from-ref>
  if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
    _XMP_fatal("broadcast failed, multiple source nodes indicated");
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, from_lower, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

// FIXME read spec
void _XMP_bcast_NODES_ENTIRE_NODES(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                   bool is_on, _XMP_nodes_t *from_nodes, ...) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  if (!from_nodes->is_member) {
    _XMP_fatal("broadcast failed, cannot find the source node");
  }

  // calc source nodes number
  int root = 0;
  int acc_nodes_size = 1;
  int from_dim, size, rank;

  if(is_on == false)  // is there on clause ?
    from_dim = from_nodes->dim;
  else
    from_dim = bcast_nodes->dim;

  int from_lower, from_upper, from_stride;
  va_list args;
  bool flag = true;
  va_start(args, from_nodes);
  for (int i = 0; i < from_dim; i++) {
    if(is_on == false){
      size = from_nodes->info[i].size;
      rank = from_nodes->info[i].rank;
    }
    else{
      size = bcast_nodes->info[i].size; 
      rank = bcast_nodes->info[i].rank;
      flag = false;
    }

    if (va_arg(args, int) == 1) {
      if(flag == false) _XMP_fatal("Sorry Not implemented bcast operation on p(*,?)");
      root += (acc_nodes_size * rank);
    }
    else {
      from_lower = va_arg(args, int) - 1;
      from_upper = va_arg(args, int) - 1;
      from_stride = va_arg(args, int);

      // check <from-ref>
      if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
	_XMP_fatal("multiple source nodes indicated in bcast directive");
      }

      root += (acc_nodes_size * (from_lower));
    }

    acc_nodes_size *= size;
  }
  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);
  
  // bcast
  MPI_Bcast(addr, count, mpi_datatype, root, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

// void _XMP_M_BCAST_EXEC_OMITTED(void *addr, int count, size_t datatype_size)
// void _XMP_M_BCAST_EXEC_GLOBAL(void *addr, int count, size_t datatype_size, int from_lower, int from_upper, int from_stride)
// void _XMP_M_BCAST_EXEC_NODES(void *addr, int count, size_t datatype_size, _XMP_nodes_t *from_nodes, ...)
