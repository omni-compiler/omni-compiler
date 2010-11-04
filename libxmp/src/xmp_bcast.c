#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void _XCALABLEMP_bcast_NODES_ENTIRE_OMITTED(_XCALABLEMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size) {
  assert(bcast_nodes != NULL);

  if (!(bcast_nodes->is_member)) {
    return;
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, _XCALABLEMP_N_DEFAULT_ROOT_RANK, *(bcast_nodes->comm));
}

void _XCALABLEMP_bcast_NODES_ENTIRE_GLOBAL(_XCALABLEMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                           int from_lower, int from_upper, int from_stride) {
  assert(bcast_nodes != NULL);

  _XCALABLEMP_validate_nodes_ref(&from_lower, &from_upper, &from_stride, _XCALABLEMP_world_size);

  if (!(bcast_nodes->is_member)) {
    return;
  }

  // check <from-ref>
  if (_XCALABLEMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
    _XCALABLEMP_fatal("broadcast failed, multiple source nodes indicated");
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, from_lower, *(bcast_nodes->comm));
}

// FIXME read spec
void _XCALABLEMP_bcast_NODES_ENTIRE_NODES(_XCALABLEMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                          _XCALABLEMP_nodes_t *from_nodes, ...) {
  assert(bcast_nodes != NULL);
  assert(from_nodes != NULL);

  if (!(bcast_nodes->is_member)) {
    return;
  }

  if (!(from_nodes->is_member)) {
    _XCALABLEMP_fatal("broadcast failed, cannot find the source node");
  }

  // calc source nodes number
  int root = 0;
  int acc_nodes_size = 1;
  int from_dim = from_nodes->dim;
  int from_lower, from_upper, from_stride;

  va_list args;
  va_start(args, from_nodes);
  for (int i = 0; i < from_dim; i++) {
    int size = from_nodes->info[i].size;
    int rank = from_nodes->info[i].rank;

    if (va_arg(args, int) == 1) {
      root += (acc_nodes_size * rank);
    }
    else {
      from_lower = va_arg(args, int);
      from_upper = va_arg(args, int);
      from_stride = va_arg(args, int);

      // check <from-ref>
      _XCALABLEMP_validate_nodes_ref(&from_lower, &from_upper, &from_stride, size);
      if (_XCALABLEMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
        _XCALABLEMP_fatal("multiple source nodes indicated in bcast directive");
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
  MPI_Bcast(addr, count, mpi_datatype, root, *(bcast_nodes->comm));
}

// void _XCALABLEMP_M_BCAST_EXEC_OMITTED(void *addr, int count, size_t datatype_size)
// void _XCALABLEMP_M_BCAST_EXEC_GLOBAL(void *addr, int count, size_t datatype_size, int from_lower, int from_upper, int from_stride)
// void _XCALABLEMP_M_BCAST_EXEC_NODES(void *addr, int count, size_t datatype_size, _XCALABLEMP_nodes_t *from_nodes, ...)
