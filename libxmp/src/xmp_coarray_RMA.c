#include <stdarg.h>
#include "mpi.h"
#include "xmp_constant.h"
#include "xmp_internal.h"

void _XMP_coarray_rma_SCALAR(int rma_code, _XMP_coarray_t *coarray, void *addr, ...) {
  _XMP_nodes_t *coarray_nodes = coarray->nodes;
  int coarray_nodes_dim = coarray_nodes->dim;
  int coarray_nodes_ref[coarray_nodes_dim];

  va_list args;
  va_start(args, addr);
  for (int i = 0; i < coarray_nodes_dim; i++) {
    coarray_nodes_ref[i] = va_arg(args, int);
  }
  va_end(args);

  int coarray_rank = _XMP_calc_linear_rank(coarray_nodes, coarray_nodes_ref);

  int type_size = coarray->type_size;
  switch (rma_code) {
    case _XMP_N_COARRAY_GET:
      MPI_Get(coarray->addr, type_size, MPI_BYTE, coarray_rank, 0, type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    case _XMP_N_COARRAY_PUT:
      MPI_Put(coarray->addr, type_size, MPI_BYTE, coarray_rank, 0, type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    default:
      _XMP_fatal("unknown coarray rma expression");
  }

  // FIXME for debug
  MPI_Win_fence(0, *((MPI_Win *)coarray->comm));
}

void _XMP_coarray_get_ARRAY(void *coarray, void *addr) {
}

void _XMP_coarray_put_ARRAY(void *addr, void *coarray) {
}

void _XMP_coarray_sync(void) {
  // FIXME fence all rma messaged
  _XMP_barrier_EXEC();
}
