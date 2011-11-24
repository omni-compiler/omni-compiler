#include <stdarg.h>
#include "mpi.h"
#include "xmp_constant.h"
#include "xmp_internal.h"

#define _XMP_SM_CALC_RMA_OFFSET(offset_addr, coarray_addr, type_size) \
(((offset_addr) - (coarray_addr)) / (type_size))

void _XMP_coarray_rma_SCALAR(int rma_code, _XMP_coarray_t *coarray, void *offset_addr, void *rma_addr, ...) {
  _XMP_nodes_t *coarray_nodes = coarray->nodes;
  int coarray_nodes_dim = coarray_nodes->dim;
  int coarray_nodes_ref[coarray_nodes_dim];

  va_list args;
  va_start(args, rma_addr);
  for (int i = 0; i < coarray_nodes_dim; i++) {
    // XXX translate 1-origin to 0-rigin
    coarray_nodes_ref[i] = va_arg(args, int) - 1;
  }
  va_end(args);

  int coarray_rank = _XMP_calc_linear_rank(coarray_nodes, coarray_nodes_ref);

  int type_size = coarray->type_size;
  switch (rma_code) {
    case _XMP_N_COARRAY_GET:
      MPI_Get(rma_addr, type_size, MPI_BYTE, coarray_rank,
              _XMP_SM_CALC_RMA_OFFSET(offset_addr, coarray->addr, type_size),
              type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    case _XMP_N_COARRAY_PUT:
      MPI_Put(rma_addr, type_size, MPI_BYTE, coarray_rank,
              _XMP_SM_CALC_RMA_OFFSET(offset_addr, coarray->addr, type_size),
              type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    default:
      _XMP_fatal("unknown coarray rma expression");
  }
}

void _XMP_coarray_rma_ARRAY(int rma_code, _XMP_coarray_t *coarray, void *rma_addr, ...) {
}

void _XMP_coarray_sync(void) {
  for (_XMP_coarray_list_t *coarray_list = _XMP_coarray_list_head; coarray_list != NULL; coarray_list = coarray_list->next) {
    _XMP_coarray_t *coarray = coarray_list->coarray;
    if ((coarray->nodes)->is_member) {
      MPI_Win_fence(0, *((MPI_Win *)coarray->comm));
    }
  }

  _XMP_barrier_EXEC();
}
