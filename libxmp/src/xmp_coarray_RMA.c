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
    // translate 1-origin to 0-rigin
    coarray_nodes_ref[i] = va_arg(args, int) - 1;
  }
  va_end(args);

  int coarray_rank = _XMP_calc_linear_rank(coarray_nodes, coarray_nodes_ref);
  int type_size = coarray->type_size;

  switch (rma_code) {
    case _XMP_N_COARRAY_GET:
      MPI_Get(rma_addr, type_size, MPI_BYTE, coarray_rank,
              _XMP_SM_CALC_RMA_OFFSET((char *)offset_addr, (char *)coarray->addr, type_size),
              type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    case _XMP_N_COARRAY_PUT:
      MPI_Put(rma_addr, type_size, MPI_BYTE, coarray_rank,
              _XMP_SM_CALC_RMA_OFFSET((char *)offset_addr, (char *)coarray->addr, type_size),
              type_size, MPI_BYTE, *((MPI_Win *)coarray->comm));
      break;
    default:
      _XMP_fatal("unknown coarray rma expression");
  }
}

// FIXME not implemented
void _XMP_coarray_rma_ARRAY(int rma_code, _XMP_coarray_t *coarray, void *rma_addr, ...) {
  va_list args;
  va_start(args, rma_addr);

  // get coarray info
  int coarray_dim = va_arg(args, int);
  int coarray_lower[coarray_dim], coarray_upper[coarray_dim], coarray_stride[coarray_dim];
  unsigned long long coarray_dim_acc[coarray_dim];
  for (int i = 0; i < coarray_dim; i++) {
    coarray_lower[i] = va_arg(args, int);
    coarray_upper[i] = va_arg(args, int);
    coarray_stride[i] = va_arg(args, int);
    coarray_dim_acc[i] = va_arg(args, unsigned long long);
  }

  // get rma_array info
  int rma_array_dim = va_arg(args, int);
  int rma_array_lower[rma_array_dim], rma_array_upper[rma_array_dim], rma_array_stride[rma_array_dim];
  unsigned long long rma_array_dim_acc[rma_array_dim];
  for (int i = 0; i < rma_array_dim; i++) {
    rma_array_lower[i] = va_arg(args, int);
    rma_array_upper[i] = va_arg(args, int);
    rma_array_stride[i] = va_arg(args, int);
    rma_array_dim_acc[i] = va_arg(args, unsigned long long);
  }

  // get coarray ref info
  _XMP_nodes_t *coarray_nodes = coarray->nodes;
  int coarray_nodes_dim = coarray_nodes->dim;
  int coarray_nodes_ref[coarray_nodes_dim];
  for (int i = 0; i < coarray_nodes_dim; i++) {
    // translate 1-origin to 0-rigin
    coarray_nodes_ref[i] = va_arg(args, int) - 1;
  }
  va_end(args);

  int coarray_rank = _XMP_calc_linear_rank(coarray_nodes, coarray_nodes_ref);

  MPI_Datatype *data_type = (MPI_Datatype *)coarray->data_type;
  if ((coarray_dim == 1) && (rma_array_dim == 1) &&
      (coarray_stride[0] == 1) && (rma_array_stride[0] == 1)) {
    switch (rma_code) {
      case _XMP_N_COARRAY_GET:
        MPI_Get((char *)rma_addr + rma_array_lower[0], rma_array_upper[0] - rma_array_lower[0] + 1, *data_type,
                coarray_rank, coarray_lower[0], coarray_upper[0] - coarray_lower[0] + 1, *data_type,
                *((MPI_Win *)coarray->comm));
        break;
      case _XMP_N_COARRAY_PUT:
        MPI_Put((char *)rma_addr + rma_array_lower[0], rma_array_upper[0] - rma_array_lower[0] + 1, *data_type,
                coarray_rank, coarray_lower[0], coarray_upper[0] - coarray_lower[0] + 1, *data_type,
                *((MPI_Win *)coarray->comm));
        break;
      default:
        _XMP_fatal("unknown coarray rma expression");
    }
  } else {
    _XMP_fatal("unsupported case: coarray");
  }
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
