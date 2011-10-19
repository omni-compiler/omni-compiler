#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"

static _XMP_coarray_t *_XMP_alloc_coarray_desc(void *addr, int type, size_t type_size) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  return c;
}

void _XMP_init_coarray_STATIC_EXEC(_XMP_coarray_t **coarray, void *addr,
                                   int type, size_t type_size, int dim, ...) {
  int dim_size[dim];

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_EXEC(dim, dim_size, _XMP_N_INT_TRUE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC_EXEC(_XMP_coarray_t **coarray, void *addr,
                                    int type, size_t type_size, int dim, ...) {
  int dim_size[dim - 1];

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim - 1; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_EXEC(dim, dim_size, _XMP_N_INT_FALSE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_STATIC_NODES_NUMBER(_XMP_coarray_t **coarray, void *addr, int type, size_t type_size, int dim,
                                           int ref_lower, int ref_upper, int ref_stride, ...) {
  int dim_size[dim];

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride,
                                                 dim_size, _XMP_N_INT_TRUE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC_NODES_NUMBER(_XMP_coarray_t **coarray, void *addr, int type, size_t type_size, int dim,
                                            int ref_lower, int ref_upper, int ref_stride, ...) {
  int dim_size[dim - 1];

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim - 1; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride,
                                                 dim_size, _XMP_N_INT_FALSE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_STATIC_NODES_NAMED(_XMP_coarray_t **coarray, void *addr, int type, size_t type_size, int dim,
                                          _XMP_nodes_t *ref_nodes, ...) {
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim];
  int ref_upper[ref_dim];
  int ref_stride[ref_dim];
  int dim_size[dim];

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      ref_lower[i] = va_arg(args, int);
      ref_upper[i] = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper, ref_stride,
                                                dim_size, _XMP_N_INT_TRUE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC_NODES_NAMED(_XMP_coarray_t **coarray, void *addr, int type, size_t type_size, int dim,
                                           _XMP_nodes_t *ref_nodes, ...) {
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim];
  int ref_upper[ref_dim];
  int ref_stride[ref_dim];
  int dim_size[dim - 1];

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      ref_lower[i] = va_arg(args, int);
      ref_upper[i] = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  for (int i = 0; i < dim - 1; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }
  va_end(args);

  _XMP_coarray_t *c = _XMP_alloc_coarray_desc(addr, type, type_size);
  c->nodes = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper, ref_stride,
                                                dim_size, _XMP_N_INT_FALSE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_comm(_XMP_coarray_t *coarray, int dim, ...) {
  size_t type_size = coarray->type_size;

  unsigned long long total_elmts = 1;
  va_list args;
  va_start(args, dim);

  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("array has no elmts in this dimension");
    }

    total_elmts *= elmts;
  }

  va_end(args);

  MPI_Win *win = _XMP_alloc(sizeof(MPI_Win));
  MPI_Win_create(coarray->addr, total_elmts * type_size, type_size,
                 MPI_INFO_NULL, MPI_COMM_WORLD, win);

  coarray->comm = win;
}

void _XMP_finalize_coarray_comm(_XMP_coarray_t *coarray) {
  if (coarray != NULL) {
    if (coarray->comm != NULL) {
      MPI_Win_free(coarray->comm);
      _XMP_free(coarray->comm);
    }
  }
}
