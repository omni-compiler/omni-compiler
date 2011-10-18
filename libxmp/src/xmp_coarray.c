#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"

void _XMP_init_coarray_STATIC(_XMP_coarray_t **coarray, void *addr,
                              int type, size_t type_size, int dim, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  va_list args;
  va_start(args, dim);

  int dim_size[dim];
  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }

  va_end(args);

  // FIXME
  //if (_XMP_world_size != coarray_size) {
  //  _XMP_fatal("wrong coarray size");
  //}

  c->comm = NULL;
  // FIXME init ci
  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC(_XMP_coarray_t **coarray, void *addr,
                               int type, size_t type_size, int dim, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  va_list args;
  va_start(args, dim);

  int dim_size[dim];
  for (int i = 0; i < dim - 1; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    dim_size[i] = elmts;
  }

  va_end(args);

  c->comm = NULL;
  // FIXME init ci
  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

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
