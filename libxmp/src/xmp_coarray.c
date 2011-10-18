#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"

static void _XMP_init_coarray_comm(_XMP_coarray_t *coarray) {
  size_t type_size = coarray->type_size;

  MPI_Win *win = _XMP_alloc(sizeof(MPI_Win));
  MPI_Win_create(coarray->addr, coarray->total_elmts * type_size, type_size,
                 MPI_INFO_NULL, MPI_COMM_WORLD, win);

  coarray->comm = win;
}

void _XMP_init_coarray_STATIC(_XMP_coarray_t **coarray, void *addr,
                              int type, size_t type_size, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  unsigned long long total_elmts = 1;
  va_list args;
  va_start(args, type_size);

  int array_dim = va_arg(args, int);
  for (int i = 0; i < array_dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("array has no elmts in this dimension");
    }

    total_elmts *= elmts;
  }

  int coarray_dim = va_arg(args, int);
  int coarray_dim_size[coarray_dim];
  for (int i = 0; i < coarray_dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    coarray_dim_size[i] = elmts;
  }

  va_end(args);

  c->total_elmts = total_elmts;

  // FIXME
  //if (_XMP_world_size != coarray_size) {
  //  _XMP_fatal("wrong coarray size");
  //}

  _XMP_init_coarray_comm(c);
  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC(_XMP_coarray_t **coarray, void *addr,
                               int type, size_t type_size, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  unsigned long long total_elmts = 1;
  va_list args;
  va_start(args, type_size);

  int array_dim = va_arg(args, int);
  for (int i = 0; i < array_dim; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("array has no elmts in this dimension");
    }

    total_elmts *= elmts;
  }

  int coarray_dim = va_arg(args, int);
  int coarray_dim_size[coarray_dim];
  for (int i = 0; i < coarray_dim - 1; i++) {
    int elmts = va_arg(args, int);
    if (elmts <= 0) {
      _XMP_fatal("coarray has no elmts in this dimension");
    }

    coarray_dim_size[i] = elmts;
  }

  va_end(args);

  c->total_elmts = total_elmts;

  _XMP_init_coarray_comm(c);
  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

  *coarray = c;
}

void _XMP_finalize_coarray(_XMP_coarray_t *coarray) {
  // FIXME??? correct?
  if (coarray != NULL) {
    MPI_Win_free(coarray->comm);
    _XMP_free(coarray->comm);
    _XMP_free(coarray);
  }
}
