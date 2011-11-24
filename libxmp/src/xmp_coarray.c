#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"

_XMP_coarray_list_t *_XMP_coarray_list_head = NULL;
_XMP_coarray_list_t *_XMP_coarray_list_tail = NULL;

static void _XMP_add_coarray(_XMP_coarray_t *coarray) {
  _XMP_coarray_list_t *coarray_list = _XMP_alloc(sizeof(_XMP_coarray_list_t));

  coarray_list->coarray = coarray;
  coarray_list->next = NULL;

  if (_XMP_coarray_list_head == NULL) {
    _XMP_coarray_list_head = coarray_list;
    _XMP_coarray_list_tail = coarray_list;
  } else {
    _XMP_coarray_list_tail->next = coarray_list;
    _XMP_coarray_list_tail = coarray_list;
  }
}

static _XMP_coarray_t *_XMP_alloc_coarray_desc(void *addr, int type, size_t type_size) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));

  c->addr = addr;
  c->type = type;
  c->type_size = type_size;

  return c;
}

void _XMP_init_coarray_STATIC(_XMP_coarray_t **coarray, void *addr,
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
  c->nodes = _XMP_init_nodes_struct_GLOBAL(dim, dim_size, _XMP_N_INT_TRUE);
  c->comm = NULL;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC(_XMP_coarray_t **coarray, void *addr,
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
  c->nodes = _XMP_init_nodes_struct_GLOBAL(dim, dim_size, _XMP_N_INT_FALSE);
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

  _XMP_nodes_t *coarray_nodes = coarray->nodes;

  MPI_Win *win = _XMP_alloc(sizeof(MPI_Win));
  MPI_Win_create(coarray->addr, total_elmts * type_size, type_size,
                 MPI_INFO_NULL, *((MPI_Comm *)coarray_nodes->comm), win);

  coarray->comm = win;

  if (coarray_nodes->is_member) {
    MPI_Win_fence(0, *win);
  }

  // FIXME correct implementation???
  _XMP_add_coarray(coarray);
}

void _XMP_finalize_coarray_comm(_XMP_coarray_t *coarray) {
  if ((coarray->nodes)->is_member) {
    MPI_Win_fence(0, *((MPI_Win *)coarray->comm));
  }

  if (coarray != NULL) {
    if (coarray->comm != NULL) {
      MPI_Win_free(coarray->comm);
      _XMP_free(coarray->comm);
    }
  }
}
