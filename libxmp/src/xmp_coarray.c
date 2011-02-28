#include <stdarg.h>
#include <xmp_internal.h>

void _XMP_init_coarray_STATIC(_XMP_coarray_t **coarray, void *addr,
                              int type, size_t type_size, int coarray_size, int dim, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->type = type;
  c->type_size = type_size;
  c->addr = addr;

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

  c->total_elmts = total_elmts;

  if (_XMP_world_size != coarray_size) {
    _XMP_fatal("wrong coarray size");
  }

  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

  *coarray = c;
}

void _XMP_init_coarray_DYNAMIC(_XMP_coarray_t **coarray, void *addr,
                               int type, size_t type_size, int dim, ...) {
  _XMP_coarray_t *c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_info_t *ci = &(c->info[0]);

  c->type = type;
  c->type_size = type_size;
  c->addr = addr;

  unsigned long long total_elmts = 1;
  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int elmts = va_arg(args, int);
    _XMP_ASSERT(elmts > 0);

    total_elmts *= elmts;
  }
  va_end(args);

  c->total_elmts = total_elmts;

  ci->size = _XMP_world_size;
  ci->rank = _XMP_world_rank;

  *coarray = c;
}

void _XMP_init_coarray_COMM(_XMP_coarray_COMM_t **coarray_comm, _XMP_coarray_t *coarray) {
  return;
}
