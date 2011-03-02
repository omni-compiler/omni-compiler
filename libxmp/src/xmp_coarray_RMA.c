#include "xmp_internal.h"
#include <stdio.h>

void _XMP_coarray_get(_XMP_coarray_t *coarray, void *addr) {
  printf("[%d] get\n", _XMP_world_rank);
}

void _XMP_coarray_put(void *addr, _XMP_coarray_t *coarray) {
  printf("[%d] put\n", _XMP_world_rank);
}
