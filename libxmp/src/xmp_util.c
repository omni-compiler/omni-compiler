/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"

void *_XMP_alloc(size_t size) {
  void *addr;

  addr = malloc(size);
  if (addr == NULL) {
    _XMP_fatal("cannot allocate memory");
  }

  return addr;
}

void _XMP_free(void *p) {
  _XMP_ASSERT(p != NULL);

  free(p);
}

void _XMP_fatal(char *msg) {
  _XMP_ASSERT(msg != NULL);

  fprintf(stderr, "[RANK:%d] XcalableMP runtime error: %s\n", _XMP_world_rank, msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void _XMP_unexpected_error(void) {
  _XMP_fatal("unexpected error in runtime");
}
