/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"

void *_XCALABLEMP_alloc(size_t size) {
  void *addr;

  addr = malloc(size);
  if (addr == NULL) {
    _XCALABLEMP_fatal("cannot allocate memory");
  }

  return addr;
}

void _XCALABLEMP_free(void *p) {
  _XCALABLEMP_ASSERT(p != NULL);

  free(p);
}

void _XCALABLEMP_fatal(char *msg) {
  _XCALABLEMP_ASSERT(msg != NULL);

  fprintf(stderr, "[RANK:%d] XcalableMP runtime error: %s\n", _XCALABLEMP_world_rank, msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void _XCALABLEMP_unexpected_error(void) {
  _XCALABLEMP_fatal("unexpected error in runtime");
}
