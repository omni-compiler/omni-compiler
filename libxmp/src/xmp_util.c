/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "xmp_internal.h"

static unsigned long long _XMP_on_ref_id_count = 0;

unsigned long long _XMP_get_on_ref_id(void) {
  if (_XMP_on_ref_id_count == ULLONG_MAX) {
    _XMP_fatal("cannot create a new nodes/template: too many");
  }

  return _XMP_on_ref_id_count++;
}

void *_XMP_alloc(size_t size) {
  void *addr;

  addr = malloc(size);
  if (addr == NULL) {
    _XMP_fatal("cannot allocate memory");
  }

  return addr;
}

void _XMP_free(void *p) {
  free(p);
}

void _XMP_fatal(char *msg) {
  fprintf(stderr, "[RANK:%d] XcalableMP runtime error: %s\n", _XMP_world_rank, msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void _XMP_fatal_nomsg(){
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void _XMP_unexpected_error(void) {
  _XMP_fatal("unexpected error in runtime");
}
