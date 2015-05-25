/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */
#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

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

_Bool union_triplet(int lb0, int ub0, int st0, int lb1, int ub1, int st1){

  if (ub0 < lb0 || ub1 < lb0) return false;

  int lb2, ub2, st2;
  int lb3,      st3;

  if (lb0 > lb1){
    lb2 = lb0;
    lb3 = lb1;
    st2 = st0;
    st3 = st1;
  }
  else {
    lb2 = lb1;
    lb3 = lb0;
    st2 = st1;
    st3 = st0;
  }

  if (ub0 > ub1){
    ub2 = ub1;
  }
  else {
    ub2 = ub0;
  }

  for (int i = lb2; i <= ub2; i += st2){
    if ((i - lb3) % st3 == 0) return true;
  }

  return false;

}
