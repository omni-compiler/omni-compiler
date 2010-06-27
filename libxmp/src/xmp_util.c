#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"

void *_XCALABLEMP_alloc(size_t size) {
  void *addr;

  addr = malloc(size);
  if (addr == NULL)
    _XCALABLEMP_fatal("cannot allocate memory");

  return addr;
}

void _XCALABLEMP_free(void *p) {
  if (p != NULL) free(p);
}

void _XCALABLEMP_fatal(char *msg) {
  fprintf(stderr, "XcalableMP runtime error: %s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

