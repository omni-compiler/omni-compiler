#include "xmp_internal.h"
#include "mpi.h"
#include "mpi-ext.h"
#include <stdio.h>

static int _XMP_runtime_working = _XMP_N_INT_FALSE;

int _XMPC_running = 1;
int _XMPF_running = 0;

void _XMP_init(int argc, char** argv) {
  if (!_XMP_runtime_working) {
#ifdef _XMP_COARRAY_GASNET
    _XMP_coarray_initialize(argc, argv);
    _XMP_post_initialize();
#endif
#ifdef _XMP_COARRAY_FJRDMA
    MPI_Init(&argc, &argv);
    _XMP_fjrdma_initialize(argc, argv);
#endif
    // XXX how to get command line args?
    _XMP_init_world(NULL, NULL);
    _XMP_runtime_working = _XMP_N_INT_TRUE;
  }
}

void _XMP_finalize(void) {
  if (_XMP_runtime_working) {
    _XMP_finalize_world();
    _XMP_runtime_working = _XMP_N_INT_FALSE;

#ifdef _XMP_COARRAY_GASNET
    _XMP_coarray_finalize();
#endif
#ifdef _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_finalize();
#endif
  }
}

char *_XMP_desc_of(void *p) {
  return (char *)p;
}
