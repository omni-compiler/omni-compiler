#include "xmp_internal.h"

static int _XMP_runtime_working = _XMP_N_INT_FALSE;

void _XMP_init(void) {
  if (!_XMP_runtime_working) {
    // XXX how to get command line args?
    _XMP_init_world(NULL, NULL);

    _XMP_runtime_working = _XMP_N_INT_TRUE;
  }
}

void _XMP_finalize(void) {
  if (_XMP_runtime_working) {
    _XMP_finalize_world();
    _XMP_runtime_working = _XMP_N_INT_FALSE;
  }
}
