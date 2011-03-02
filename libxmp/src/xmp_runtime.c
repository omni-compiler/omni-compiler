#include "xmp_internal.h"

static _Bool _XMP_runtime_working = false;

void _XMP_init(void) {
  if (!_XMP_runtime_working) {
    // XXX how to get command line args?
    _XMP_init_world(NULL, NULL);

    _XMP_runtime_working = true;
  }
}

void _XMP_finalize(void) {
  if (_XMP_runtime_working) {
    _XMP_finalize_world();
    _XMP_runtime_working = false;
  }
}
