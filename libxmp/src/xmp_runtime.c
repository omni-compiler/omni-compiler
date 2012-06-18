#include "xmp_internal.h"

static int _XMP_runtime_working = _XMP_N_INT_FALSE;

void _XMP_init(int argc, char** argv) {
  if (!_XMP_runtime_working) {
    // XXX how to get command line args?
		_XMP_init_world(NULL, NULL);

    _XMP_runtime_working = _XMP_N_INT_TRUE;

#ifdef _COARRAY_GASNET
		_XMP_coarray_initialize(argc, argv);
#endif
  }
}

void _XMP_finalize(void) {
  if (_XMP_runtime_working) {
    _XMP_finalize_world();
    _XMP_runtime_working = _XMP_N_INT_FALSE;

#ifdef _COARRAY_GASNET
		_XMP_coarray_finalize();
#endif
  }
}

char *_XMP_desc_of(void *p) {
  return (char *)p;
}
