#include "xmp_internal.h"

static _Bool _XMP_runtime_working = false;

void _XMP_init(int *argc, char ***argv) {
  if (!_XMP_runtime_working) {
    _XMP_init_world(argc, argv);

    _XMP_runtime_working = true;
  }
}

void _XMP_init_in_constructor(void) {
  _XMP_init(NULL, NULL);
}

void _XMP_finalize(void) {
  if (_XMP_runtime_working) {
    _XMP_finalize_world();
    _XMP_runtime_working = false;
  }
}
