/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

extern int _XMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  _XMP_init_world(&argc, &argv);

  _XMP_barrier_WORLD();
  int ret = _XMP_main(argc, argv);
  _XMP_barrier_WORLD();

  return _XMP_finalize_world(ret);
}
