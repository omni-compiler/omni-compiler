/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

extern int _XMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  _XMP_gpu_init();
  int ret = _XMP_main(argc, argv);
  _XMP_gpu_finalize();

  return ret;
}
