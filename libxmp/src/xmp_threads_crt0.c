/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

extern int _XMP_main(int argc, char *argv[]);
extern void ompc_init(int argc, char *argv[]);
extern void ompc_terminate(int);

int main(int argc, char *argv[]) {
  _XMP_init(&argc, &argv);
  ompc_init(argc,argv);

  int ret = _XMP_main(argc, argv);

  _XMP_finalize();
  ompc_terminate(ret);

  return 0;
}
