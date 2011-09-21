/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

extern void ompc_init(int argc, char *argv[]);
extern int _XMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  ompc_init(argc, argv);
  return _XMP_main(argc, argv);
}
