/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdio.h>
#include "xmp_internal.h"

#include "xmp_tlog.h"

void
__xmp_tlog_initialize(void) {
  tlog_initialize();
}

void
__xmp_tlog_finalize(void) {
  tlog_finalize();
}

extern int _XMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  __xmp_tlog_initialize();
  int ret = _XMP_main(argc, argv);
  __xmp_tlog_finalize();

  return ret;
}
