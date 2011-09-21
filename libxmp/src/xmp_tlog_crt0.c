/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdio.h>
#include "xmp_internal.h"

#include "xmp_tlog.h"

static void _XMP_tlog_initialize(void) {
  tlog_initialize();
}

static void _XMP_tlog_finalize(void) {
  tlog_finalize();
}

extern int _XMP_main(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  _XMP_tlog_initialize();
  int ret = _XMP_main(argc, argv);
  _XMP_tlog_finalize();

  return ret;
}
