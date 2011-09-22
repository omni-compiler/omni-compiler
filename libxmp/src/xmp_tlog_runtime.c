/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_tlog.h"

void _XMP_tlog_init(void) {
  tlog_initialize();
}

void _XMP_tlog_finalize(void) {
  tlog_finalize();
}
