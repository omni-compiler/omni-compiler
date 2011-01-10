/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_math_function.h"

int _XMP_modi_ll_i(long long value, int cycle) {
  int mod = value % cycle;
  if (mod < 0) {
    return (mod += cycle) % cycle;
  }
  else {
    return mod;
  }
}

int _XMP_modi_i_i(int value, int cycle) {
  int mod = value % cycle;
  if (mod < 0) {
    return (mod += cycle) % cycle;
  }
  else {
    return mod;
  }
}
