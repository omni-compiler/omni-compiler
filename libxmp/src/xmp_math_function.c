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

int _XMP_ceili(int a, int b) {
  int div = a / b;

  if ((a < 0) || (b < 0)) {
    return div;
  } else {
    if ((a % b) == 0) {
      return div;
    } else {
      return div + 1;
    }
  }
}

int _XMP_floori(int a, int b) {
  int div = a / b;

  if ((a < 0) || (b < 0)) {
    if ((a % b) == 0) {
      return div;
    } else {
      return div - 1;
    }
  } else {
    return div;
  }
}

int _XMP_gcd(int a, int b) {
  int r = a % b;
  if (r == 0) {
    return b;
  } else {
    return _XMP_gcd(b, r);
  }
}

int _XMP_lcm(int a, int b) {
  return (a * b) / _XMP_gcd(a, b);
}
