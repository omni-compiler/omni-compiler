#include "xmp_math_function.h"

int _XCALABLEMP_modi_ll_i(long long value, int cycle) {
  int mod = value % cycle;
  if (mod < 0) {
    return (mod += cycle) % cycle;
  }
  else {
    return mod;
  }
}

int _XCALABLEMP_modi_i_i(int value, int cycle) {
  int mod = value % cycle;
  if (mod < 0) {
    return (mod += cycle) % cycle;
  }
  else {
    return mod;
  }
}
