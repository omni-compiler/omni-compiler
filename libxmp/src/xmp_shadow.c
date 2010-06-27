#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"

void _XCALABLEMP_init_shadow(_XCALABLEMP_array_t *array, ...) {
  if (array == NULL) return;

  int dim = array->dim;
  va_list args;
  va_start(args, array);
  for (int i = 0; i < dim; i++) {
    _XCALABLEMP_array_info_t *ai = &(array->info[i]);
    int type = va_arg(args, int);
    switch (type) {
      case _XCALABLEMP_N_SHADOW_NORMAL:
        {
          int lo = va_arg(args, int);
          if (lo < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          int hi = va_arg(args, int);
          if (hi < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          if ((lo != 0) || (hi != 0)) {
            ai->shadow_type = _XCALABLEMP_N_SHADOW_NORMAL;
            ai->shadow_size_lo = lo;
            ai->shadow_size_hi = hi;
          }

          break;
        }
      case _XCALABLEMP_N_SHADOW_FULL:
        ai->shadow_type = _XCALABLEMP_N_SHADOW_FULL;
        break;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
}
