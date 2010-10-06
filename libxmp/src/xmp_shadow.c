#include <stdarg.h>
#include "xmp_array_section.h"
#include "xmp_constant.h"
#include "xmp_internal.h"

void _XCALABLEMP_init_shadow(_XCALABLEMP_array_t *array, ...) {
  if (array == NULL) return;

  int dim = array->dim;
  va_list args;
  va_start(args, array);
  for (int i = 0; i < dim; i++) {
    int type = va_arg(args, int);
    _XCALABLEMP_array_info_t *ai = &(array->info[i]);
    ai->shadow_type = type;

    switch (type) {
      case _XCALABLEMP_N_SHADOW_NONE:
        break;
      case _XCALABLEMP_N_SHADOW_NORMAL:
        {
          int lo = va_arg(args, int);
          if (lo < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          int hi = va_arg(args, int);
          if (hi < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          if ((lo != 0) || (hi != 0)) {
            ai->shadow_size_lo = lo;
            ai->shadow_size_hi = hi;

            ai->par_size += lo + hi;
          }
        } break;
      case _XCALABLEMP_N_SHADOW_FULL:
        ai->par_size = ai->ser_size;
        break;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
}

// FIXME not consider full shadow
void _XCALABLEMP_pack_shadow_NORMAL_2_BASIC(void **lo_buffer, void **hi_buffer, void *array_addr,
                                            _XCALABLEMP_array_t *array_desc, int array_index, int array_type) {
  int array_dim = array_desc->dim;
  _XCALABLEMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XCALABLEMP_template_chunk_t *ti = ai->align_template_chunk;

  int lower0, upper0, stride0, dim_acc0,
      lower1, upper1, stride1;

  // pack lo shadow
  if (ai->shadow_size_lo > 0) {
    if (array_index == 0) {
      lower0 = 0;
      upper0 = 0;
      stride0 = 0;
      dim_acc0 = 0;

      lower1 = 0;
      upper1 = 0;
      stride1 = 0;
    }
    else { // array_index == 1
      lower0 = 0;
      upper0 = 0;
      stride0 = 0;
      dim_acc0 = 0;

      lower1 = 0;
      upper1 = 0;
      stride1 = 0;
    }

//    _XCALABLEMP_pack_array_2_DOUBLE(*lo_buffer, array_addr,
//                                    lower0, upper0, stride0, dim_acc0,
//                                    lower1, upper1, stride1);
  }

  // pack hi shadow
  if (ai->shadow_size_hi > 0) {
    if (array_index == 0) {
      lower0 = 0;
      upper0 = 0;
      stride0 = 0;
      dim_acc0 = 0;

      lower1 = 0;
      upper1 = 0;
      stride1 = 0;
    }
    else { // array_index == 1
      lower0 = 0;
      upper0 = 0;
      stride0 = 0;
      dim_acc0 = 0;

      lower1 = 0;
      upper1 = 0;
      stride1 = 0;
    }

//    _XCALABLEMP_pack_array_2_DOUBLE(*hi_buffer, array_addr,
//                                    lower0, upper0, stride0, dim_acc0,
//                                    lower1, upper1, stride1);
  }
}
