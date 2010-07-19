#include "xmp_internal.h"

void _XCALABLEMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, size_t type_size, _XCALABLEMP_array_t *array, ...) {
  if (array == NULL) return;
}

_Bool _XCALABLEMP_gmove_exec_home_SCALAR(_XCALABLEMP_array_t *array, ...) {
  return true;
}

void _XCALABLEMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, size_t type_size,
                                       _XCALABLEMP_array_t *dst_array, _XCALABLEMP_array_t *src_array, ...) {
  // isend
  if (src_array == NULL) return;

  // recv
  if (dst_array == NULL) return;
}
