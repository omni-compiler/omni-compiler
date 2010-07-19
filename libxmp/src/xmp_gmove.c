#include <stdarg.h>
#include "xmp_internal.h"

void _XCALABLEMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, size_t type_size, _XCALABLEMP_array_t *array, ...) {
  if (array == NULL) return;

  _XCALABLEMP_template_t *template = array->align_template;
  if (template == NULL)
    _XCALABLEMP_fatal("null template descriptor detected");

  _XCALABLEMP_nodes_t *nodes = template->onto_nodes;
  if (nodes == NULL)
    _XCALABLEMP_fatal("null nodes descriptor detected");

  int array_dim = array->dim;
  int template_dim = template->dim;
  int nodes_dim = nodes->dim;

  int *src_nodes = _XCALABLEMP_alloc(sizeof(int) * nodes_dim);

  va_list args;
  va_start(args, array);
  for(int i = 0; i < array_dim; i++) {
    int index = va_arg(args, int);
  }
  va_end(args);
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
