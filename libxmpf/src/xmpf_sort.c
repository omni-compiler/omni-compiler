#include "xmp_internal.h"

void xmp_sort_up_(_XMP_array_t **a_desc, _XMP_array_t **b_desc){
  _XMP_sort(*a_desc, *b_desc, 1);
}

void xmp_sort_down_(_XMP_array_t **a_desc, _XMP_array_t **b_desc){
  _XMP_sort(*a_desc, *b_desc, 0);
}
