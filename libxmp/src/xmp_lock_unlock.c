#include "xmp_internal.h"

void _xmp_lock(_XMP_coarray_t* c, int position, int target_node){
#ifdef _XMP_GASNET
  _xmp_gasnet_lock(c, position, target_node);
#else
  _XMP_fatal("Cannt use lock Function");
#endif
}

void _xmp_unlock(_XMP_coarray_t* c, int position, int target_node){
#ifdef _XMP_GASNET
  _xmp_gasnet_unlock(c, position, target_node);
#else
  _XMP_fatal("Cannt use lock Function");
#endif
}

void _xmp_lock_initialize(_xmp_lock_t* lock, int number_of_elements){
#ifdef _XMP_GASNET
  _xmp_gasnet_lock_initialize(lock, number_of_elements);
#else
  _XMP_fatal("Cannt use lock Function");
#endif
}
