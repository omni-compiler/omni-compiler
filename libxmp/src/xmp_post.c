#include "xmp_internal.h"

void _XMP_post_initialize(){
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post_initialize();
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post(int node, int tag){
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(node, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_wait(int num, ...){
#ifdef _XMP_COARRAY_GASNET
  int node, tag;
  va_list args;

  va_start(args, num);
  switch (num){
  case 0:
    _xmp_gasnet_wait(num);
    break;
  case 1:
    node = va_arg(args, int);
    _xmp_gasnet_wait(num, node);
    break;
  case 2:
    node = va_arg(args, int);
    tag  = va_arg(args, int);
    _xmp_gasnet_wait(num, node, tag);
    break;
  default:
    _XMP_fatal("_XMP_wait Error");
    break;
  }
  va_end(args);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

