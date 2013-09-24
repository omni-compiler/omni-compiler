#include "xmp_internal.h"

void _XMP_post_initialize(){
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post_initialize();
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post(_XMP_nodes_t *node_desc, int dims, ...){
  int i, j;
  int target_node = 0, node_distance[dims], tag;

  for(i=0;i<dims;i++){
    node_distance[i] = 1;
    for(j=0;j<i;j++){
      node_distance[i] *= node_desc->info[j].size;
    }
  }

  va_list args;
  va_start( args, dims );
  
  for(i=0;i<dims;i++)  // target_node is rank
    target_node += (va_arg(args, int) - 1) * node_distance[i];

  tag = va_arg(args, int);
  va_end( args );

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target_node, tag);
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
    node = va_arg(args, int) - 1;
    _xmp_gasnet_wait(num, node);
    break;
  case 2:
    node = va_arg(args, int) - 1;
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

