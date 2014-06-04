#include "xmp_internal.h"

void _XMP_post_wait_initialize()
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post_wait_initialize();
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_1(_XMP_nodes_t *node_desc, int num1, int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(num1-1, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_2(_XMP_nodes_t *node_desc, int num1, int num2, int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int tag){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_5(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
		 int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_6(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
		 int num6, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  target += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_7(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
		 int num6, int num7, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  target += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;
  target += (num7-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size * node_desc->info[5].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#else
  _XMP_fatal("Cannot use post function");
#endif
}

#ifdef _AA
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
#endif


