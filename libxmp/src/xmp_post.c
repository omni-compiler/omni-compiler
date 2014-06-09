#include "xmp_internal.h"

void _XMP_post_wait_initialize()
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post_wait_initialize();
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post_wait_initialize();
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_1(_XMP_nodes_t *node_desc, int num1, int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(node_desc->comm_rank, num1-1, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(num1-1, tag);
#endif
}

void _XMP_post_2(_XMP_nodes_t *node_desc, int num1, int num2, int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int tag){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
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
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
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
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
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
  _xmp_gasnet_post(node_desc->comm_rank, target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}
