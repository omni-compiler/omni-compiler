#include "xmp_internal.h"

void _XMP_wait()
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait();
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait();
#endif
}

void _XMP_wait_tag_1(_XMP_nodes_t *node_desc, int num1, int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(num1-1, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(num1-1, tag);
#endif
}

void _XMP_wait_notag_1(_XMP_nodes_t *node_desc, int num1)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(num1-1);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(num1-1);
#endif
}

void _XMP_wait_tag_2(_XMP_nodes_t *node_desc, int num1, int num2, int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_2(_XMP_nodes_t *node_desc, int num1, int num2)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int tag){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_5(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
		     int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_5(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_6(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
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
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_6(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5,
		       int num6)
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
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_7(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5, 
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
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_7(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int num5,
		       int num6, int num7)
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
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}
