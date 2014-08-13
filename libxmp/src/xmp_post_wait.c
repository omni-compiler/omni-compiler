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

void _XMP_post_1(const _XMP_nodes_t *node_desc, const int num1, const int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(num1-1, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(num1-1, tag);
#endif
}

void _XMP_post_2(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  target += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_post(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int tag)
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
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

void _XMP_post_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int num7, const int tag)
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
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_post(target, tag);
#endif
}

/*****************************************/
void _XMP_wait()
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait();
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait();
#endif
}

void _XMP_wait_tag_1(const _XMP_nodes_t *node_desc, const int num1, const int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(num1-1, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(num1-1, tag);
#endif
}

void _XMP_wait_notag_1(const _XMP_nodes_t *node_desc, const int num1)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(num1-1);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(num1-1);
#endif
}

void _XMP_wait_tag_2(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_tag(target, tag);
#endif
}

void _XMP_wait_notag_2(const _XMP_nodes_t *node_desc, const int num1, const int num2)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#elif _XMP_COARRAY_FJRDMA
  _xmp_fjrdma_wait_notag(target);
#endif
}

void _XMP_wait_tag_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int tag)
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

void _XMP_wait_notag_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3)
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

void _XMP_wait_tag_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int tag)
{
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

void _XMP_wait_notag_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4)
{
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

void _XMP_wait_tag_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		     const int tag)
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

void _XMP_wait_notag_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5)
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

void _XMP_wait_tag_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		     const int num6, const int tag)
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

void _XMP_wait_notag_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5,
		       const int num6)
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

void _XMP_wait_tag_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		     const int num6, const int num7, const int tag)
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

void _XMP_wait_notag_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5,
		       const int num6, const int num7)
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
