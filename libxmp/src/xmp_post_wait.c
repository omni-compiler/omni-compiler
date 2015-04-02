/**
 * Wrapper functions for post/wait
 *
 * @file
 */
#include "xmp_internal.h"

void _XMP_post_wait_initialize()
{
#ifdef _XMP_GASNET
  _xmp_gasnet_post_wait_initialize();
#elif _XMP_FJRDMA
  _xmp_fjrdma_post_wait_initialize();
#else
  _XMP_fatal("Cannot use post function");
#endif
}

void _XMP_post_1(const _XMP_nodes_t *node_desc, const int num1, const int tag)
{
  int node_num = num1-1;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_2(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int tag)
{
  int node_num = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_post_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int num7, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;
  node_num += (num7-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size * node_desc->info[5].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_post(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_post(node_num, tag);
#endif
}

void _XMP_wait_noargs()
{
#ifdef _XMP_GASNET
  _xmp_gasnet_wait_noargs();
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_noargs();
#endif
}

void _XMP_wait_1(const _XMP_nodes_t *node_desc, const int num1, const int tag)
{
  int node_num = num1-1;
#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_1(const _XMP_nodes_t *node_desc, const int num1)
{
  int node_num = num1-1;
#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_2(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int tag)
{
  int node_num = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_2(const _XMP_nodes_t *node_desc, const int num1, const int num2)
{
  int node_num = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_3(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_4(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  
#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_5(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_6(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5,
		      const int num6)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}

void _XMP_wait_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5, 
		 const int num6, const int num7, const int tag)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;
  node_num += (num7-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size * node_desc->info[5].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait(node_num, tag);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait(node_num, tag);
#endif
}

void _XMP_wait_node_7(const _XMP_nodes_t *node_desc, const int num1, const int num2, const int num3, const int num4, const int num5,
		      const int num6, const int num7)
{
  int node_num = num1-1;
  node_num += (num2-1)*node_desc->info[0].size;
  node_num += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  node_num += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;
  node_num += (num5-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size;
  node_num += (num6-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size;
  node_num += (num7-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size*
    node_desc->info[3].size * node_desc->info[4].size * node_desc->info[5].size;

#ifdef _XMP_GASNET
  _xmp_gasnet_wait_node(node_num);
#elif _XMP_FJRDMA
  _xmp_fjrdma_wait_node(node_num);
#endif
}
