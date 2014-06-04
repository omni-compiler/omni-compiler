#include "xmp_internal.h"

void _XMP_wait()
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait();
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_tag_1(_XMP_nodes_t *node_desc, int num1, int tag)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(num1-1, tag);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_notag_1(_XMP_nodes_t *node_desc, int num1)
{
#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(num1-1);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_tag_2(_XMP_nodes_t *node_desc, int num1, int num2, int tag)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_notag_2(_XMP_nodes_t *node_desc, int num1, int num2)
{
  int target = num1-1 + (num2-1)*node_desc->info[0].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_tag_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int tag)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_notag_3(_XMP_nodes_t *node_desc, int num1, int num2, int num3)
{
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_tag_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4, int tag){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_tag(target, tag);
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

void _XMP_wait_notag_4(_XMP_nodes_t *node_desc, int num1, int num2, int num3, int num4){
  int target = num1-1;
  target += (num2-1)*node_desc->info[0].size;
  target += (num3-1)*node_desc->info[0].size*node_desc->info[1].size;
  target += (num4-1)*node_desc->info[0].size*node_desc->info[1].size*node_desc->info[2].size;

#ifdef _XMP_COARRAY_GASNET
  _xmp_gasnet_wait_notag(target);
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
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
#else
  _XMP_fatal("Cannot use wait function");
#endif
}

#ifdef _AA
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
#endif
