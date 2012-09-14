#include "xmp_internal.h"
#include "xmp_atomic.h"
#define _XMP_POST_WAIT_CHUNK 16
#define _XMP_POST_WAIT_DELETED_NODE -1

typedef struct request_list{
  int node;
  int tag;
} request_list_t;

typedef struct post_wait_obj{
  gasnet_hsl_t    hsl;
  int             request_num;  /* How many requests form post node are stored */
  request_list_t  *list;
  int             list_size;
  int             *deleted_node_list;
  int             deleted_node_index;
} post_wait_obj_t;

static post_wait_obj_t pw;

void _xmp_gasnet_post_initialize(){
  gasnet_hsl_init(pw.hsl);
  pw.request_num         = 0;
  pw.list                = malloc(sizeof(request_list_t) * _XMP_POST_WAIT_CHUNK);
  pw.list_size           = _XMP_POST_WAIT_CHUNK;
  pw.deleted_node_list   = malloc(sizeof(int) * _XMP_POST_WAIT_CHUNK);
  pw.deleted_node_index  = 0;
}

static void _xmp_gasnet_do_post(int node, int tag){
  gasnet_hsl_lock(pw.hsl);
  if(pw.deleted_node_index == 0){
    pw.list[0].node = node;
    pw.list[0].tag = tag;
  } else{
    if(pw.list_size < pw.deleted_node_index){
      request_list_t *old_list = pw.list;
      pw.list_size += _XMP_POST_WAIT_CHUNK;
      pw.list = malloc(sizeof(request_list_t) * pw.list_size);
      memcpy(pw.list, old_list, sizeof(request_list_t) * pw.list_size);
      free(old_list);
    }
    pw.deleted_node_index--;
    pw.list[pw.deleted_node_index].node = node;
    pw.list[pw.deleted_node_index].tag  = tag;
  }
  gasnet_hsl_unlock(pw.hsl);
}

void _xmp_gasnet_post_request(gasnet_token_t token, int node, int tag){
  _xmp_gasnet_do_post(node, tag);
}

void _xmp_gasnet_post(int target_node, int tag){
  target_node -= 1;   // for 1-origin in XMP

  if(target_node >= gasnet_nodes()){
    _XMP_fatal("xmp_gasnet_post.c : Target Node ID is illegal.");
  }

  int mynode = (int)gasnet_mynode();
  
  if(target_node == mynode){
    _xmp_gasnet_do_post(mynode, tag);
  } else{
    gasnet_AMRequestShort2(target_node, _XMP_GASNET_POST_REQUEST, mynode, tag);
  }
}

static void _xmp_gasnet_do_wait(int node, int tag){}
void _xmp_gasnet_wait_request(gasnet_token_t token, int node, int tag){
  _xmp_gasnet_do_wait(node, tag);
}

void _xmp_gasnet_wait(int num, ...){
  int node, tag;
  va_list args;

  va_start(args, num);
  switch (num) {
  case 0:
    printf("Wait 0\n");
    break;
  case 1:
    node = va_arg(args, int);
    printf("Wait 1 %d\n", node);
    break;
  case 2:
    node = va_arg(args, int);
    tag  = va_arg(args, int);
    printf("Wait 2 %d %d\n", node, tag);
    break;
  }
  va_end(args);

}
