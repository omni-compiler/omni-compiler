#include "xmp_internal.h"
#include "xmp_atomic.h"
#define _XMP_POST_WAIT_QUEUESIZE 32
#define _XMP_POST_WAIT_QUEUECHUNK 512

typedef struct request_list{
  int node;
  int tag;
} request_list_t;

typedef struct post_wait_obj{
  gasnet_hsl_t    hsl;
  int             wait_num;  /* How many requests form post node are waited */
  request_list_t  *list;
  int             list_size;
} post_wait_obj_t;

static post_wait_obj_t pw;

void _xmp_gasnet_post_initialize(){
  gasnet_hsl_init(&pw.hsl);
  pw.wait_num            = 0;
  pw.list                = malloc(sizeof(request_list_t) * _XMP_POST_WAIT_QUEUESIZE);
  pw.list_size           = _XMP_POST_WAIT_QUEUESIZE;
}

static void _xmp_pw_push(int node, int tag){
  pw.list[pw.wait_num].node = node;
  pw.list[pw.wait_num].tag  = tag;
  pw.wait_num++;
}

static void _xmp_gasnet_do_post(int node, int tag){
  gasnet_hsl_lock(&pw.hsl);
  if(pw.wait_num == 0){
    _xmp_pw_push(node, tag);
  } 
  else if(pw.wait_num < 0){ // This statement does not executed.
    _XMP_fatal("xmp_gasnet_do_post() : Variable pw.wait_num is illegal.");
  }
  else{  // pw.wait_num > 0
    if(pw.list_size == pw.wait_num){
      request_list_t *old_list = pw.list;
      pw.list_size += _XMP_POST_WAIT_QUEUECHUNK;
      pw.list = malloc(sizeof(request_list_t) * pw.list_size);
      memcpy(pw.list, old_list, sizeof(request_list_t) * pw.wait_num);
      free(old_list);
    }
    else if(pw.list_size < pw.wait_num){  // This statement does not executed.
      _XMP_fatal("xmp_gasnet_do_post() : Variable pw.wait_num is illegal.");
    }
    _xmp_pw_push(node, tag);
  }
  gasnet_hsl_unlock(&pw.hsl);
}

void _xmp_gasnet_post_request(gasnet_token_t token, int node, int tag){
  _xmp_gasnet_do_post(node, tag);
}

void _xmp_gasnet_post(int target_node, int tag){
  target_node -= 1;   // for 1-origin in XMP

  int mynode = (int)gasnet_mynode();
  
  if(target_node == mynode){
    _xmp_gasnet_do_post(mynode, tag);
  } else{
    gasnet_AMRequestShort2(target_node, _XMP_GASNET_POST_REQUEST, mynode, tag);
  }
}

void _xmp_pw_cutdown(int index){
  if(index != pw.wait_num-1){  // Not tail index
    int i;
    for(i=index+1;i<pw.wait_num;i++){
      pw.list[i-1] = pw.list[i];
    }
  }
  pw.wait_num--;
}

static int _xmp_pw_remove_anonymous(){
  if(pw.wait_num > 0){
    pw.wait_num--;
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

static int _xmp_pw_remove_notag(int node){
  int i;
  for(i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static int _xmp_pw_remove(int node, int tag){
  int i;
  for(i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node && tag == pw.list[i].tag){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void _xmp_gasnet_wait(int num, ...){
  int target_node, tag;
  va_list args;
  
  va_start(args, num);
  switch (num) {
  case 0:
    GASNET_BLOCKUNTIL(_xmp_pw_remove_anonymous());
    break;
  case 1:
    target_node = va_arg(args, int) - 1;  // for 1-origin in XMP
    GASNET_BLOCKUNTIL(_xmp_pw_remove_notag(target_node));
    break;
  case 2:
    target_node = va_arg(args, int) - 1;  // for 1-origin in XMP
    tag  = va_arg(args, int);
    GASNET_BLOCKUNTIL(_xmp_pw_remove(target_node, tag));
    break;
  }
  va_end(args);
}
