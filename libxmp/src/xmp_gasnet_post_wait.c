#include "xmp_internal.h"
#include "xmp_atomic.h"

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
static int _mynode;

void _xmp_gasnet_post_wait_initialize()
{
  gasnet_hsl_init(&pw.hsl);
  pw.wait_num            = 0;
  pw.list                = malloc(sizeof(request_list_t) * _XMP_POST_WAIT_QUEUESIZE);
  pw.list_size           = _XMP_POST_WAIT_QUEUESIZE;
  _mynode                = gasnet_mynode();
}

static void _xmp_pw_push(const int node, const int tag)
{
  pw.list[pw.wait_num].node = node;
  pw.list[pw.wait_num].tag  = tag;
  pw.wait_num++;
}

static void _xmp_gasnet_do_post(const int node, const int tag)
{
  gasnet_hsl_lock(&pw.hsl);
  if(pw.list_size == pw.wait_num){
    request_list_t *old_list = pw.list;
    pw.list_size += _XMP_POST_WAIT_QUEUECHUNK;
    pw.list = malloc(sizeof(request_list_t) * pw.list_size);
    memcpy(pw.list, old_list, sizeof(request_list_t) * pw.wait_num);
    free(old_list);
  }
  _xmp_pw_push(node, tag);
  gasnet_hsl_unlock(&pw.hsl);
}

void _xmp_gasnet_post_request(gasnet_token_t token, const int node, const int tag)
{
  _xmp_gasnet_do_post(node, tag);
}

void _xmp_gasnet_post(const int target_node, const int tag)
{
  if(target_node == _mynode){
    _xmp_gasnet_do_post(_mynode, tag);
  } else{
    gasnet_AMRequestShort2(target_node, _XMP_GASNET_POST_REQUEST, _mynode, tag);
  }
}

static void _xmp_pw_cutdown(const int index)
{
  if(index != pw.wait_num-1){  // Not tail index
    for(int i=index+1;i<pw.wait_num;i++){
      pw.list[i-1] = pw.list[i];
    }
  }
  pw.wait_num--;
}

static int _xmp_pw_remove_anonymous()
{
  if(pw.wait_num > 0){
    pw.wait_num--;
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

static int _xmp_pw_remove_notag(const int node)
{
  for(int i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static int _xmp_pw_remove(const int node, const int tag)
{
  for(int i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node && tag == pw.list[i].tag){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void _xmp_gasnet_wait()
{
  GASNET_BLOCKUNTIL(_xmp_pw_remove_anonymous());
}

void _xmp_gasnet_wait_tag(const int node, const int tag)
{
  GASNET_BLOCKUNTIL(_xmp_pw_remove(node, tag));
}

void _xmp_gasnet_wait_notag(const int node)
{
  GASNET_BLOCKUNTIL(_xmp_pw_remove_notag(node));
}
