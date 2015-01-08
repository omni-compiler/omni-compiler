#include "xmp_internal.h"
#include "xmp_atomic.h"

typedef struct _XMP_post_request_info{
  int node;
  int tag;
} _XMP_post_request_info_t;

typedef struct _XMP_post_request{
  int                      num;      /* How many post requests are in table */
  int                      max_size; /* Max size of table */
  _XMP_post_request_info_t *table;
  gasnet_hsl_t             hsl;
} _XMP_post_request_t;

static _XMP_post_request_t _post_request;

void _xmp_gasnet_post_wait_initialize()
{
  gasnet_hsl_init(&_post_request.hsl);
  _post_request.num      = 0;
  _post_request.max_size = _XMP_POST_REQUEST_INITIAL_TABLE_SIZE;
  _post_request.table    = malloc(sizeof(_XMP_post_request_info_t) * _post_request.max_size);
}

static void add_request(const int node, const int tag)
{
  _post_request.table[_post_request.num].node = node;
  _post_request.table[_post_request.num].tag  = tag;
  _post_request.num++;
}

static void do_post(const int node, const int tag)
{
  gasnet_hsl_lock(&_post_request.hsl);
  if(_post_request.num == _post_request.max_size){
    _XMP_post_request_info_t *old_table = _post_request.table;
    _post_request.max_size += _XMP_POST_REQUEST_INCREMENT_TABLE_SIZE;
    _post_request.table = malloc(sizeof(_XMP_post_request_info_t) * _post_request.max_size);
    memcpy(_post_request.table, old_table, sizeof(_XMP_post_request_info_t) * _post_request.num);
    free(old_table);
  }
  add_request(node, tag);
  gasnet_hsl_unlock(&_post_request.hsl);
}

void _xmp_gasnet_post_request(gasnet_token_t token, const int node, const int tag)
{
  do_post(node, tag);
}

void _xmp_gasnet_post(const int node, const int tag)
{
  if(node == _XMP_world_rank){
    do_post(_XMP_world_rank, tag);
  } else{
    gasnet_AMRequestShort2(node, _XMP_GASNET_POST_REQUEST, _XMP_world_rank, tag);
  }
}

static void shrink_table(const int index)
{
  if(index != _post_request.num-1){  // Not tail index
    for(int i=index+1;i<_post_request.num;i++){
      _post_request.table[i-1] = _post_request.table[i];
    }
  }
  _post_request.num--;
}

static int remove_request_anonymous()
{
  if(_post_request.num > 0){
    _post_request.num--;
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

static int remove_request_notag(const int node)
{
  for(int i=_post_request.num-1;i>=0;i--){
    if(node == _post_request.table[i].node){
      shrink_table(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static int remove_request(const int node, const int tag)
{
  for(int i=_post_request.num-1;i>=0;i--){
    if(node == _post_request.table[i].node && tag == _post_request.table[i].tag){
      shrink_table(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void _xmp_gasnet_wait()
{
  GASNET_BLOCKUNTIL(remove_request_anonymous());
}

void _xmp_gasnet_wait_notag(const int node)
{
  GASNET_BLOCKUNTIL(remove_request_notag(node));
}

void _xmp_gasnet_wait_tag(const int node, const int tag)
{
  GASNET_BLOCKUNTIL(remove_request(node, tag));
}

