/**
 * Post/wait functions using GASNet
 *
 * @file
 */
#include "xmp_internal.h"

/* postreq = post request */
typedef struct _XMP_postreq_info{
  int node;
  int tag;
} _XMP_postreq_info_t;

typedef struct _XMP_postreq{
  _XMP_postreq_info_t *table;   /**< Table for post requests */
  int                 num;      /**< How many post requests are in table */
  int                 max_size; /**< Max size of table */
  gasnet_hsl_t        hsl;      /**< Lock object for GASNet */
} _XMP_postreq_t;

static _XMP_postreq_t _postreq;

/**
 * Initialize environment for post/wait directives
 */
void _xmp_gasnet_post_wait_initialize()
{
  gasnet_hsl_init(&_postreq.hsl);
  _postreq.num      = 0;
  _postreq.max_size = _XMP_POSTREQ_TABLE_INITIAL_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);
}

static void add_request(const int node, const int tag)
{
  _postreq.table[_postreq.num].node = node;
  _postreq.table[_postreq.num].tag  = tag;
  _postreq.num++;
}

static void do_post(const int node, const int tag)
{
  gasnet_hsl_lock(&_postreq.hsl);
  if(_postreq.num == _postreq.max_size){
    _postreq.max_size *= _XMP_POSTREQ_TABLE_INCREMENT_RATIO;
    size_t next_size = sizeof(_XMP_postreq_info_t) * _postreq.max_size;
    _XMP_postreq_info_t *tmp;
    if((tmp = realloc(_postreq.table, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _postreq.table = tmp;
  }
  add_request(node, tag);
  gasnet_hsl_unlock(&_postreq.hsl);
}

void _xmp_gasnet_postreq(gasnet_token_t token, const int node, const int tag)
{
  do_post(node, tag);
}

/**
 * Post operation
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_gasnet_post(const int node, const int tag)
{
  _XMP_gasnet_sync_memory();
  
  if(node == _XMP_world_rank){
    do_post(_XMP_world_rank, tag);
  } else{
    gasnet_AMRequestShort2(node, _XMP_GASNET_POSTREQ, _XMP_world_rank, tag);
  }
}

static void shift_postreq(const int index)
{
  if(index != _postreq.num-1){  // Not tail index
    for(int i=index+1;i<_postreq.num;i++){
      _postreq.table[i-1] = _postreq.table[i];
    }
  }
  _postreq.num--;
}

static bool remove_request_noargs()
{
  if(_postreq.num > 0){
    _postreq.num--;
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

static bool remove_request_node(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static bool remove_request(const int node, const int tag)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node && tag == _postreq.table[i].tag){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

/**
 * Wait operation without node-ref and tag
 */
void _xmp_gasnet_wait_noargs()
{
  _XMP_gasnet_sync_memory();
  GASNET_BLOCKUNTIL(remove_request_noargs());
}

/**
 * Wait operation with node-ref
 *
 * @param[in] node node number
 */
void _xmp_gasnet_wait_node(const int node)
{
  _XMP_gasnet_sync_memory();
  GASNET_BLOCKUNTIL(remove_request_node(node));
}

/**
 * Wait operation with node-ref and tag
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_gasnet_wait(const int node, const int tag)
{
  _XMP_gasnet_sync_memory();
  GASNET_BLOCKUNTIL(remove_request(node, tag));
}


