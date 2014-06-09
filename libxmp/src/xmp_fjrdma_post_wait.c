#include "xmp_internal.h"
#include "xmp_atomic.h"

typedef struct request_list{
  int node;
  int tag;
} request_list_t;

typedef struct post_wait_obj{
  int             wait_num;  /* How many requests form post node are waited */
  request_list_t  *list;
  int             list_size;
} post_wait_obj_t;

static post_wait_obj_t _pw;
static uint64_t *_each_addr, _laddr
static double *_token;
static struct FJMPI_Rdma_cq cq;

void _xmp_fjrdma_post_wait_initialize(){
  _pw.wait_num  = 0;
  _pw.list      = malloc(sizeof(request_list_t) * _XMP_POST_WAIT_QUEUESIZE);
  _pw.list_size = _XMP_POST_WAIT_QUEUESIZE;
  
  _each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  _token = _XMP_alloc(sizeof(double));
  _laddr = FJMPI_Rdma_reg_mem(POST_WAIT_ID, _token, sizeof(double));

  for(int i=0; i<_XMP_world_size; i++)
    if(i != _XMP_world_rank)
      while((_each_addr[i] = FJMPI_Rdma_get_remote_addr(i, POST_WAID_ID)) == FJMPI_RDMA_ERROR);

  // Memo: Reterun wrong local address by using FJMPI_Rdma_get_remote_addr.
  // So FJMPI_Rdma_reg_mem should be used.
  _each_addr[_XMP_world_rank] = _laddr;
}

static void _xmp_pw_push(int node, int tag){
  _pw.list[pw.wait_num].node = node;
  _pw.list[pw.wait_num].tag  = tag;
  _pw.wait_num++;
}

static void _xmp_fjrdma_do_post(int node, int tag)
{
  if(pw.list_size == pw.wait_num){
    request_list_t *old_list = _pw.list;
    _pw.list_size += _XMP_POST_WAIT_QUEUECHUNK;
    _pw.list = malloc(sizeof(request_list_t) * pw.list_size);
    memcpy(_pw.list, old_list, sizeof(request_list_t) * _pw.wait_num);
    free(old_list);
  }

  _xmp_pw_push(node, tag);
}

void _xmp_fjrdma_post(int target_node, int tag)
{
  if(tag < 0 || tag > 14)
    fprintf(stderr, "tag is %d : On the K computer or FX10, 0 <= tag && tag <= 14\n");

  if(target_node == _XMP_world_rank)
    _xmp_fjrdma_do_post(_XMP_world_rank, tag);
  else
    FJMPI_Rdma_put(target_node, tag, _each_addr[target_node], _laddr, sizeof(double), FLAG_NIC);
}

/********************/
static void _xmp_pw_cutdown(int index)
{
  if(index != pw.wait_num-1){  // Not tail index
    for(int i=index+1;i<pw.wait_num;i++){
      pw.list[i-1] = pw.list[i];
    }
  }
  pw.wait_num--;
}

static int _xmp_pw_remove_notag(int node)
{
  for(int i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static int _xmp_pw_remove(int node, int tag)
{
  for(int i=pw.wait_num-1;i>=0;i--){
    if(node == pw.list[i].node && tag == pw.list[i].tag){
      _xmp_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void _xmp_fjrdma_wait_tag(int node, int tag)
{
  while(1){
    while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
    _xmp_pw_push(cq.pid, cq.tag);
    if(_xmp_pw_remove(node, tag))
      break;
  }
}

void _xmp_fjrdma_wait_notag(int node)
{
  while(1){
    while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
    _xmp_pw_push(cq.pid, cq.tag);
    if(_xmp_pw_remove_notag(node))
      break;
  }
}

void _xmp_fjrdma_wait()
{
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
}
