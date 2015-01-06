#include "xmp_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct post_request_info{
  int node;
  int tag;
} post_request_info_t;

typedef struct post_request{
  int                 num;      /* How many post requests are in table */
  int                 max_size; /* Max size of table */
  post_request_info_t *table;
} post_request_t;

static uint64_t _local_rdma_addr, *_remote_rdma_addr;
static post_request_t _post_request;

void _xmp_fjrdma_post_wait_initialize()
{
  _post_request.num      = 0;
  _post_request.max_size = _XMP_POST_REQUEST_INITIAL_TABLE_SIZE;
  _post_request.table    = malloc(sizeof(post_request_info_t) * _post_request.max_size);
  
  double *_token    = _XMP_alloc(sizeof(double));
  _local_rdma_addr  = FJMPI_Rdma_reg_mem(_XMP_POST_REQUEST_ID, _token, sizeof(double));
  _remote_rdma_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);

  // Reduce network overload by Fujitsu (This process is temporal)
  for(int ncount=0,i=1; i<_XMP_world_size; ncount++,i++){
    int partner_rank = (_XMP_world_rank+i)%_XMP_world_size;
    if(partner_rank != _XMP_world_rank)
      while((_remote_rdma_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _XMP_POST_REQUEST_ID)) == FJMPI_RDMA_ERROR);

    if(ncount >= 3000){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }
  // (end of temporal process)
}

static void add_request(const int node, const int tag)
{
  if(_post_request.num == _post_request.max_size){  // If table is full
    post_request_info_t *old_table = _post_request.table;
    _post_request.max_size += _XMP_POST_REQUEST_INCREMENT_TABLE_SIZE;
    _post_request.table = malloc(sizeof(post_request_info_t) * _post_request.max_size);
    memcpy(_post_request.table, old_table, sizeof(post_request_info_t) * _post_request.num);
    free(old_table);
  }
  
  _post_request.table[_post_request.num].node = node;
  _post_request.table[_post_request.num].tag  = tag;
  _post_request.num++;
}

void _xmp_fjrdma_post(const int node, const int tag)
{
  if(tag < 0 || tag > 14){
    fprintf(stderr, "tag is %d : On the K computer or FX10, 0 <= tag <= 14\n", tag);
    _XMP_fatal_nomsg();
  }

  if(node == _XMP_world_rank){
    add_request(node, tag);
  }
  else{
    FJMPI_Rdma_put(node, tag, _remote_rdma_addr[node], _local_rdma_addr, sizeof(double), _XMP_POST_REQUEST_NIC_FLAG);
    struct FJMPI_Rdma_cq cq;
    while(FJMPI_Rdma_poll_cq(_XMP_POST_REQUEST_SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
  }
}

static void shrink_table(const int index)
{
  if(index != _post_request.num-1){  // Not last request
    for(int i=index+1;i<_post_request.num;i++){
      _post_request.table[i-1] = _post_request.table[i];
    }
  }
  _post_request.num--;
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

void _xmp_fjrdma_wait_tag(const int node, const int tag)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    // If the post request with the node and the tag is not in table, return false;
    int is_in_table = remove_request(node, tag);
    if(is_in_table) break;

    if(FJMPI_Rdma_poll_cq(_XMP_POST_REQUEST_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_request(cq.pid, cq.tag);
  }
}

void _xmp_fjrdma_wait_notag(const int node)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    // If the post request with the node is not in table, return false;
    int is_in_table = remove_request_notag(node);
    if(is_in_table) break;
    
    if(FJMPI_Rdma_poll_cq(_XMP_POST_REQUEST_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_request(cq.pid, cq.tag);
  }
}

void _xmp_fjrdma_wait()
{
  if(_post_request.num == 0){
    struct FJMPI_Rdma_cq cq;
    while(FJMPI_Rdma_poll_cq(_XMP_POST_REQUEST_RECV_NIC, &cq) != FJMPI_RDMA_HALFWAY_NOTICE);
  }
  else
    _post_request.num--;
}
