/**
 * Post/wait functions using Fujitsu RDMA
 *
 * @file
 */
#include "xmp_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* postreq = post request */
typedef struct _XMP_postreq_info{
  int node;
  int tag;
} _XMP_postreq_info_t;

typedef struct _XMP_postreq{
  _XMP_postreq_info_t *table;   /**< Table for post requests */
  int                 num;      /**< How many post requests are in table */
  int                 max_size; /**< Max size of table */
} _XMP_postreq_t;

/**
 * Address for using RDMA
 */
static uint64_t _local_rdma_addr, *_remote_rdma_addr;
static _XMP_postreq_t _postreq;

/**
 * Initialize environment for post/wait directives
 */
void _xmp_fjrdma_post_wait_initialize()
{
  _postreq.num      = 0;
  _postreq.max_size = _XMP_POSTREQ_INITIAL_TABLE_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);
  
  double *token    = _XMP_alloc(sizeof(double));
  _local_rdma_addr  = FJMPI_Rdma_reg_mem(_XMP_POSTREQ_ID, token, sizeof(double));
  _remote_rdma_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);

  // Obtain remote RDMA addresses
  // Reduce network overload by Fujitsu (This process is temporal)
  for(int ncount=0,i=1; i<_XMP_world_size; ncount++,i++){
    int partner_rank = (_XMP_world_rank+i)%_XMP_world_size;
    if(partner_rank != _XMP_world_rank)
      while((_remote_rdma_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _XMP_POSTREQ_ID)) == FJMPI_RDMA_ERROR);

    if(ncount >= 3000){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }
  // (end of temporal process)
}

static void add_postreq(const int node, const int tag)
{
  if(_postreq.num == _postreq.max_size){  // If table is full
    _XMP_postreq_info_t *old_table = _postreq.table;
    _postreq.max_size += _XMP_POSTREQ_INCREMENT_TABLE_SIZE;
    _postreq.table = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);
    memcpy(_postreq.table, old_table, sizeof(_XMP_postreq_info_t) * _postreq.num);
    free(old_table);
  }
  
  _postreq.table[_postreq.num].node = node;
  _postreq.table[_postreq.num].tag  = tag;
  _postreq.num++;
}

/**
 * Post operation
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_fjrdma_post(const int node, const int tag)
{
  if(tag < 0 || tag > 14){
    fprintf(stderr, "tag is %d : On the K computer or FX10, 0 <= tag <= 14\n", tag);
    _XMP_fatal_nomsg();
  }

  if(node == _XMP_world_rank){
    add_postreq(node, tag);
  }
  else{
    FJMPI_Rdma_put(node, tag, _remote_rdma_addr[node], _local_rdma_addr, sizeof(double), _XMP_POSTREQ_NIC_FLAG);
    struct FJMPI_Rdma_cq cq;
    while(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
  }
}

static void shift_postreq(const int index)
{
  if(index != _postreq.num-1){  // Not last request
    for(int i=index+1;i<_postreq.num;i++){
      _postreq.table[i-1] = _postreq.table[i];
    }
  }
  _postreq.num--;
}

static bool remove_postreq_node(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static bool remove_postreq(const int node, const int tag)
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
 * Wait operation with node-ref and tag
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_fjrdma_wait(const int node, const int tag)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    bool has_table = remove_postreq(node, tag); // If the table does not have the post request, return false;
    if(has_table) break;

    if(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_postreq(cq.pid, cq.tag);
  }
}

/**
 * Wait operation with only node-ref
 *
 * @param[in] node node number
 */
void _xmp_fjrdma_wait_node(const int node)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    bool has_table = remove_postreq_node(node); // If the table does not have the post request, return false;
    if(has_table) break;
    
    if(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_postreq(cq.pid, cq.tag);
  }
}

/**
 * Wait operation without node-ref and tag
 */
void _xmp_fjrdma_wait_noargs()
{
  if(_postreq.num == 0){
    struct FJMPI_Rdma_cq cq;
    while(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) != FJMPI_RDMA_HALFWAY_NOTICE);
  }
  else
    _postreq.num--;
}
