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
  _postreq.max_size = _XMP_POSTREQ_TABLE_INITIAL_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);
  
  double *token    = _XMP_alloc(sizeof(double));
  _local_rdma_addr  = FJMPI_Rdma_reg_mem(_XMP_POSTREQ_ID, token, sizeof(double));
  _remote_rdma_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);

  // Obtain remote RDMA addresses
  MPI_Barrier(MPI_COMM_WORLD);
  for(int ncount=0,i=1; i<_XMP_world_size+1; ncount++,i++){
    int partner_rank = (_XMP_world_rank + _XMP_world_size - i) % _XMP_world_size;
    if(partner_rank == _XMP_world_rank)
      _remote_rdma_addr[partner_rank] = _local_rdma_addr;
    else
      _remote_rdma_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _XMP_POSTREQ_ID);

    if(ncount > _XMP_FJRDMA_INTERVAL){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }

}

static void add_postreq(const int node, const int tag)
{
  if(_postreq.num == _postreq.max_size){  // If table is full
    _postreq.max_size *= _XMP_POSTREQ_TABLE_INCREMENT_RATIO;
    size_t next_size = sizeof(_XMP_postreq_info_t) * _postreq.max_size;
    _XMP_postreq_info_t *tmp;
    if((tmp = realloc(_postreq.table, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _postreq.table = tmp;
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
  // tag = 14(_XMP_FJRDMA_SYNC_IMAGES_TAG) is used in xmp_coarray_fjrdma.c for xmp_sync_images()
  if(tag < 0 || tag > 13){
    fprintf(stderr, "tag is %d : On the K computer or FX10, 0 <= tag <= 13\n", tag);
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

/**
 * Post operation for sync_images
 *
 * @param[in] number of nodes
 * @param[in] node set
 */
void _xmp_fjrdma_post_sync_images(const int num, const int node_set[num])
{
  int tag = _XMP_FJRDMA_SYNC_IMAGES_TAG;
  int num_of_puts = 0;
  struct FJMPI_Rdma_cq cq;

  for(int i=0;i<num;i++){
    int target = node_set[i];
    if(target == _XMP_world_rank){
      add_postreq(target, tag);
    }
    else{
      FJMPI_Rdma_put(target, tag, _remote_rdma_addr[target], _local_rdma_addr, sizeof(double), _XMP_POSTREQ_NIC_FLAG);
      num_of_puts++;
    }
  }

  for(int i=0;i<num_of_puts;i++)
    while(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
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

// If the table does not have the post request, return false
static bool remove_postreq(const int node, const int tag)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(_postreq.table[i].node == node && _postreq.table[i].tag == tag && _postreq.table[i].tag != _XMP_FJRDMA_SYNC_IMAGES_TAG){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

// If the table does not have the post request, return false
static bool remove_postreq_node(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(_postreq.table[i].node == node && _postreq.table[i].tag != _XMP_FJRDMA_SYNC_IMAGES_TAG){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

// If the table does not have the post request, return false
static bool remove_postreq_noargs()
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(_postreq.table[i].tag != _XMP_FJRDMA_SYNC_IMAGES_TAG){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

// If the table does not have the post request, return false
static bool remove_postreq_sync_images(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(_postreq.table[i].node == node && _postreq.table[i].tag == _XMP_FJRDMA_SYNC_IMAGES_TAG){
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
  if(tag < 0 || tag > 13){
    fprintf(stderr, "tag is %d : On the K computer or FX10, 0 <= tag <= 13\n", tag);
    _XMP_fatal_nomsg();
  }

  struct FJMPI_Rdma_cq cq;

  while(1){
    bool table_has_postreq = remove_postreq(node, tag);
    if(table_has_postreq) break;

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
    bool table_has_postreq = remove_postreq_node(node);
    if(table_has_postreq) break;
    
    if(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_postreq(cq.pid, cq.tag);
  }
}

/**
 * Wait operation without node-ref and tag
 */
void _xmp_fjrdma_wait_noargs()
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    bool table_has_postreq = remove_postreq_noargs();
    if(table_has_postreq) break;

    if(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      add_postreq(cq.pid, cq.tag);
  }
}

/**
 * Wait operation for sync_images
 *
 * @param[in] number of nodes
 * @param[in] node set
 */
void _xmp_fjrdma_wait_sync_images(const int num, const int node_set[num])
{
  struct FJMPI_Rdma_cq cq;

  for(int i=0;i<num;i++){
    int target = node_set[i];
    while(1){
      bool table_has_postreq = remove_postreq_sync_images(target);
      if(table_has_postreq) break;
      
      if(FJMPI_Rdma_poll_cq(_XMP_POSTREQ_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
	add_postreq(cq.pid, cq.tag);
    }
  }
}
