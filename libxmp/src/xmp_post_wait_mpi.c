//#define DEBUG 1
#include <stdlib.h>
#include <pthread.h>
#include "xmp_internal.h"
#include <time.h>
#include <sys/time.h>

typedef struct _XMP_postreq_info{
  int node;
  int tag;
} _XMP_postreq_info_t;

typedef struct _XMP_postreq{
  _XMP_postreq_info_t *table;   /**< Table for post requests */
  int                 num;      /**< How many post requests are in table */
  int                 max_size; /**< Max size of table */
} _XMP_postreq_t;

static _XMP_postreq_t _postreq;

/**
 * Initialize environment for post/wait directives
 */
void _xmp_mpi_post_wait_initialize()
{
  _postreq.num      = 0;
  _postreq.max_size = _XMP_POSTREQ_TABLE_INITIAL_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);

  _postreq.table[0].node = 0;
  _postreq.table[_postreq.max_size-1].node=0;
}

static void add_request(const int node, const int tag)
{
  _postreq.table[_postreq.num].node = node;
  _postreq.table[_postreq.num].tag  = tag;
  _postreq.num++;
}

static void do_post(const int node, const int tag)
{
  if(_postreq.num == _postreq.max_size){
    XACC_DEBUG("reallocation\n");
    _postreq.max_size *= _XMP_POSTREQ_TABLE_INCREMENT_RATIO;
    size_t next_size = sizeof(_XMP_postreq_info_t) * _postreq.max_size;
    _XMP_postreq_info_t *tmp;
    if((tmp = realloc(_postreq.table, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _postreq.table = tmp;
  }

  add_request(node, tag);
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

inline static bool remove_request_noargs()
{
  if(_postreq.num > 0){
    shift_postreq(_postreq.num-1);
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

inline static bool remove_request_node(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

inline static bool remove_request(const int node, const int tag)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node && tag == _postreq.table[i].tag){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void receive_request(int *node, int *tag)
{
  MPI_Status status;
  XACC_DEBUG("receive_request");
  MPI_Recv(tag, 1, MPI_INT, MPI_ANY_SOURCE, _XMP_N_MPI_TAG_POSTREQ, MPI_COMM_WORLD, &status);
  *node = status.MPI_SOURCE;
  XACC_DEBUG("receivee_request, node=%d, tag=%d", *node, *tag);
}

/**
 * Post operation
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_mpi_post(const int node, int tag)
{
  if(node == _XMP_world_rank){
    do_post(_XMP_world_rank, tag);
  } else{
    //XACC_DEBUG("post (node=%d, tag=%d)", node, tag);
    MPI_Send(&tag, 1, MPI_INT, node, _XMP_N_MPI_TAG_POSTREQ, MPI_COMM_WORLD);
  }
}

/**
 * Wait operation without node-ref and tag
 */
void _xmp_mpi_wait_noargs()
{
  if(remove_request_noargs()){
    return;
  }

  int tag;
  int node;
  receive_request(&tag, &node);
}

/**
 * Wait operation with node-ref
 *
 * @param[in] node node number
 */
void _xmp_mpi_wait_node(const int node)
{
  if(remove_request_node(node)){
    return;
  }

  while(1){
    int recv_tag;
    int recv_node;
    receive_request(&recv_node, &recv_tag);
    if(recv_node == node){
      return;
    }
    do_post(recv_node, recv_tag);
  }
}

/**
 * Wait operation with node-ref and tag
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_mpi_wait(const int node, const int tag)
{
  XACC_DEBUG("wait (%d,%d)", node,tag);
  if(remove_request(node, tag)){
    XACC_DEBUG("wait end (already recved)");
    return;
  }

  while(1){
    int recv_tag;
    int recv_node;
    receive_request(&recv_node, &recv_tag);
    if(recv_node == node && recv_tag == tag){
      XACC_DEBUG("wait end (recved)");
      return;
    }
    do_post(recv_node, recv_tag);
  }
}
