#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#include "xmp.h"
#define MEMID_MAX 511
#define _XMP_POST_WAIT_QUEUESIZE 32
#define _XMP_POST_WAIT_QUEUECHUNK 512

typedef struct PWMessage{
  int tag;
  int flag;
} PWMessage_t;

typedef struct request_list{
  int node;
  int tag;
} request_list_t;
typedef struct post_wait_obj{
  int wait_num;
  request_list_t *list;
  int list_size;
} post_wait_obj_t;

static bool *_memid;
static uint64_t **raddr_start;
static int tag = 0;
static int ranksize, myrank;
static void** gatable;
struct FJMPI_Rdma_cq cq;
uint64_t laddr, raddr;
uint64_t laddrack, raddrack;
static post_wait_obj_t pw;
volatile PWMessage_t *sbuf, *rbuf;
volatile int *sack, *rack;

int _XMP_fjrdma_initialize()
{
  int ret = FJMPI_Rdma_init();
  _memid = (bool*)malloc(sizeof(bool) * MEMID_MAX);
  if(_memid == NULL){
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: _memid)\n");
      _XMP_fatal("RDMA MEMID error!");
  }
  gatable = (void**)malloc(sizeof(void*) * MEMID_MAX);
  if(gatable == NULL){
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: gatable)\n");
      _XMP_fatal("RDMA MEMID error!");
  }

  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  raddr_start = (uint64_t**)malloc(sizeof(uint64_t*)*MEMID_MAX);
  if(raddr_start == NULL){
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: raddr_start)\n");
      _XMP_fatal("RDMA MEMID error!");
  }
  for (int i = 0; i < MEMID_MAX; i++) {
    raddr_start[i] = (uint64_t *)malloc(sizeof(uint64_t)*(ranksize));
    if (raddr_start[i] == NULL) {
	fprintf(stderr, "malloc error. ");
	fprintf(stderr, "(xmp_fjrdma_coarray: init: raddr_start[%d])\n", i);
	_XMP_fatal("RDMA MEMID error!");
    }
    _memid[i] = 0;
  }
  return ret;
}

int _XMP_fjrdma_finalize()
{
  return FJMPI_Rdma_finalize();
}

void _XMP_fjrdma_sync_memory()
{
  ;
//    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
}

void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
}

uint64_t _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray, void **buf, unsigned long long num_of_elmts){
  uint64_t dma_addr;
  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  char **each_addr = (char **)_XMP_alloc(sizeof(char *) * ranksize);
  int tmp_memid;
  int checkmemid;
  for(checkmemid = 0; checkmemid < MEMID_MAX; checkmemid++){
    if(_memid[checkmemid] == 0){
      _memid[checkmemid] = 1;
      tmp_memid = checkmemid;
      break;
    }
  }
  if(checkmemid == MEMID_MAX){
    _XMP_fatal("RDMA MEMID error!");
  }

  if(coarray->image_dims != 1) {
    _XMP_fatal("not yet (image_dim < 2)!");
  }
  *buf = (void *)malloc(coarray->elmt_size * num_of_elmts);
  if(*buf == NULL)
      _XMP_fatal("malloc *buf error!");

  dma_addr = FJMPI_Rdma_reg_mem(tmp_memid, *buf, coarray->elmt_size * num_of_elmts);
  if(dma_addr == FJMPI_RDMA_ERROR)
    _XMP_fatal("malloc dma_addr error!");

  /* kokowo hazusu kotowo kanngaero */
  for (int i = 0; i < ranksize; i++) {
    while ((raddr_start[tmp_memid][i] = FJMPI_Rdma_get_remote_addr(i, tmp_memid)) == FJMPI_RDMA_ERROR) {
      ;
    }
    each_addr[i] = (char *)raddr_start[tmp_memid][i];
  }

  /* kokowo hazusu kotowo kanngaero*/
  coarray->addr = each_addr;
  gatable[tmp_memid] = coarray->addr[0];

  return dma_addr;
}

void _XMP_fjrdma_put(int target_image, int dest_continuous, int src_continuous, int dest_dims, int src_dims,
		     _XMP_array_section_t *dest_info, _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dest, void *src, long long length, long long array_length, int *image_size)
{
  int destnode = image_size[0]-1;
  int typesize = dest->elmt_size;
  int dest_val_id;
  int tmp_memid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  for(int i = 0; i < MEMID_MAX; i++){
    if(_memid[i] == 0){
      _memid[i] = 1;
      tmp_memid = i;
      break;
    }
  }

  uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (void*)(src), typesize*length);
  for(dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++) { /* aitewo sagasu */
    if(dest->addr[0] == gatable[dest_val_id])
      break;
  }
  if (dest_val_id == MEMID_MAX) {
    _XMP_fatal("no match dest address!");
  }
  uint64_t r_dmaaddr = get_addr(raddr_start[dest_val_id][destnode], dest_dims, dest_info, typesize);
  uint64_t l_dmaaddr = get_addr(laddr_start, src_dims, src_info, typesize);
  int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;

  if (FJMPI_Rdma_put(destnode, tag%14, r_dmaaddr, l_dmaaddr, typesize*length, options))
    _XMP_fatal("fjrdma_coarray.c:_put error");

  FJMPI_Rdma_dereg_mem(tmp_memid);
  _memid[tmp_memid] = 0;
  if (tag == 14) {
      tag = 0;
  } else {
      tag++;
  }
}

void _XMP_fjrdma_get(int target_image, int dest_continuous, int src_continuous, int dest_dims, int src_dims,
		     _XMP_array_section_t *dest_info, _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dest, void *src, long long length, long long array_length, int *image_size)
{
  int destnode = image_size[0]-1;
  int typesize = dest->elmt_size;
  int dest_val_id;
  int tmp_memid;
    
  for(int i = 0; i < MEMID_MAX; i++){
    if(_memid[i] == 0){
      _memid[i] = 1;
      tmp_memid = i;
      break;
    }
  }
  if(tmp_memid == MEMID_MAX)
    _XMP_fatal("RDMA ERROR! (MEMID over)");

  //    uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (volatile void*)(src), typesize*length);
  uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (void*)(src), typesize*length);

  for(dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++){
    if(dest->addr[0] == gatable[dest_val_id])
      break;
  }
  if(dest_val_id == MEMID_MAX)
    _XMP_fatal("no match dest address (GET)");

  uint64_t r_dmaaddr = get_addr(raddr_start[dest_val_id][destnode], dest_dims, dest_info, typesize);
  uint64_t l_dmaaddr = get_addr(laddr_start, src_dims, src_info, typesize);
  int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;
  if(FJMPI_Rdma_get(destnode, tag%14, r_dmaaddr, l_dmaaddr, typesize*length, options))
    _XMP_fatal("(fjrdma_coarray.c:_get) error!");

  int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  while(flag == 0){
    flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  }
  if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
    fprintf(stderr, "fjrdma_coarray.c (%d, %d || %d, %d) %d\n",
	    cq.pid, destnode, cq.tag, tag % 14, flag);
    _XMP_fatal("(fjrdma_coarray.c:_get) CQ error!");
  }
  FJMPI_Rdma_dereg_mem(tmp_memid);
  _memid[tmp_memid] = 0;
  if (tag == 14) {
      tag = 0;
  } else {
      tag++;
  }
}

static uint64_t get_addr(uint64_t start_addr, int dims, _XMP_array_section_t *info, int typesize)
{
  uint64_t target_addr = start_addr;
  for(int i = 0; i < dims; i++){
    int dist_length = 1;
    for(int j = i + 1; j < dims; j++){
      dist_length *= info[j].size;
    }
    target_addr += typesize*info[i].start*dist_length;
  }
  return target_addr;
}

void _XMP_post(xmp_desc_t dummy1, int dummy2,int target_node, int tag)
{
  int dummy_sbuf;
  MPI_Request *req;

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  target_node -= 1;   // for 1-origin in XMP
  fjrdma_post(target_node, tag);
}

void _XMP_wait(int dummy, int target_node, int tag)
{
  int dummy_rbuf;
  MPI_Status status;
  target_node -= 1;   // 0, 1, 2, ...
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  fjrdma_wait(target_node, tag);
}

void fjrdma_init(){
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

  pw.wait_num  = 0;
  pw.list      = malloc(sizeof(request_list_t) * _XMP_POST_WAIT_QUEUESIZE);
  pw.list_size = _XMP_POST_WAIT_QUEUESIZE;

  sbuf = (PWMessage_t*)malloc(sizeof(PWMessage_t));
  rbuf = (PWMessage_t*)malloc(ranksize * sizeof(PWMessage_t));
  sbuf[0].flag == 1;
  laddr = FJMPI_Rdma_reg_mem(0, (void*)sbuf, sizeof(PWMessage_t));
  FJMPI_Rdma_reg_mem(1, (void*)rbuf, ranksize * sizeof(PWMessage_t));
  while ((raddr = FJMPI_Rdma_get_remote_addr(myrank, 1)) == FJMPI_RDMA_ERROR);

  sack = (int*)malloc(sizeof(int));
  rack = (int*)malloc(ranksize * sizeof(int));
  sack[0] == 1;
  laddrack = FJMPI_Rdma_reg_mem(2, (void*)sack, sizeof(int));
  FJMPI_Rdma_reg_mem(3, (void*)rack, ranksize * sizeof(int));
  while ((raddrack = FJMPI_Rdma_get_remote_addr(myrank, 3)) == FJMPI_RDMA_ERROR);
}

void fjrdma_finalize(){
}

void fjrdma_pw_push(int node, int tag)
{
  pw.list[pw.wait_num].node = node;
  pw.list[pw.wait_num].tag  = tag;
  pw.wait_num++;
}

void fjrdma_do_post(int node, int tag)
{
  if(pw.wait_num == 0){
    fjrdma_pw_push(node, tag);
  }
  else if (pw.wait_num < 0) { // This statement does not executed.
    _XMP_fatal("fjrdma_do_post() : Variable pw.wait_num is illegal.");
  }
  else {  // pw.wait_num > 0
    if (pw.list_size == pw.wait_num) {
      request_list_t *old_list = pw.list;
      pw.list_size += _XMP_POST_WAIT_QUEUECHUNK;
      pw.list = malloc(sizeof(request_list_t) * pw.list_size);
      memcpy(pw.list, old_list, sizeof(request_list_t) * pw.wait_num);
      free(old_list);
    }
    else if(pw.list_size < pw.wait_num) {  // This statement does not executed.
      _XMP_fatal("fjrdma_do_post() : Variable pw.wait_num is illegal.");
    }
    fjrdma_pw_push(node, tag);
  }
}

void fjrdma_post(int target_node, int tag)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  if(target_node == myrank){
    fjrdma_do_post(myrank, tag);
  }
  else{
    while (rack[target_node] == 0) {
      rack[target_node] = 0;
      sbuf[0].tag = tag;
      int nic_flag = FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC1 | FJMPI_RDMA_STRONG_ORDER;
      FJMPI_Rdma_put(target_node, 1, raddr + sizeof(PWMessage_t) * target_node, laddr, sizeof(PWMessage_t), nic_flag);
      rack[target_node] = 1;
    }
  }
}

void fjrdma_pw_cutdown(int index)
{
  if (index != pw.wait_num - 1) {  // Not tail index
    for (int i = index + 1; i < pw.wait_num; i++) {
      pw.list[i - 1] = pw.list[i];
    }
  }
  pw.wait_num--;
}

static int fjrdma_pw_remove(int node, int tag)
{
  for(int i = pw.wait_num-1; i >= 0; i--){
    if(node == pw.list[i].node && tag == pw.list[i].tag){
      fjrdma_pw_cutdown(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void fjrdma_wait(int sourcerank, int tag)
{
  while(fjrdma_pw_remove(sourcerank, tag)){
    if (rbuf[sourcerank].flag == 1){
      fjrdma_do_post(sourcerank, tag);
      rbuf[sourcerank].flag = 0;
      int nic_flag = FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC1;
      nic_flag |= FJMPI_RDMA_REMOTE_NOTICE;
      FJMPI_Rdma_put(sourcerank, 3, raddrack + sizeof(int) * sourcerank, laddrack, sizeof(int), nic_flag);
      while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_LOCAL_NIC0, &cq) != FJMPI_RDMA_NOTICE);
    }
  }
}
