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
#define MEMID 0
#define TAG 0
static int FLAG_NIC = FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC1 | FJMPI_RDMA_IMMEDIATE_RETURN;
static int SEND_NIC = FJMPI_RDMA_LOCAL_NIC0;
static int _memid = 1; // _memid = 0 (macro MEMID) is used to put/get operations.
static int _num_of_puts = 0;
static int _commsize, _myrank;

void _XMP_fjrdma_initialize()
{
  int ret = FJMPI_Rdma_init();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");

  MPI_Comm_size(MPI_COMM_WORLD, &_commsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &_myrank);
}

void _XMP_fjrdma_finalize()
{
  int ret = FJMPI_Rdma_finalize();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");
}

void _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray, void **buf, unsigned long long coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _commsize);
  int memid = _memid++;
  if(_memid == MEMID_MAX)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 511.");

  *buf = _XMP_alloc(coarray_size);
  FJMPI_Rdma_reg_mem(memid, *buf, coarray_size);

  for(int i=0; i<_commsize; i++)
    while((each_addr[i] = FJMPI_Rdma_get_remote_addr(i, memid)) == FJMPI_RDMA_ERROR);

  coarray->addr = (void *)each_addr;
}

static void XMP_fjrdma_from_c_to_c_put(int target_image, uint64_t dst_point, uint64_t src_point,
				       _XMP_coarray_t *dst, void *src, long long transfer_size){
  uint64_t raddr = (uint64_t)dst->addr[target_image] + dst_point;
  uint64_t laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)src+src_point), (size_t)transfer_size);

  fprintf(stderr, "%PRIu64\t%p\t%PRIu64\t%p\t%p\n", laddr, src, src_point, (void *)((char *)src+src_point), (char *)src+src_point);

  FJMPI_Rdma_put(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);
  FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_put(int dst_continuous, int src_continuous, int target_image, int dst_dims, int src_dims, 
		     _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dst, void *src, long long length)
{
  long long transfer_size = dst->elmt_size * length;
  _num_of_puts++;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    uint64_t dst_point = (uint64_t)get_offset(dst_info, dst_dims);
    uint64_t src_point = (uint64_t)get_offset(src_info, src_dims);
    XMP_fjrdma_from_c_to_c_put(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Not implemented");
  }
}

static void XMP_fjrdma_from_c_to_c_get(int target_image, uint64_t dst_point, uint64_t src_point,
				       void *dst, _XMP_coarray_t *src, long long transfer_size){
  uint64_t raddr = (uint64_t)src->addr[target_image] + src_point;
  uint64_t laddr = FJMPI_Rdma_reg_mem(MEMID, (char *)dst+dst_point, transfer_size);

  // To complete put operations before the fellowing get operation.
  _XMP_fjrdma_sync_memory();

  FJMPI_Rdma_get(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);
  
  // To complete the above get operation.
  struct FJMPI_Rdma_cq cq;
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE)

  FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_get(int src_continuous, int dst_continuous, int target_image, int src_dims, int dst_dims, 
		     _XMP_array_section_t *src_info, _XMP_array_section_t *dst_info,
		     _XMP_coarray_t *src, void *dst, long long length)
{
  long long transfer_size = src->elmt_size * length;
  
  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    uint64_t dst_point = (uint64_t)get_offset(dst_info, dst_dims);
    uint64_t src_point = (uint64_t)get_offset(src_info, src_dims);
    XMP_fjrdma_from_c_to_c_get(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Not implemented");
  }
}

void _XMP_fjrdma_sync_memory()
{
  int num_of_notice = 0, ret;
  struct FJMPI_Rdma_cq cq;

  while(1){
    ret = FJMPI_Rdma_poll_cq(SEND_NIC, &cq);
    if(ret == FJMPI_RDMA_NOTICE)
      num_of_notice++;
    if(num_of_notice == _num_of_puts)
      break;
  }

  _num_of_puts = 0;
}

void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}
