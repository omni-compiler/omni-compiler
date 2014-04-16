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
#define FLAG_NIC (FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC1 | FJMPI_RDMA_IMMEDIATE_RETURN)
#define SEND_NIC FJMPI_RDMA_LOCAL_NIC0
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
  uint64_t laddr = FJMPI_Rdma_reg_mem(memid, *buf, coarray_size);

  for(int i=0; i<_commsize; i++)
    if(i != _myrank)
      while((each_addr[i] = FJMPI_Rdma_get_remote_addr(i, memid)) == FJMPI_RDMA_ERROR);

  // Memo: Reterun wrong local address by using FJMPI_Rdma_get_remote_addr.
  // So FJMPI_Rdma_reg_mem should be used.
  each_addr[_myrank] = laddr;

  coarray->addr = (void *)each_addr;
}

static void XMP_fjrdma_from_c_to_c_put(int target_image, uint64_t dst_point, uint64_t src_point,
				       _XMP_coarray_t *dst_desc, void *src, _XMP_coarray_t *src_desc,
				       long long transfer_size)
/* If a local array is a coarray, src_desc != NULL. */
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_image] + dst_point;
  uint64_t laddr;
  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)src+src_point), (size_t)transfer_size);
  else
    laddr = (uint64_t)src_desc->addr[_myrank] + src_point;

  FJMPI_Rdma_put(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_put(int dst_continuous, int src_continuous, int target_image, int dst_dims, int src_dims, 
		     _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dst_desc, void *src, _XMP_coarray_t *src_desc, long long length)
{
  long long transfer_size = dst_desc->elmt_size * length;
  _num_of_puts++;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    uint64_t dst_point = (uint64_t)get_offset(dst_info, dst_dims);
    uint64_t src_point = (uint64_t)get_offset(src_info, src_dims);
    XMP_fjrdma_from_c_to_c_put(target_image, dst_point, src_point, dst_desc, src, src_desc, transfer_size);
  }
  else{
    _XMP_fatal("Not implemented");
  }
}

static void XMP_fjrdma_from_c_to_c_get(int target_image, uint64_t dst_point, uint64_t src_point,
				       void *dst, _XMP_coarray_t *dst_desc, _XMP_coarray_t *src_desc, 
				       long long transfer_size)
/* If a local array is a coarray, dst_desc != NULL. */
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_image] + src_point;
  uint64_t laddr;

  if(dst_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)dst+dst_point), (size_t)transfer_size);
  else
    laddr = (uint64_t)dst_desc->addr[_myrank] + dst_point;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();

  FJMPI_Rdma_get(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);

  // To complete the above get operation.
  struct FJMPI_Rdma_cq cq;
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_get(int src_continuous, int dst_continuous, int target_image, int src_dims, int dst_dims, 
		     _XMP_array_section_t *src_info, _XMP_array_section_t *dst_info,
		     _XMP_coarray_t *src_desc, void *dst, _XMP_coarray_t *dst_desc, long long length)
{
  long long transfer_size = src_desc->elmt_size * length;
  
  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    uint64_t dst_point = (uint64_t)get_offset(dst_info, dst_dims);
    uint64_t src_point = (uint64_t)get_offset(src_info, src_dims);
    XMP_fjrdma_from_c_to_c_get(target_image, dst_point, src_point, dst, dst_desc, src_desc, transfer_size);
  }
  else{
    _XMP_fatal("Not implemented");
  }
}

void _XMP_fjrdma_sync_memory()
{
  int num_of_notice = 0;
  struct FJMPI_Rdma_cq cq;

  while(num_of_notice != _num_of_puts){
    if(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) == FJMPI_RDMA_NOTICE)
      num_of_notice++;
  }

  _num_of_puts = 0;
}

void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}
