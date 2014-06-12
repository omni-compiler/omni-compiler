#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#include "xmp.h"
#define TAG 0
static int _num_of_puts = 0;
static struct FJMPI_Rdma_cq cq;

void _XMP_fjrdma_shortcut_put(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, const int transfer_size)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_image] + dst_point;
  uint64_t laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_point;
  FJMPI_Rdma_put(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);
  _num_of_puts++;
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
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_point;

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

void _XMP_fjrdma_shortcut_get(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			      const int transfer_size)
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_image] + src_point;
  uint64_t laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_point;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();
  FJMPI_Rdma_get(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);

  // To complete the above get operation.
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);
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
    laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_point;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();

  FJMPI_Rdma_get(target_image, TAG, raddr, laddr, transfer_size, FLAG_NIC);

  // To complete the above get operation.
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
  while(_num_of_puts != 0){
    if(FJMPI_Rdma_poll_cq(SEND_NIC, &cq) == FJMPI_RDMA_NOTICE)
      _num_of_puts--;
  }
}

void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}
