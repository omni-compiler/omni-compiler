#include "xmp_internal.h"

void _XMP_fjrdma_atomic_define(int target_rank, _XMP_coarray_t *dst_desc, size_t dst_offset, int value,
			       _XMP_coarray_t *src_desc, size_t src_offset, size_t elmt_size)
{
  uint64_t raddr, laddr;
  if(target_rank != _XMP_world_rank)
    raddr = (uint64_t)dst_desc->addr[target_rank] + elmt_size * dst_offset;
  else
    raddr = (uint64_t)dst_desc->laddr_the_same_node + elmt_size * dst_offset;

  if(src_desc == NULL){
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, &value, elmt_size);
  }
  else{
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + elmt_size * src_offset;
  }

  FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, elmt_size, _XMP_COARRAY_FLAG_NIC);
  _XMP_add_num_of_puts();
  _XMP_fjrdma_sync_memory_put(); // ensure to complete the above put operation.

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

void _XMP_fjrdma_atomic_ref(int target_rank ,_XMP_coarray_t *dst_desc, size_t dst_offset, int* value,
			    _XMP_coarray_t *src_desc, size_t src_offset, size_t elmt_size)
{
  uint64_t raddr, laddr;
  if(target_rank != _XMP_world_rank)
    raddr = (uint64_t)dst_desc->addr[target_rank] + elmt_size * dst_offset;
  else
    raddr = (uint64_t)dst_desc->laddr_the_same_node + elmt_size * dst_offset;

  if(src_desc == NULL){
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, value, elmt_size);
  }
  else{
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + elmt_size * src_offset;
  }

  FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, elmt_size, _XMP_COARRAY_FLAG_NIC);
  _XMP_add_num_of_gets();
  _XMP_fjrdma_sync_memory_get(); // ensure to complete the above get operation.

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

