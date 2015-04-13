#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define FJRDMA_MAX_SIZE 16777212
#define FJRDMA_MAX_MEMID 511
#define FJRDMA_MAX_MPUT 1993
#define FJRDMA_TAG 0
#define FJRDMA_START_MEMID 2
extern uint64_t* _XMP_calc_raddrs_mput(const uint64_t, const _XMP_array_section_t*, const int, const size_t);
extern uint64_t* _XMP_calc_laddrs_mput(const uint64_t, const size_t);
extern size_t*   _XMP_calc_lengths_mput(const size_t, const size_t);

static int _num_of_puts = 0;
static struct FJMPI_Rdma_cq _cq;
static int _memid = FJRDMA_START_MEMID; // _memid = 0 (macro MEMID in xmp_internal.h) is used to put/get operations.
                                        // _memid = 1 (macro POST_WAID_ID in xmp_internal.h) is used to post/wait operations.

void _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray, void **buf, const size_t coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  if(_memid == FJRDMA_MAX_MEMID)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 510.");

  *buf = _XMP_alloc(coarray_size);
  uint64_t laddr = FJMPI_Rdma_reg_mem(_memid, *buf, coarray_size);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int i=1; i<_XMP_world_size+1; i++){
    int partner_rank = (_XMP_world_rank+i)%_XMP_world_size;
    if(partner_rank == _XMP_world_rank)
      each_addr[partner_rank] = laddr;
    else
      while((each_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _memid)) == FJMPI_RDMA_ERROR);

    if(i%3000 == 0)
      MPI_Barrier(MPI_COMM_WORLD);
  }

  coarray->real_addr = *buf;
  coarray->addr = (void *)each_addr;
  _memid++;
}

void _XMP_fjrdma_coarray_lastly_deallocate()
{
  if(_memid == FJRDMA_START_MEMID) return;

  _memid--;
  FJMPI_Rdma_dereg_mem(_memid);
}

void _XMP_fjrdma_shortcut_put(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, const size_t transfer_size)
{
  if(transfer_size > FJRDMA_MAX_SIZE){
    fprintf(stderr, "transfer_size is too large %zu\n", transfer_size);
    exit(1);
  }

  uint64_t raddr = (uint64_t)dst_desc->addr[target_image] + dst_point;
  uint64_t laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_point;
  FJMPI_Rdma_put(target_image, FJRDMA_TAG, raddr, laddr, transfer_size, FLAG_NIC);
  _num_of_puts++;
}

static void _fjrdma_c_to_c_put(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			       const _XMP_coarray_t *dst_desc, const void *src, const _XMP_coarray_t *src_desc,
			       const size_t transfer_size)
/* If a local array is a coarray, src_desc != NULL. */
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_image] + dst_point;
  uint64_t laddr;

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)src+src_point), transfer_size);
  else
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_point;

  _num_of_puts++;
  FJMPI_Rdma_put(target_image, FJRDMA_TAG, raddr, laddr, transfer_size, FLAG_NIC);

  if(src_desc == NULL)    FJMPI_Rdma_dereg_mem(MEMID);
}

static void _fjrdma_scalar_mput(const int target_image, const _XMP_array_section_t *dst_info, const int dst_dims, 
				const uint64_t src_point, const _XMP_coarray_t *dst_desc, 
				const void *src, const _XMP_coarray_t *src_desc, const size_t transfer_coarray_elmts)
/* If a local array is a coarray, src_desc != NULL. */
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_image];
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)src+src_point), elmt_size);
  else
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_point;

  uint64_t *raddrs = _XMP_calc_raddrs_mput(raddr, dst_info, dst_dims, transfer_coarray_elmts);
  uint64_t *laddrs = _XMP_calc_laddrs_mput(laddr, transfer_coarray_elmts);
  size_t  *lengths = _XMP_calc_lengths_mput(elmt_size, transfer_coarray_elmts);

  if(transfer_coarray_elmts <= FJRDMA_MAX_MPUT){
    _num_of_puts++;
    FJMPI_Rdma_mput(target_image, FJRDMA_TAG, raddrs, laddrs, lengths, 0, transfer_coarray_elmts, FLAG_NIC);
  }
  else{
    int times      = transfer_coarray_elmts / FJRDMA_MAX_MPUT + 1;
    int rest_elmts = transfer_coarray_elmts - FJRDMA_MAX_MPUT * (times - 1);
    _num_of_puts += times;
    for(int i=0;i<times;i++){
      size_t trans_elmts = (i != times-1)? FJRDMA_MAX_MPUT : rest_elmts;
      FJMPI_Rdma_mput(target_image, FJRDMA_TAG, &raddrs[i*FJRDMA_MAX_MPUT], &laddrs[i*FJRDMA_MAX_MPUT], 
		      &lengths[i*FJRDMA_MAX_MPUT], 0, trans_elmts, FLAG_NIC);
    }
  }

  free(raddrs); free(laddrs); free(lengths);
  if(src_desc == NULL)   FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_put(const int dst_continuous, const int src_continuous, const int target_image, const int dst_dims, const int src_dims, 
		     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		     const void *src, const _XMP_coarray_t *src_desc, const int transfer_coarray_elmts, const int transfer_array_elmts)
{
  if(transfer_coarray_elmts == transfer_array_elmts){
    size_t transfer_size = dst_desc->elmt_size * transfer_coarray_elmts;
    if(transfer_size > FJRDMA_MAX_SIZE){
      fprintf(stderr, "transfer_size is too large %zu\n", transfer_size);
      exit(1);
    }

    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      uint64_t dst_point = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
      uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
      _fjrdma_c_to_c_put(target_image, dst_point, src_point, dst_desc, src, src_desc, transfer_size);
    }
    else{
      _XMP_fatal("Not implemented");
    }
  }
  else{
    uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
    _fjrdma_scalar_mput(target_image, dst_info, dst_dims, src_point, 
			dst_desc, src, src_desc, transfer_coarray_elmts);
  }
}

void _XMP_fjrdma_shortcut_get(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			      const size_t transfer_size)
{
  if(transfer_size > FJRDMA_MAX_SIZE){
    fprintf(stderr, "transfer_size is too large %d\n", transfer_size);
    exit(1);
  }

  uint64_t raddr = (uint64_t)src_desc->addr[target_image] + src_point;
  uint64_t laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_point;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();
  FJMPI_Rdma_get(target_image, FJRDMA_TAG, raddr, laddr, transfer_size, FLAG_NIC);

  // To complete the above get operation.
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &_cq) != FJMPI_RDMA_NOTICE);
}

static void _fjrdma_c_to_c_get(const int target_image, const uint64_t dst_point, const uint64_t src_point,
			       const void *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
			       const size_t transfer_size)
/* If a local array is a coarray, dst_desc != NULL. */
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_image] + src_point;
  uint64_t laddr;

  if(dst_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(MEMID, (void *)((char *)dst+dst_point), transfer_size);
  else
    laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_point;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();

  FJMPI_Rdma_get(target_image, FJRDMA_TAG, raddr, laddr, transfer_size, FLAG_NIC);

  // To complete the above get operation.
  while(FJMPI_Rdma_poll_cq(SEND_NIC, &_cq) != FJMPI_RDMA_NOTICE);

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(MEMID);
}

void _XMP_fjrdma_get(const int src_continuous, const int dst_continuous, const int target_image, const int src_dims, const int dst_dims, 
		     const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info,
		     const _XMP_coarray_t *src_desc, const void *dst, const _XMP_coarray_t *dst_desc, const int length)
{
  size_t transfer_size = src_desc->elmt_size * length;
  if(transfer_size > FJRDMA_MAX_SIZE){
    fprintf(stderr, "transfer_size is too large %zu\n", transfer_size);
    exit(1);
  }
  
  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    uint64_t dst_point = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
    uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
    _fjrdma_c_to_c_get(target_image, dst_point, src_point, dst, dst_desc, src_desc, transfer_size);
  }
  else{
    _XMP_fatal("Not implemented");
  }
}

void _XMP_fjrdma_sync_memory()
{
  while(_num_of_puts != 0){
    if(FJMPI_Rdma_poll_cq(SEND_NIC, &_cq) == FJMPI_RDMA_NOTICE)
      _num_of_puts--;
  }
}

void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}
