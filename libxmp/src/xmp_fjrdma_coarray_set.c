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
#define TAG 0
static int _memid = 1; // _memid = 0 (macro MEMID in xmp_fjrdma_coarray_do.c) is used to put/get operations.

void _XMP_fjrdma_initialize()
{
  int ret = FJMPI_Rdma_init();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");
}

void _XMP_fjrdma_finalize()
{
  int ret = FJMPI_Rdma_finalize();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");
}

void _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray, void **buf, unsigned long long coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  int memid = _memid++;
  if(_memid == MEMID_MAX)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 511.");

  *buf = _XMP_alloc(coarray_size);
  uint64_t laddr = FJMPI_Rdma_reg_mem(memid, *buf, coarray_size);

  for(int i=0; i<_XMP_world_size; i++)
    if(i != _XMP_world_rank)
      while((each_addr[i] = FJMPI_Rdma_get_remote_addr(i, memid)) == FJMPI_RDMA_ERROR);

  // Memo: Reterun wrong local address by using FJMPI_Rdma_get_remote_addr.
  // So FJMPI_Rdma_reg_mem should be used.
  each_addr[_XMP_world_rank] = laddr;

  coarray->addr = (void *)each_addr;
}


