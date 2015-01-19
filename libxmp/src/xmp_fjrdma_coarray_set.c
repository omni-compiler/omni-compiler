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
static int _memid = 2; // _memid = 0 (macro MEMID in xmp_internal.h) is used to put/get operations.
                       // _memid = 1 (macro POST_WAID_ID in xmp_internal.h) is used to post/wait operations.

void _XMP_fjrdma_initialize(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &_XMP_world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_XMP_world_size);

  int ret = FJMPI_Rdma_init();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");
}

void _XMP_fjrdma_finalize()
{
  int ret = FJMPI_Rdma_finalize();
  if(ret) _XMP_fatal("FJMPI_Rdma_init error!");
}

void _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray, void **buf, const size_t coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  int memid = _memid++;
  if(_memid == MEMID_MAX)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 510.");

  *buf = _XMP_alloc(coarray_size);
  uint64_t laddr = FJMPI_Rdma_reg_mem(memid, *buf, coarray_size);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int i=1; i<_XMP_world_size+1; i++){
    int partner_rank = (_XMP_world_rank+i)%_XMP_world_size;
    while((each_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, memid)) == FJMPI_RDMA_ERROR);

    if(i%3000 == 0)
      MPI_Barrier(MPI_COMM_WORLD);
  }

  coarray->addr = (void *)each_addr;
}


