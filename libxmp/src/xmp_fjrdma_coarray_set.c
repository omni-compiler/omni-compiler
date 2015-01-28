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
#define XMP_FJRDMA_MAX_NODES 16384
static int num_of_nodes = 0;

static int compare_char(const void *x, const void *y)
{
  return strcmp((char *)x, (char *)y);
}

static int get_num_of_physical_nodes()
{
  int  namelen, max_namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int tag = 0;
  MPI_Status s;

  MPI_Get_processor_name(processor_name, &namelen);
  MPI_Allreduce(&namelen, &max_namelen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  char all_processor_name[_XMP_world_size][max_namelen];

  if(_XMP_world_rank == 0){
    for(int i=1;i<_XMP_world_size;i++){
      MPI_Recv(all_processor_name[i], sizeof(char)*max_namelen, MPI_CHAR, i, tag, MPI_COMM_WORLD, &s);
    }
    memcpy(all_processor_name[0], processor_name, sizeof(char)*max_namelen);
  }
  else{
    MPI_Send(processor_name, sizeof(char)*max_namelen, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }
  
  int num = 1;
  if(_XMP_world_rank == 0){
    qsort(all_processor_name, _XMP_world_size, sizeof(char)*max_namelen, compare_char);
    for(int i=0;i<_XMP_world_size-1;i++)
      if(strncmp(all_processor_name[i], all_processor_name[i+1], max_namelen))
        num++;
  }

  MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return num;
}

void _XMP_fjrdma_initialize(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &_XMP_world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_XMP_world_size);

  if(_XMP_world_size >= XMP_FJRDMA_MAX_NODES)
    num_of_nodes = get_num_of_physical_nodes();

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
  if(num_of_nodes >= XMP_FJRDMA_MAX_NODES){
    if(_XMP_world_rank == 0)
      fprintf(stderr, "Coarray cannot be used in more than %d physical nodes (Now using %d physical nodes)\n", 
	      XMP_FJRDMA_MAX_NODES, num_of_nodes);
    _XMP_fatal_nomsg();
  }

  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  int memid = _memid++;
  if(_memid == MEMID_MAX)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 510.");

  *buf = _XMP_alloc(coarray_size);
  uint64_t laddr = FJMPI_Rdma_reg_mem(memid, *buf, coarray_size);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int i=1; i<_XMP_world_size+1; i++){
    int partner_rank = (_XMP_world_rank+i)%_XMP_world_size;
    if(partner_rank == _XMP_world_rank)
      each_addr[partner_rank] = laddr;
    else
      while((each_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, memid)) == FJMPI_RDMA_ERROR);

    if(i%3000 == 0)
      MPI_Barrier(MPI_COMM_WORLD);
  }

  coarray->real_addr = *buf;
  coarray->addr = (void *)each_addr;
}
