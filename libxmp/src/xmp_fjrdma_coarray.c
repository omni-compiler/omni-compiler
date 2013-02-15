#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define _XMP_FJRDMA_ALIGNMENT 8

static int _memid = 0;
static unsigned long long _xmp_coarray_shift = 0;
static uint64_t **raddr;
static int tag = 0;
static int numprocs, rank;
struct FJMPI_Rdma_cq cq;
//uint64_t gatable[510];
void* gatable[510];

int _XMP_fjrdma_initialize() {
    int ret = FJMPI_Rdma_init();

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    raddr = (uint64_t**)malloc(sizeof(uint64_t*)*510);
    for (int i = 0; i < 510; i++) {
	raddr[i] = (uint64_t *)malloc(sizeof(uint64_t)*(numprocs));
    }
    return ret;
}

int _XMP_fjrdma_finalize() {
    return FJMPI_Rdma_finalize();
}

void _XMP_fjrdma_sync_memory() {
  ;
}

void _XMP_fjrdma_sync_all(){
  _XMP_fjrdma_sync_memory();
}

uint64_t _XMP_fjrdma_reg_mem(_XMP_coarray_t *coarray, void **buf, unsigned long long num_of_elmts) {
    uint64_t retregmem;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char **each_addr = (char **)_XMP_alloc(sizeof(char *) * numprocs);

    if ((-1 < _memid) && (_memid > 510)) {
	_XMP_fatal("RDMA MEMID error!");
    }
    if (coarray->image_dims != 1) {
	_XMP_fatal("not yet (image_dim < 2)!");
    }
    if (coarray->elmt_size % _XMP_FJRDMA_ALIGNMENT == 0) {
	_xmp_coarray_shift += coarray->elmt_size * num_of_elmts;
    } else {
	int tmp = ((coarray->elmt_size / _XMP_FJRDMA_ALIGNMENT) + 1) * _XMP_FJRDMA_ALIGNMENT;
	_xmp_coarray_shift += tmp * num_of_elmts;
    }
    
    *buf = (void *)malloc(coarray->elmt_size * num_of_elmts);
    retregmem = FJMPI_Rdma_reg_mem(_memid, *buf, coarray->elmt_size * num_of_elmts);
    if (retregmem == FJMPI_RDMA_ERROR) {
      fprintf(stderr, "reg_mem Error!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < numprocs; i++) {
	while ((raddr[_memid][i] = FJMPI_Rdma_get_remote_addr(i, _memid)) == FJMPI_RDMA_ERROR) {
	  ;
	}
	each_addr[i] = (char *)raddr[_memid][i];
    }
    coarray->addr = each_addr;
    gatable[_memid] = coarray->addr[0];
    printf("buf=%d raddr=%d\n", *buf, raddr[_memid][0]);
    _memid++;
    return retregmem;
}

void _XMP_fjrdma_put(int target_image,
		     int dest_continuous,
		     int src_continuous,
		     int dest_dims,
		     int src_dims,
		     _XMP_array_section_t *dest_info,
		     _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dest,
		     void *src,
		     long long length,
		     int *image_size) {

    int destnode = image_size[0]-1;
    int typesize = dest->elmt_size;
    int dest_val_id;
    uint64_t laddr = FJMPI_Rdma_reg_mem(_memid, (void*)(src), typesize);
    if (laddr == FJMPI_RDMA_ERROR) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (dest_val_id=0; dest_val_id < 510; dest_val_id++) {
      if (dest->addr[1] == gatable[dest_val_id])
	break;
    }
    //    printf("dest_val_id=%d\n", dest_val_id);
    //    printf("[PUT] raddr=%d %d %d\n", dest->addr[0], dest->addr[1], (void*)dest);
    //    printf("[PUT] laddr=%d raddr=%d\n", laddr, raddr[0][destnode]);
    if (FJMPI_Rdma_put(destnode,
		       tag%14,
		       raddr[dest_val_id][destnode] + typesize*dest_info[0].start,
		       laddr,
		       typesize,
		       FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1)) {
        fprintf(stderr, "(fjrdma_coarray.c:_put)put error!\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    while(flag == 0) {
      flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    }
    if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
      fprintf(stderr, "(fjrdma_coarray.c)CQ ERROR01 (PUT) (%d, %d || %d, %d) %d\n", cq.pid, destnode, cq.tag, tag % 14, flag);
    }
    /*
    if(FJMPI_Rdma_dereg_mem(_memid) != 0) {
      fprintf(stderr, "dereg_mem error\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */
    _memid++;
    tag++;
}

void _XMP_fjrdma_get(int target_image,
		     int dest_continuous,
		     int src_continuous,
		     int dest_dims,
		     int src_dims,
		     _XMP_array_section_t *dest_info,
		     _XMP_array_section_t *src_info,
		     _XMP_coarray_t *dest,
		     void *src,
		     long long length,
		     int *image_size) {
    int destnode = image_size[0]-1;
    int typesize = dest->elmt_size;
    int dest_val_id;
    uint64_t laddr = FJMPI_Rdma_reg_mem(_memid, (void*)(src), typesize);
    for (dest_val_id=0; dest_val_id < 510; dest_val_id++) {
      if (dest->addr[1] == gatable[dest_val_id])
	break;
    }
    printf("destnode=%d, tag=%d, raddr=%d, dest_val_id=%d, typesize=%d\n", destnode, tag, raddr[dest_val_id][destnode], dest_val_id,typesize);
    //    printf("dest_info[0].start=%d, laddr");
    if (FJMPI_Rdma_get(destnode,
		       tag%14,
		       raddr[dest_val_id][destnode] + typesize*dest_info[0].start,
		       laddr + typesize * src_info[0].start,
		       typesize,
		       FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1)) {
        fprintf(stderr, "(fjrdma_coarray.c:_get)get error!\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    while(flag == 0) {
      flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    }
    if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
      fprintf(stderr, "(fjrdma_coarray.c:_get)CQ ERROR01 (GET) (%d, %d || %d, %d) %d\n", cq.pid, destnode, cq.tag, tag % 14, flag);
    }
    /*
    if(FJMPI_Rdma_dereg_mem(_memid) != 0) {
      fprintf(stderr, "dereg_mem error\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */
    _memid++;
    tag++;
}
