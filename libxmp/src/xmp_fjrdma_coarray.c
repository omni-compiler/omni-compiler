#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define MEMID_MAX 510
//#define FJRDMA_DEBUG

static int _memid = 0;
static uint64_t **raddr_start;
static int tag = 0;
static int numprocs, rank;
struct FJMPI_Rdma_cq cq;
void* gatable[MEMID_MAX];

int _XMP_fjrdma_initialize() {
    int ret = FJMPI_Rdma_init();

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    raddr_start = (uint64_t**)malloc(sizeof(uint64_t*)*510);
    for (int i = 0; i < MEMID_MAX; i++) {
	raddr_start[i] = (uint64_t *)malloc(sizeof(uint64_t)*(numprocs));
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

    if ((_memid < 0) || (MEMID_MAX < _memid)) {
	_XMP_fatal("RDMA MEMID error!");
    }
    if (coarray->image_dims != 1) {
	_XMP_fatal("not yet (image_dim < 2)!");
    }
    *buf = (void *)malloc(coarray->elmt_size * num_of_elmts);
    retregmem = FJMPI_Rdma_reg_mem(_memid, *buf, coarray->elmt_size * num_of_elmts);
    if (retregmem == FJMPI_RDMA_ERROR) {
      _XMP_fatal("reg_memory error!");
    }

    for (int i = 0; i < numprocs; i++) {
	while ((raddr_start[_memid][i] = FJMPI_Rdma_get_remote_addr(i, _memid)) == FJMPI_RDMA_ERROR) {
	  ;
	}
	each_addr[i] = (char *)raddr_start[_memid][i];
    }
    coarray->addr = each_addr;
    gatable[_memid] = coarray->addr[0];
    //    fprintf(stderr, "(ST)ga_add=%x\n", coarray->addr[0]);
    //    printf("buf=%d raddr_start=%d\n", *buf, raddr_start[_memid][0]);
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
		     long long array_length,
		     int *image_size) {

    int destnode = image_size[0]-1;
    int typesize = dest->elmt_size;
    int dest_val_id;
    uint64_t laddr_start = FJMPI_Rdma_reg_mem(_memid, (void*)(src), typesize);
    if (laddr_start == FJMPI_RDMA_ERROR) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++) {
      //      fprintf(stderr, "search%d(%x, %x)\n", dest_val_id, dest->addr[0], gatable[dest_val_id]);
      if (dest->addr[0] == gatable[dest_val_id])
	break;
    }
    if (dest_val_id == MEMID_MAX) {
      _XMP_fatal("no match dest address!");
    }
#ifdef FJRDMA_DEBUG
    fprintf(stderr, "===PUT arg===\n");
    fprintf(stderr, "array_length=%d  ", array_length);
    fprintf(stderr, "coarray_length=%d  ", length);
    fprintf(stderr, "destnode=%d  ", destnode);
    fprintf(stderr, "tag=%d  ", tag);
    fprintf(stderr, "dest_val_id=%d  ", dest_val_id);
    fprintf(stderr, "typesize=%d\n", typesize);
    fprintf(stderr, "dest_info[i].start: ");
    for (int i = 0; i < 2; i++) {
      fprintf(stderr, "[%d]=%d ", i, dest_info[i].start);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "raddr_start[%d][%d]=%d\n", dest_val_id, destnode, raddr_start[dest_val_id][destnode]);
    fprintf(stderr, "laddr_start=%d\n", laddr_start);
    fprintf(stderr, "Coarray address: ");
    for (int i = 0; i < numprocs; i++) {
      fprintf(stderr, "[%d]=%d ", i, dest->addr[i]);
    }
    fprintf(stderr, "\n");
    //    fprintf(stderr, "source_value=%d or %d or %lf\n", *(int*)(src), *(long*)(src), *(float*)(src));
    fprintf(stderr, "dest_info: ");
    for (int i = 0; i < dest_dims; i++) {
      fprintf(stderr, "[%d]=%d ", i, dest_info[0].start);
    }
    fprintf(stderr, "\n");
#endif
    uint64_t raddr = remote_addr(raddr_start[dest_val_id][destnode], dest_dims, dest_info, typesize);
    uint64_t laddr = local_addr(laddr_start, src_dims, src_info, typesize);
    int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;
    if (FJMPI_Rdma_put(destnode, tag%14, raddr, laddr, typesize, options)) {
      _XMP_fatal("fjrdma_coarray.c:_put error");
    }
    int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    while(flag == 0) {
      flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    }
    if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
      fprintf(stderr, "fjrdma_coarray.c (%d, %d || %d, %d) %d\n", cq.pid, destnode, cq.tag, tag % 14, flag);
      _XMP_fatal("CQ error (PUT)");
    }
#ifdef FJRDMA_DEBUG
    fprintf(stderr, "===END PUT===\n");
#endif
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
		     long long array_length,
		     int *image_size) {
    int destnode = image_size[0]-1;
    int typesize = dest->elmt_size;
    int dest_val_id;
    uint64_t laddr_start = FJMPI_Rdma_reg_mem(_memid, (void*)(src), typesize);
    for (dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++) {
      //      fprintf(stderr, "search%d(%x, %x)\n", dest_val_id, dest->addr[0], gatable[dest_val_id]);
      if (dest->addr[0] == gatable[dest_val_id])
	break;
    }
    if (dest_val_id == MEMID_MAX) {
      _XMP_fatal("no match dest address (GET)");
    }
#ifdef FJRDMA_DEBUG
    fprintf(stderr, "===GET arg===\n");
    fprintf(stderr, "array_length=%d  ", array_length);
    fprintf(stderr, "coarray_length=%d  ", length);
    fprintf(stderr, "destnode=%d  ", destnode);
    fprintf(stderr, "tag=%d  ", tag);
    fprintf(stderr, "dest_val_id=%d  ", dest_val_id);
    fprintf(stderr, "typesize=%d\n", typesize);
    fprintf(stderr, "dest_info[i].start: ");
    for (int i = 0; i < 2; i++) {
      fprintf(stderr, "[%d]=%d ", i, dest_info[i].start);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "raddr_start[%d][%d]=%d\n", dest_val_id, destnode, raddr_start[dest_val_id][destnode]);
    fprintf(stderr, "laddr_start=%d\n", laddr_start + typesize * src_info[0].start);
    fprintf(stderr, "Coarray address: ");
    //    for (int i = 0; i < numprocs; i++) {
    for (int i = 0; i < numprocs; i++) {
      fprintf(stderr, "[%d]=%d ", i, dest->addr[i]);
    }
    fprintf(stderr, "\n");
    //    fprintf(stderr, "source_value=%d or %d or %lf\n", *(int*)(src), *(long*)(src), *(float*)(src));
#endif
    uint64_t raddr = remote_addr(raddr_start[dest_val_id][destnode], dest_dims, dest_info, typesize);
    uint64_t laddr = local_addr(laddr_start, src_dims, src_info, typesize);
    int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;
    if (FJMPI_Rdma_get(destnode, tag%14, raddr, laddr, typesize, options)) {
      _XMP_fatal("(fjrdma_coarray.c:_get) error!");
    }
    int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    while(flag == 0) {
      flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
    }
    if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
      fprintf(stderr, "fjrdma_coarray.c (%d, %d || %d, %d) %d\n", cq.pid, destnode, cq.tag, tag % 14, flag);
      _XMP_fatal("(fjrdma_coarray.c:_get) CQ error!");
    }
#ifdef FJRDMA_DEBUG
    fprintf(stderr, "===END GET===\n");
#endif
    _memid++;
    tag++;
}

uint64_t remote_addr(uint64_t start_addr, int dims, _XMP_array_section_t *info, int typesize) {
    uint64_t target_addr = start_addr;
    for (int i = 0; i < dims; i++) {
#ifdef FJRDMA_DEBUG
      fprintf(stderr, "info[%d].start=%d, info[%d].size=%lld\n", i, info[i].start, i, info[i].size);
#endif
      int dist_length=1;
      for (int j = i + 1; j < dims; j++) {
#ifdef FJRDMA_DEBUG
	fprintf(stderr, "[remote] i=%d j=%d size=%lld\n", i, j, info[j].size);
#endif
	dist_length *= info[j].size;
      }
#ifdef FJRDMA_DEBUG
      fprintf(stderr, "dist_length=%d\n", dist_length);
#endif
      target_addr += typesize*info[i].start*dist_length;
#ifdef FJRDMA_DEBUG
      fprintf(stderr, "info[%d].start=%d, info[%d].distance=%lld target_addr=%d\n", i, info[i].start, i, info[i].size, target_addr);
#endif
    }
    return target_addr;
}

uint64_t local_addr(uint64_t start_addr, int dims, _XMP_array_section_t *info, int typesize) {
    uint64_t target_addr = start_addr;
    for (int i = 0; i < dims; i++) {
#ifdef FJRDMA_DEBUG
      fprintf(stderr, "info[%d].start=%d, info[%d].size=%lld\n", i, info[i].start, i, info[i].size);
#endif
      int dist_length=1;
      for(int j = i + 1; j < dims; j++) {
#ifdef FJRDMA_DEBUG
	fprintf(stderr, "[local] i=%d j=%d size=%lld\n", i, j, info[j].size);
#endif
	dist_length *= info[j].size;
      }
#ifdef FJRDMA_DEBUG
      fprintf(stderr, "dist_length=%d\n", dist_length);
#endif
      target_addr += typesize*info[i].start*dist_length;
    }
    return target_addr;
}
