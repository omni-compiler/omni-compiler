#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define MEMID_MAX 511
//#define MEMID_MAX 100
//#define MEMID_MAX 10
//#define FJRDMA_DEBUG
//#define FJRDMA_DEBUG_INST2
//#define FJRDMA_DEBUG_INST
//#define FJRDMA_DEBUG_LSET
//#define FJRDMA_DEBUG_PUT
//#define FJRDMA_DEBUG_GET
//#define FJRDMA_DEBUG_ADDR
//#define FJRDMA_DEBUG_PW
#define POLLINPUT

static bool *_memid;
static uint64_t **raddr_start;
static int tag = 0;
static int numprocs, rank;
static void** gatable;
struct FJMPI_Rdma_cq cq;

int _XMP_fjrdma_initialize() {
  int ret = FJMPI_Rdma_init();
  _memid = (bool*)malloc(sizeof(bool) * MEMID_MAX);
  if (_memid == NULL) {
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: _memid)\n");
      _XMP_fatal("RDMA MEMID error!");
  }
  gatable = (void**)malloc(sizeof(void*) * MEMID_MAX);
  if (gatable == NULL) {
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: gatable)\n");
      _XMP_fatal("RDMA MEMID error!");
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  raddr_start = (uint64_t**)malloc(sizeof(uint64_t*)*MEMID_MAX);
  if (raddr_start == NULL) {
      fprintf(stderr, "malloc error. (xmp_fjrdma_coarray: init: raddr_start)\n");
      _XMP_fatal("RDMA MEMID error!");
  }
  for (int i = 0; i < MEMID_MAX; i++) {
    raddr_start[i] = (uint64_t *)malloc(sizeof(uint64_t)*(numprocs));
    if (raddr_start[i] == NULL) {
	fprintf(stderr, "malloc error. ");
	fprintf(stderr, "(xmp_fjrdma_coarray: init: raddr_start[%d])\n", i);
	_XMP_fatal("RDMA MEMID error!");
    }
    _memid[i] = 0;
  }
  return ret;
}

int _XMP_fjrdma_finalize() {
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr, "return FJMPI_Rdma_finalize();\n");
#endif
  return FJMPI_Rdma_finalize();
}

void _XMP_fjrdma_sync_memory() {
    ;
//    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
}

void _XMP_fjrdma_sync_all() {
  _XMP_fjrdma_sync_memory();
#ifdef FJRDMA_DEBUG_INST
    fprintf(stderr, "_XMP_fjrdma_sync_memory();\n");
#endif
}

uint64_t _XMP_fjrdma_local_set(_XMP_coarray_t *coarray,
			       void **buf,
			       unsigned long long num_of_elmts) {
  uint64_t dma_addr;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char **each_addr = (char **)_XMP_alloc(sizeof(char *) * numprocs);
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "char **each_addr = (char **)_XMP_alloc(sizeof(char *) * %d);\n",
	  numprocs);
#endif
  int tmp_memid;
  int checkmemid;
  for (checkmemid = 0; checkmemid < MEMID_MAX; checkmemid++) {
    if (_memid[checkmemid] == 0) {
      _memid[checkmemid] = 1;
      tmp_memid = checkmemid;
      break;
    }
  }
  if (checkmemid == MEMID_MAX) {
    _XMP_fatal("RDMA MEMID error!");
  }

  if (coarray->image_dims != 1) {
    _XMP_fatal("not yet (image_dim < 2)!");
  }
  *buf = (void *)malloc(coarray->elmt_size * num_of_elmts);
  if (*buf == NULL) {
      fprintf(stderr, "malloc error. ");
      fprintf(stderr, "(xmp_fjrdma_coarray: reg_mem: buf)\n");
      _XMP_fatal("RDMA MEMID error!");
  }
  dma_addr = FJMPI_Rdma_reg_mem(tmp_memid, *buf, coarray->elmt_size * num_of_elmts);
  if (dma_addr == FJMPI_RDMA_ERROR) {
    _XMP_fatal("reg_memory error!");
  }
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "*buf = (void *)malloc(%d * %d);\n",
	  coarray->elmt_size,  num_of_elmts);
  fprintf(stderr,
	  "dma_addr = FJMPI_Rdma_reg_mem(%d, *buf, %d * %d);\n",
	  tmp_memid, coarray->elmt_size, num_of_elmts);
#endif
#ifdef FJRDMA_DEBUG_LSET
  //  if (rank == 0) {
    fprintf(stderr, "[LoSet: %d](cp1) memid=%d, arg3=%d ",
	    rank, tmp_memid, coarray->elmt_size * num_of_elmts);
    fprintf(stderr, "(elmt_size=%d, num_of_elmts=%d)\n",
	    coarray->elmt_size, num_of_elmts);
    //  }
#endif

  /* kokowo hazusu kotowo kanngaero */
  for (int i = 0; i < numprocs; i++) {
    while ((raddr_start[tmp_memid][i] = FJMPI_Rdma_get_remote_addr(i, tmp_memid)) == FJMPI_RDMA_ERROR) {
      ;
    }
#ifdef FJRDMA_DEBUG_LSET
    fprintf(stderr, "[LoSet: %d](cp2) raddr_start[%d][%d]=%llu(%p,%d)\n",
      rank, tmp_memid, i, raddr_start[tmp_memid][i], raddr_start[tmp_memid][i], raddr_start[tmp_memid][i]);
#endif
    each_addr[i] = (char *)raddr_start[tmp_memid][i];
  }
  /* kokowo hazusu kotowo kanngaero*/
  coarray->addr = each_addr;
  gatable[tmp_memid] = coarray->addr[0];
#ifdef FJRDMA_DEBUG_LSET
  //  if (rank == 0) {
        fprintf(stderr, "[LoSet: %d](cp3) ga_addr=%llu(%p,%d),  ", rank, coarray->addr[0], coarray->addr[0], coarray->addr[0]);
	fprintf(stderr, "buf=%llu(%p,%d),  DMA_ADDR=%p\n", *buf, *buf, *buf, dma_addr);
        fprintf(stderr, "[LoSet: %d](cp4) gatable[%d]=%p\n\n", rank, tmp_memid, coarray->addr[0]);
    //  }
#endif
  return dma_addr;
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
  assert(0 < dest_dims);
  assert(0 < src_dims);
  assert(0 < length);
  assert(0 < array_length);
  int destnode = image_size[0]-1;
  int typesize = dest->elmt_size;
  int dest_val_id;
  int tmp_memid;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[START PUT: %d]\n", rank);
#endif
  for (int i = 0; i < MEMID_MAX; i++) {
    if (_memid[i] == 0) {
      _memid[i] = 1;
      tmp_memid = i;
      break;
    }
  }
  assert(0 <= tmp_memid && tmp_memid < MEMID_MAX);
  uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (void*)(src), typesize*length);
  if (laddr_start == FJMPI_RDMA_ERROR) {
    _XMP_fatal("FJRDMA PUT ERROR!");
  }
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "uint64_t laddr_start = FJMPI_Rdma_reg_mem(%d, (void*)(src), %d*%d);\n",
	  tmp_memid, typesize, length);
#endif
#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[PUT: %d](cp1) ", rank);
  fprintf(stderr, "memid=%d src=%p typesize=%d length=%d (laddr_start=%p)\n",
	  tmp_memid, src, typesize, length, laddr_start);
#endif
  for (dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++) { /* aitewo sagasu */
#ifdef FJRDMA_DEBUG_PUT
    fprintf(stderr, "[PUT: %d](cp2) search%d(%llu(%p,%d), %llu(%p,%d))\n",
    	    rank, dest_val_id,
	    dest->addr[0], dest->addr[0], dest->addr[0],
	    gatable[dest_val_id], gatable[dest_val_id], gatable[dest_val_id]);
#endif
    if (dest->addr[0] == gatable[dest_val_id])
      break;
  }
  if (dest_val_id == MEMID_MAX) {
    _XMP_fatal("no match dest address!");
  }
#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[PUT: %d](cp3) ", rank);
  fprintf(stderr, "array_length=%d, ", array_length);
  fprintf(stderr, "coarray_length=%d, ", length);
  fprintf(stderr, "destnode=%d, ", destnode);
  fprintf(stderr, "tag=%d\n", tag);
  fprintf(stderr, "[PUT: %d](cp4) ", rank);
  fprintf(stderr, "dest_val_id=%d, ", dest_val_id);
  fprintf(stderr, "typesize=%d  ", typesize);
  fprintf(stderr, "dest_info[i].start: ");
  for (int i = 0; i < 2; i++) {
    fprintf(stderr, "[%d]=%d ", i, dest_info[i].start);
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "[PUT: %d](cp5) ", rank);
  fprintf(stderr, "raddr_start[%d][%d]=%llu(%p,%d)  ",
	  dest_val_id, destnode,
	  raddr_start[dest_val_id][destnode], raddr_start[dest_val_id][destnode], raddr_start[dest_val_id][destnode]);
  fprintf(stderr, "laddr_start=%llu(%p,%d)\n", laddr_start, laddr_start, laddr_start);
  fprintf(stderr, "[PUT: %d](cp6) Coarray address(dest->addr[i]): ", rank);
  for (int i = 0; i < numprocs; i++) {
    fprintf(stderr, "[%d]=%d ", i, dest->addr[i]);
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "[PUT: %d](cp7) dest_info: ", rank);
  for (int i = 0; i < dest_dims; i++) {
    fprintf(stderr, "[%d]=%d ", i, dest_info[0].start);
  }
  fprintf(stderr, "\n");
#endif
  uint64_t r_dmaaddr = get_addr(raddr_start[dest_val_id][destnode],
				dest_dims,
				dest_info,
				typesize);
  uint64_t l_dmaaddr = get_addr(laddr_start,
				src_dims,
				src_info,
				typesize);
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "uint64_t r_dmaaddr = get_addr(raddr_start[%d][%d], %d, dest_info, %d);\n",
	  dest_val_id, destnode, dest_dims, typesize);
  fprintf(stderr,
	  "uint64_t l_dmaaddr = get_addr(laddr_start, %d, src_info, %d);\n",
	  src_dims, typesize);
#endif
#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[PUT: %d](cp8) ", rank);
  fprintf(stderr, "r_dmaaddr=%llu(%p,%d)\n",r_dmaaddr, r_dmaaddr, r_dmaaddr);
  fprintf(stderr, "[PUT: %d](cp8) ", rank);
  fprintf(stderr, "l_dmaaddr=%llu(%p,%d)\n",l_dmaaddr, l_dmaaddr, l_dmaaddr);
#endif
  int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;
  if (FJMPI_Rdma_put(destnode, tag%14, r_dmaaddr, l_dmaaddr, typesize*length, options)) {
    _XMP_fatal("fjrdma_coarray.c:_put error");
  }
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;\n");
  fprintf(stderr,
	  "if (FJMPI_Rdma_put(%d, %d, r_dmaaddr, l_dmaaddr, %d*%d, %d)) {\n",
	  destnode, tag%14, typesize, length, options);
  fprintf(stderr, "\t_XMP_fatal(...);\n");
  fprintf(stderr, "};\n");
#endif
#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[PUT: %d](cp9) ", rank);
  fprintf(stderr, "Rdma_put:args(%d, %d, %llu(%p,%d), %llu(%p,%d), %d, %d)\n",
	  destnode, tag%14, r_dmaaddr, l_dmaaddr, typesize*length, options);
#endif
#ifdef POLLINPUT
  int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  while(flag == 0) {
    flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  }
  if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
    fprintf(stderr, "fjrdma_coarray.c (%d, %d || %d, %d) %d\n", cq.pid, destnode, cq.tag, tag % 14, flag);
    _XMP_fatal("CQ error (PUT)");
  }
  //  while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
#endif
#ifdef FJRDMA_DEBUG_PUT
  fprintf(stderr, "[END PUT: %d]\n\n\n", rank);
#endif
  FJMPI_Rdma_dereg_mem(tmp_memid);
  _memid[tmp_memid] = 0;
  if (tag == 14) {
      tag = 0;
  } else {
      tag++;
  }
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
  int tmp_memid;
    
  for (int i = 0; i < MEMID_MAX; i++) {
    if (_memid[i] == 0) {
      _memid[i] = 1;
      tmp_memid = i;
      break;
    }
  }
  if (tmp_memid == MEMID_MAX) {
    _XMP_fatal("RDMA ERROR! (MEMID over)");
  }
    //    uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (volatile void*)(src), typesize*length);
  uint64_t laddr_start = FJMPI_Rdma_reg_mem(tmp_memid, (void*)(src), typesize*length);
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "uint64_t laddr_start = FJMPI_Rdma_reg_mem(%d, (void*)(src), %d*%d);\n",
	  tmp_memid, typesize, length);
#endif
#ifdef FJRDMA_DEBUG_GET
  fprintf(stderr, "get typesize=%d length=%d\n", typesize, length);
#endif
  for (dest_val_id=0; dest_val_id < MEMID_MAX; dest_val_id++) {
#ifdef FJRDMA_DEBUG_GET
    //    fprintf(stderr, "search%d(%llu, %llu)\n",
    fprintf(stderr, "search%d(%p, %p)\n",
	    dest_val_id, dest->addr[0], gatable[dest_val_id]);
#endif
    if (dest->addr[0] == gatable[dest_val_id])
      break;
  }
  if (dest_val_id == MEMID_MAX) {
    _XMP_fatal("no match dest address (GET)");
  }
#ifdef FJRDMA_DEBUG_GET
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
  uint64_t r_dmaaddr = get_addr(raddr_start[dest_val_id][destnode], dest_dims, dest_info, typesize);
  uint64_t l_dmaaddr = get_addr(laddr_start, src_dims, src_info, typesize);
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "uint64_t r_dmaaddr = get_addr(raddr_start[%d][%d], %d, dest_info, %d);\n",
	  dest_val_id, destnode, dest_dims, typesize);
  fprintf(stderr,
	  "uint64_t l_dmaaddr = get_addr(laddr_start, %d, src_info, %d);\n",
	  src_dims, typesize);
#endif
  int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;
  if (FJMPI_Rdma_get(destnode, tag%14, r_dmaaddr, l_dmaaddr, typesize*length, options)) {
    _XMP_fatal("(fjrdma_coarray.c:_get) error!");
  }
#ifdef FJRDMA_DEBUG_INST
  fprintf(stderr,
	  "int options = FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1;\n");
  fprintf(stderr,
	  "if (FJMPI_Rdma_get(%d, %d, r_dmaaddr, l_dmaaddr, %d*%d, %d)) {\n",
	  destnode, tag%14, typesize, length, options);
  fprintf(stderr, "\t_XMP_fatal(...);\n");
  fprintf(stderr, "};\n");
#endif
  int flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  while(flag == 0) {
    flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
  }
  if((cq.pid != destnode) || (cq.tag != tag%14) || flag != FJMPI_RDMA_NOTICE) {
    fprintf(stderr, "fjrdma_coarray.c (%d, %d || %d, %d) %d\n",
	    cq.pid, destnode, cq.tag, tag % 14, flag);
    _XMP_fatal("(fjrdma_coarray.c:_get) CQ error!");
  }
#ifdef FJRDMA_DEBUG_GET
  fprintf(stderr, "===END GET===\n");
#endif
  FJMPI_Rdma_dereg_mem(tmp_memid);
  _memid[tmp_memid] = 0;
  if (tag == 14) {
      tag = 0;
  } else {
      tag++;
  }
}

uint64_t get_addr(uint64_t start_addr,
		  int dims,
		  _XMP_array_section_t *info,
		  int typesize) {
  uint64_t target_addr = start_addr;
  for (int i = 0; i < dims; i++) {
#ifdef FJRDMA_DEBUG_ADDR
    fprintf(stderr, "info[%d].start=%d,  ", i, info[i].start);
    fprintf(stderr, "info[%d].size=%lld\n", i, info[i].size);
#endif
    int dist_length = 1;
    for (int j = i + 1; j < dims; j++) {
#ifdef FJRDMA_DEBUG_ADDR
      fprintf(stderr, "[remote] i=%d j=%d size=%lld\n", i, j, info[j].size);
#endif
      dist_length *= info[j].size;
    }
#ifdef FJRDMA_DEBUG_ADDR
    fprintf(stderr, "dist_length=%d\n", dist_length);
#endif
    target_addr += typesize*info[i].start*dist_length;
#ifdef FJRDMA_DEBUG_ADDR
    fprintf(stderr, "info[%d].start=%d  ", i, info[i].start);
    fprintf(stderr, "info[%d].distance=%lld  ", i, info[i].size);
    fprintf(stderr, "target_addr=%d\n",target_addr);
#endif
  }
  return target_addr;
}

void _XMP_post(int target_node, int tag) {
  int dummy_sbuf;
  MPI_Request *req;
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "POST: start\n");
#endif
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  target_node -= 1;   // for 1-origin in XMP
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "POST [%d(me)]->[%d] tag=%d\n", rank, target_node, tag);
#endif
  MPI_Isend(&dummy_sbuf, 1, MPI_INT, target_node, tag, MPI_COMM_WORLD, req);
//  MPI_Send(&dummy_sbuf, 1, MPI_INT, target_node, tag, MPI_COMM_WORLD);
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "POST: finished\n");
#endif
}

void _XMP_wait(int dummy, int target_node, int tag) {
  int dummy_rbuf;
  MPI_Status status;
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "WAIT: start\n");
#endif
  target_node -= 1;   // 0, 1, 2, ...
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "WAIT [%d]->[%d(me)] tag=%d\n", rank, tag, target_node);
#endif
  MPI_Recv(&dummy_rbuf, 1, MPI_INT, target_node, tag, MPI_COMM_WORLD, &status);
//  MPI_Recv(&dummy_rbuf, 1, MPI_INT, target_node, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
#ifdef FJRDMA_DEBUG_PW
  fprintf(stderr, "WAIT finished\n");
#endif
}
