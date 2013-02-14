#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
//#include "xmp_atomic.h"
#define _XMP_FJRDMA_ALIGNMENT 8

static int _memid = 0;
static int rdma_num = 0;
static unsigned long long _xmp_coarray_shift = 0;
//static char **_xmp_fjrdma_buf;
static uint64_t **raddr;
static int tag = 0;
static int numprocs, rank;
struct FJMPI_Rdma_cq cq;

int _XMP_fjrdma_initialize() {
    int ret;
//    fprintf(stderr, "CP(fjrdma_initialize01)\n");
    ret = FJMPI_Rdma_init();

//    fprintf(stderr, "CP(fjrdma_initialize02)\n");
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    raddr = (uint64_t**)malloc(sizeof(uint64_t*)*512);
    for (int i = 0; i < 512; i++) {
	raddr[i] = (uint64_t *)malloc(sizeof(uint64_t)*(numprocs));
    }

//    _xmp_fjrdma_buf = (char **)_XMP_alloc(sizeof(char*) * numprocs);

    return ret;
}

int _XMP_fjrdma_finalize() {
    return FJMPI_Rdma_finalize();
}

void _XMP_fjrdma_sync_memory() {
    fprintf(stderr, "fjrdma_sync_memory [%d] %d\n", rank, rdma_num);
    /*
    fprintf(stderr, "fjrdma_sync_memory01 [%d] %d\n", rank, rdma_num);
    while (rdma_num > 0) {
	fprintf(stderr, "fjrdma_sync_memory02 [%d] %d\n", rank, rdma_num);
	if (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, NULL) == FJMPI_RDMA_NOTICE) {
	    rdma_num--;
	}
	fprintf(stderr, "fjrdma_sync_memory finish[%d] %d\n", rank, rdma_num);
    }
    */
}

void _XMP_fjrdma_sync_all(){
  _XMP_fjrdma_sync_memory();
  _XMP_barrier_EXEC();
}

uint64_t _XMP_fjrdma_reg_mem(_XMP_coarray_t *coarray, void **buf, unsigned long long num_of_elmts) {
    uint64_t ret;
//    int numprocs, rank;
//    char **each_addr;
    fprintf(stderr, "CP(reg_mem01)\n");
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    _xmp_fjrdma_buf = (char **)_XMP_alloc(sizeof(char*) * numprocs);
    if ((-1 < _memid) && (_memid > 510)) {
	_XMP_fatal("RDMA MEMID error!");
    }
    if (coarray->image_dims != 1) {
	_XMP_fatal("not yet (image_dim < 2)!");
    }
//    each_addr = (uint64_t *)_XMP_alloc(sizeof(uint64_t) * numprocs);
//    each_addr = (char **)_XMP_alloc(sizeof(char *) * numprocs);
//    fprintf(stderr, "CP(reg_mem02) numprocs=%d\n", numprocs);
    for (int i = 0; i < numprocs; i++) {
	fprintf(stderr, "CP(reg_mem02-1) i=%d shift=%d\n", i, _xmp_coarray_shift);
//	fprintf(stderr, "CP(reg_mem02-2) i=%d buf=%d\n", i, _xmp_fjrdma_buf[i]);
//	each_addr[i] = (char *)(_xmp_fjrdma_buf[i]) + _xmp_coarray_shift;
    }

//    fprintf(stderr, "CP(reg_mem03)\n");
    if (coarray->elmt_size % _XMP_FJRDMA_ALIGNMENT == 0) {
//	fprintf(stderr, "CP(reg_mem03-1: if)\n");
	_xmp_coarray_shift += coarray->elmt_size * num_of_elmts;
    } else {
//	fprintf(stderr, "CP(reg_mem03-2: else)\n");
	int tmp = ((coarray->elmt_size / _XMP_FJRDMA_ALIGNMENT) + 1) * _XMP_FJRDMA_ALIGNMENT;
	_xmp_coarray_shift += tmp * num_of_elmts;
    }
    
    fprintf(stderr, "CP(reg_mem04)\n");
    /*
    if(_xmp_coarray_shift > _xmp_heap_size) {
	if(rank == 0) {
	    fprintf(stderr, "Cannot allocate coarray. Now HEAP SIZE of coarray is %d MB\n", (int)(_xmp_heap_size/1024/1024));
	    fprintf(stderr, "But %d MB is needed\n", (int)(_xmp_coarray_shift/1024/1024));
	}
	_XMP_fatal("Please set XMP_COARRAY_HEAP_SIZE=<number>\n");
    }
    */

#if 1
    fprintf(stderr, "reg_mem04-1(rank, elmt_size, num_of_elmts)=(%d, %d, %d)\n", rank, coarray->elmt_size, num_of_elmts);
    *buf = (void *)malloc(coarray->elmt_size * num_of_elmts);
    ret = FJMPI_Rdma_reg_mem(_memid, *buf, coarray->elmt_size * num_of_elmts);
    for (int i = 0; i < numprocs; i++) {
	while ((raddr[_memid][i] = FJMPI_Rdma_get_remote_addr(i, _memid)) == FJMPI_RDMA_ERROR) {
	    ;
	}
    }
    fprintf(stderr, "CP(reg_mem05) num_of_elmts=%d elmt_size=%d\n", num_of_elmts, coarray->elmt_size);
//    size_t length = (size_t)(num_of_elmts * coarray->elmt_size);
//    fprintf(stderr, "CP(reg_num05-1) _memid=%d\n", _memid);
//    while((ret = FJMPI_Rdma_reg_mem(_memid, *buf, length)) == FJMPI_RDMA_ERROR) {
//	fprintf(stderr, "while loop\n");
//	;
//    }
    fprintf(stderr, "CP(reg_mem06)\n");
#else
    ret = FJMPI_Rdma_reg_mem(_memid
			     (void *)_XMP_COARRAY_ADDR_result,
			     sizeof(struct anon_type_26_Pair)*DATA_NUM_X*DATA_NUM_Y);
#endif
//    for (int i = 0; i < numprocs; i++) {
//	while ((each_addr[i] = (void *)FJMPI_Rdma_get_remote_addr(i, _memid)) == (void *)FJMPI_RDMA_ERROR);
//    }
    _memid++;
//    *buf = each_addr[rank];
    return ret;
}

#if 1
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

    uint64_t laddr;
    laddr = FJMPI_Rdma_reg_mem(_memid, (void*)(&src), dest->elmt_size);
    fprintf(stderr, "laddr reg memory(%d, %d)\n", FJMPI_RDMA_ERROR, laddr);
    if (laddr == FJMPI_RDMA_ERROR) {
	fprintf(stderr, "laddr reg memory error (%d)\n", laddr);
	MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
	fprintf(stderr, "laddr reg memory noerror\n");
    }
    if (FJMPI_Rdma_put(image_size[0],
		       (tag++)%14,
		       (uint64_t)(raddr[image_size[0]] + sizeof(long)*dest_info[0].start),
		       laddr,
		       sizeof(long),
		       FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1)) {
	fprintf(stderr, "put error!\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(stderr, "put arg image_size[0]=%d tag=%d raddr[image_size[0]]=%d dest_info[0].start=%d laddr=%d sizeof(long)=%d FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1=%d\n",image_size[0], tag, raddr[image_size[0]], dest_info[0].start, laddr, sizeof(long), FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1);
/*    
    if(dest_continuous == _XMP_N_INT_TRUE) {
	long long dest_point = get_offset(dest_info, dest_dims);
	long long src_point  = get_offset(src_info, src_dims);
	
	FJMPI_Rdma_put();
    }
*/
    fprintf(stderr, "PUT\n");
    rdma_num++;
    fprintf(stderr, "fjrdma_sync_memory01 [%d] %d\n", rank, rdma_num);
//    fprintf(stderr, "fjrdma_sync_memory01d [%d] %d %d\n", rank, FJMPI_RDMA_NOTICE, FJMPI_RDMA_HALFWAY_NOTICE);
    while (rdma_num > 0) {
	int flag;
	fprintf(stderr, "fjrdma_sync_memory02 [%d] %d\n", rank, rdma_num);
	flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
	fprintf(stderr, "fjrdma_sync_memory02d [%d] %d %d %d\n", rank, flag, FJMPI_RDMA_NOTICE, FJMPI_RDMA_HALFWAY_NOTICE);
	FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
	fprintf(stderr, "fjrdma_sync_memory03 [%d] %d\n", rank, rdma_num);
	while(flag == 0) {
	    fprintf(stderr, "fjrdma_sync_memory03-ws [%d] %d\n", rank, rdma_num);
	    flag = FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq);
	    fprintf(stderr, "fjrdma_sync_memory03-we [%d] %d\n", rank, rdma_num);
	}
	if((cq.pid != image_size[0]) || (cq.tag != (tag-1)%14) || flag != FJMPI_RDMA_NOTICE) {
	     fprintf(stderr, "CQ ERROR01 (%d, %d || %d, %d) %d\n", cq.pid, image_size[0], cq.tag, tag-1, flag);
	}
	/*
	if (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, &cq) == FJMPI_RDMA_NOTICE) {
	    rdma_num--;
	    fprintf(stderr, "fjrdma_sync_memory03 [%d] %d\n", rank, rdma_num);
	}
	*/
	fprintf(stderr, "fjrdma_sync_memory04 [%d] %d\n", rank, rdma_num);
    }
    fprintf(stderr, "fjrdma_sync_memory finish[%d] %d\n", rank, rdma_num);
}
#else
void _XMP_gasnet_put(int target_image, int dest_continuous, int src_continuous,
		     int dest_dims, int src_dims, _XMP_array_section_t *dest_info, 
		     _XMP_array_section_t *src_info, _XMP_coarray_t *dest, void *src, long long length){
    static uint64_t laddr;
    laddr = FJRdma_reg_mem(_memid, (void*)(&buf), coarray->elmt_size);
    if(dest_continuous == _XMP_N_INT_TRUE){
	long long dest_point = get_offset(dest_info, dest_dims);
	long long src_point  = get_offset(src_info, src_dims);

	gasnet_put_nbi_bulk(target_image, dest->addr[target_image]+dest_point, ((char *)src)+src_point, 
			    dest->elmt_size*length);
    }
}
#endif

#if 1
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
    static uint64_t laddr;
    laddr = FJMPI_Rdma_reg_mem(_memid, (void*)(&src), dest->elmt_size);
    if (FJMPI_Rdma_put(image_size[0],
		       (tag++)%14,
		       (uint64_t)(raddr[image_size[0]] + sizeof(long)*dest_info[0].start),
		       laddr,
		       sizeof(long),
		       FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC1)) {
	fprintf(stderr, "put error!\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(stderr, "GET\n");

    rdma_num++;
    fprintf(stderr, "fjrdma_sync_memory01 [%d] %d\n", rank, rdma_num);
    while (rdma_num > 0) {
	fprintf(stderr, "fjrdma_sync_memory02 [%d] %d\n", rank, rdma_num);
	if (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, NULL) == FJMPI_RDMA_NOTICE) {
	    rdma_num--;
	}
	fprintf(stderr, "fjrdma_sync_memory finish[%d] %d\n", rank, rdma_num);
    }
}

#else
void _XMP_gasnet_get(int target_image, int src_continuous, int dest_continuous,
		     int src_dims, int dest_dims, _XMP_array_section_t *src_info,
                     _XMP_array_section_t *dest_info, _XMP_coarray_t *src, void *dest, long long length){

    if(dest_continuous == _XMP_N_INT_TRUE){
      long long dest_point = get_offset(dest_info, dest_dims);
      long long src_point  = get_offset(src_info, src_dims);

      gasnet_get_bulk(((char *)dest)+dest_point, target_image, ((char *)src->addr[target_image])+src_point, 
    		    src->elmt_size*length);
    }
    rdma_num++;
}
#endif
