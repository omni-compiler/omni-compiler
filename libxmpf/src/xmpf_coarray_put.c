/*
 *   COARRAY PUT
 *
 */

#include "xmpf_internal.h"

static void _coarray_putVector(int serno, void *baseAddr, int vlength,
                               int coindex, void* rhs);



extern void xmpf_coarray_put_array_(int *serno, void *baseAddr, int *coindex,
                                    void *rhs, int *rank, ...)
{
  if (*rank == 0) {   // scalar 
    _coarray_putVector(*serno, baseAddr, 1, *coindex, rhs);
  } else {
    _XMP_fatal("internal error: not supported yet in xmpf_coarray_put.c");
  }
}   

void _coarray_putVector(int serno, void *baseAddr, int vlength,
                        int coindex, void* rhs)
{
  void* desc = _XMPF_get_coarrayDesc(serno);
  int start = _XMPF_get_coarrayStart(serno, baseAddr);

  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, rhs, NULL);
}

/***
extern void _XMPF_coarray_put_array(int serno, void *baseAddr, int coindex,
                                   void *rts, int rank, ...);
***/



/************
static void _coarray_put77d0(void *desc, int costart, int count,
                             int unit, int node, void *rhs);


// always contiguous
void xmpf_coarray_put77d0_(int *descrId, int *unitLen, void *baseAddr,
                           int *coindex, void *rhs )
{
  coarray_info_t *info = get_coarray_info(*descrId);
  int costart = (baseAddr - info->co_addr) / info->unit_size;
  int count = *unitLen / info->unit_size;

  _coarray_put77d0(info->desc, costart, count,
                   info->unit_size, *coindex - 1, rhs);
}
***********/
