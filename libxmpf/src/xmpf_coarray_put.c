/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

#define MAX_RANK 15

static void _coarray_putLoops(int serno, void *baseAddr, int vlength,
                              int coindex, void* rhs,
                              int loops, void *nextAddr[], int count[]);
static void _coarray_putArray(int serno, void *baseAddr, int coindex, void *rhs,
                              int vlength, int rank, void *nextAddr[], int count[]);
static void _coarray_putVector(int serno, void *baseAddr, int vlength,
                               int coindex, void* rhs);



extern void xmpf_coarray_put_array_(int *serno, void *baseAddr, int *coindex,
                                    void *rhs, int *rank, ...)
{
  if (*rank == 0) {   // scalar 
    _coarray_putVector(*serno, baseAddr, 1, *coindex, rhs);
    return;
  }

  void *nextAddr[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  for (int i = 0; i < *rank; i++) {
    nextAddr[i] = va_arg(argList, void*);
    count[i] = *(va_arg(argList, int*));
  }

  int vlength = _XMPF_get_coarrayElement(*serno);

  _coarray_putArray(*serno, baseAddr, *coindex, rhs, 
                    vlength, *rank, nextAddr, count);
}


void _coarray_putArray(int serno, void *baseAddr, int coindex, void *rhs,
                       int vlength, int rank, void *nextAddr[], int count[])
{
  if (rank == 0) {  // fully contiguous
    _coarray_putVector(serno, baseAddr, vlength, coindex, rhs);
    return;
  }

  if ((size_t)baseAddr + vlength == (size_t)nextAddr[0]) {  // yet contiguous
    _coarray_putArray(serno, baseAddr, coindex, rhs,
                      vlength * count[0], rank - 1,
                      nextAddr + 1, count + 1);
    return;
  }

  // not contiguous any more
  _coarray_putLoops(serno, baseAddr, vlength, coindex, rhs,
                    rank, nextAddr, count);
}
  
void _coarray_putLoops(int serno, void *baseAddr, int vlength,
                       int coindex, void* rhs,
                       int loops, void *nextAddr[], int count[])
{
  int i;
  void *p;

  int n = count[loops - 1];
  size_t step = (size_t)nextAddr[loops - 1] - (size_t)baseAddr;

  if (loops == 1) {
    for (i = 0, p = baseAddr; i < n; i++, p += step)
      _coarray_putVector(serno, p, vlength, coindex, rhs);
  } else {
    for (i = 0, p = baseAddr; i < n; i++, p += step)
      _coarray_putLoops(serno, p, vlength, coindex, rhs,
                        loops - 1, nextAddr, count);
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
