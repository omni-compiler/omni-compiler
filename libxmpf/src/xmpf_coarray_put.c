/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

#define MAX_RANK 15

static void* _putLoops(int serno, void *baseAddr, int bytes,
                       int coindex, void* rhs,
                       int loops, void *nextAddr[], int count[], int isSpread);

static void _putCoarray(int serno, void *baseAddr, int coindex, void *rhs,
                        int bytes, int rank, void *nextAddr[], int count[],
                        int isSpread);

static void _putVector_bytes(int serno, void *baseAddr, int bytes,
                             int coindex, void* rhs);
static void _putVector_elements(void *desc, int start, int vlength,
                                int coindex, void* rhs);


/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_put_array_(int *serno, void *baseAddr, int *coindex,
                                    void *rhs, int *rank, ...)
{
  if (*rank == 0) {   // scalar 
    void* desc = _XMPF_get_coarrayDesc(*serno);
    int start = _XMPF_get_coarrayStart(*serno, baseAddr);
    _putVector_elements(desc, start, 1, *coindex, rhs);
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

  int bytes = _XMPF_get_coarrayElement(*serno);

  _putCoarray(*serno, baseAddr, *coindex, rhs, 
              bytes, *rank, nextAddr, count, 0 /*isSpread*/);
}


void _putCoarray(int serno, void *baseAddr, int coindex, void *rhs,
                 int bytes, int rank, void *nextAddr[], int count[], int isSpread)
{
  if (rank == 0) {  // fully contiguous
    if (isSpread) {
      _XMP_fatal("xmpf_coarray_put.c: not supported \"<array-coindexed-var> = <scalar-expr>\"");
      //void* spread = _setRhs(serno, rhs, bytes);
      //_putVector_bytes(serno, baseAddr, bytes, coindex, spread);
      //_resetRhs();
    } else {
      if (_XMPF_coarrayMsg)
        fprintf(stderr, "**** %d bytes fully contiguous (%s)\n",
                bytes, __FUNCTION__);

      _putVector_bytes(serno, baseAddr, bytes, coindex, rhs);
    }
    return;
  }

  if ((size_t)baseAddr + bytes == (size_t)nextAddr[0]) {  // contiguous
    _putCoarray(serno, baseAddr, coindex, rhs,
                bytes * count[0], rank - 1,
                nextAddr + 1, count + 1, isSpread);
    return;
  }

  // not contiguous any more
  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** put %d contiguous bytes", bytes);
    for (int i = 0; i < rank; i++)
      fprintf(stderr, ", %d times", count[i]);
    fprintf(stderr, " (%s)\n", __FUNCTION__);
  }

  _putLoops(serno, baseAddr, bytes, coindex, rhs,
            rank, nextAddr, count, isSpread);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end put\n");
  }
}

  
void* _putLoops(int serno, void *baseAddr, int bytes,
                int coindex, void* rhs,
                int loops, void *nextAddr[], int count[], int isSpread)
{
  int i;
  int n = count[loops - 1];
  size_t src = (size_t)rhs;
  size_t dst = (size_t)baseAddr;
  size_t step = (size_t)nextAddr[loops - 1] - (size_t)baseAddr;

  if (loops == 1) {
    for (i = 0; i < n; i++) {
      _putVector_bytes(serno, (void*)dst, bytes, coindex, (void*)src);
      src += bytes;
      dst += step;
    }
    rhs = (void*)src;
  } else {
    for (i = 0; i < n; i++) {
      rhs = _putLoops(serno, baseAddr, bytes, coindex, rhs,
                      loops - 1, nextAddr, count, isSpread);
    }
  }

  return rhs;
}   


void _putVector_bytes(int serno, void *baseAddr, int bytes,
                      int coindex, void* rhs)
{
  void* desc = _XMPF_get_coarrayDesc(serno);
  int start = _XMPF_get_coarrayStart(serno, baseAddr);
  int element = _XMPF_get_coarrayElement(serno);
  int vlength = bytes / element;

  _putVector_elements(desc, start, vlength, coindex, rhs);
}


void _putVector_elements(void *desc, int start, int vlength,
                         int coindex, void* rhs)
{
  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "****   start=%d, vlength=%d, coindex=%d (%s)\n",
            start, vlength, coindex, __FUNCTION__);
  }

  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, rhs, NULL);
}


