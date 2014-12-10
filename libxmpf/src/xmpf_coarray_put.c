/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

#define MAX_RANK 15

static void _putLoops(int serno, void *baseAddr, int bytes,
                      int coindex, size_t *src,
                      int loops, int skip[], int count[],
                      int isSpread);

static void _putCoarray(int serno, void *baseAddr, int coindex, void *rhs,
                        int bytes, int rank, int skip[], int count[],
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

  void *nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, void*);
    skip[i] = (size_t)nextAddr - (size_t)baseAddr;
    count[i] = *(va_arg(argList, int*));
  }

  int bytes = _XMPF_get_coarrayElement(*serno);

  _putCoarray(*serno, baseAddr, *coindex, rhs, 
              bytes, *rank, skip, count, 0 /*isSpread*/);
}


void _putCoarray(int serno, void *baseAddr, int coindex, void *rhs,
                 int bytes, int rank, int skip[], int count[],
                 int isSpread)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg)
      fprintf(stderr, "**** %d bytes fully contiguous (%s)\n",
              bytes, __FILE__);

    if (isSpread) {
      _XMP_fatal("Not supported: \"<array-coindexed-var> = <scalar-expr>\""
                 __FILE__);
    } else {
      _putVector_bytes(serno, baseAddr, bytes, coindex, rhs);
    }
    return;
  }

  if (bytes == skip[0]) {  // contiguous
    _putCoarray(serno, baseAddr, coindex, rhs,
                bytes * count[0], rank - 1, skip + 1, count + 1,
                isSpread);
    return;
  }

  // not contiguous any more
  size_t src = (size_t)rhs;

  if (_XMPF_coarrayMsg) {
    char work[200];
    char* p;
    sprintf(work, "**** put, %d-byte contiguous", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, ", %d %d-byte skips", count[i], skip[i]);
      p += strlen(p);
    }
    fprintf(stderr, "%s (%s)\n", work, __FILE__);
  }

  _putLoops(serno, baseAddr, bytes, coindex, &src,
            rank, skip, count, isSpread);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end put\n");
  }
}

  
void _putLoops(int serno, void *baseAddr, int bytes,
               int coindex, size_t *src,
               int loops, int skip[], int count[], int isSpread)
{
  size_t dst = (size_t)baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _putVector_bytes(serno, (void*)dst, bytes, coindex, (void*)(*src));
      *src += bytes;
      dst += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      _putLoops(serno, baseAddr + i * gap, bytes, coindex, src,
                loops - 1, skip, count, isSpread);
    }
  }
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
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, rhs, NULL);
}


