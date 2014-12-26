/*
 *   COARRAY GET
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

static void _getCoarray(int serno, void *baseAddr, int coindex, void *res,
                        int bytes, int rank, int skip[], int count[]);

static void _getVectorIter(int serno, void *baseAddr, int bytes,
                           int coindex, size_t *dst,
                           int loops, int skip[], int count[]);

static void _getVectorByByte(int serno, void *baseAddr, int bytes,
                             int coindex, void* dst);
static void _getVectorByElement(void *desc, int start, int vlength,
                                int coindex, void* dst);


/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_get_array_(int *serno, void *baseAddr, int *element,
                                    int *coindex, void *res, int *rank, ...)
{
  // element is not used.

  if (*rank == 0) {   // scalar 
    void* desc = _XMPF_get_coarrayDesc(*serno);
    int start = _XMPF_get_coarrayStart(*serno, baseAddr);
    _getVectorByElement(desc, start, 1, *coindex, res);
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

  _getCoarray(*serno, baseAddr, *coindex, res, 
              bytes, *rank, skip, count);
}


void _getCoarray(int serno, void *baseAddr, int coindex, void *res,
                 int bytes, int rank, int skip[], int count[])
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg)
      fprintf(stderr, "**** %d bytes fully contiguous (%s)\n",
              bytes, __FILE__);

    _getVectorByByte(serno, baseAddr, bytes, coindex, res);
    return;
  }

  if (bytes == skip[0]) {  // contiguous
    _getCoarray(serno, baseAddr, coindex, res,
                bytes * count[0], rank - 1, skip + 1, count + 1);
    return;
  }

  // not contiguous any more
  size_t dst = (size_t)res;

  if (_XMPF_coarrayMsg) {
    char work[200];
    char* p;
    sprintf(work, "**** get, %d-byte contiguous", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, ", %d %d-byte skips", count[i], skip[i]);
      p += strlen(p);
    }
    fprintf(stderr, "%s (%s)\n", work, __FILE__);
  }

  _getVectorIter(serno, baseAddr, bytes, coindex, &dst,
                 rank, skip, count);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end get\n");
  }
}

  
void _getVectorIter(int serno, void *baseAddr, int bytes,
                    int coindex, size_t *dst,
                    int loops, int skip[], int count[])
{
  size_t src = (size_t)baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _getVectorByByte(serno, (void*)src, bytes, coindex, (void*)(*dst));
      *dst += bytes;
      src += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      _getVectorIter(serno, baseAddr + i * gap, bytes, coindex, dst,
                     loops - 1, skip, count);
    }
  }
}


void _getVectorByByte(int serno, void *baseAddr, int bytes,
                      int coindex, void* dst)
{
  void* desc = _XMPF_get_coarrayDesc(serno);
  int start = _XMPF_get_coarrayStart(serno, baseAddr);
  // The element that was recorded when the data was allocated is used.
  int element = _XMPF_get_coarrayElement(serno);
  int vlength = bytes / element;

  _getVectorByElement(desc, start, vlength, coindex, dst);
}


void _getVectorByElement(void *desc, int start, int vlength,
                         int coindex, void* dst)
{
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // coindexed-object
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // result
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_GET_CODE, desc, dst, NULL);
}


