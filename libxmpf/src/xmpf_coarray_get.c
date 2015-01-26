/*
 *   COARRAY GET
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

// communication schemes
#define SCHEME_Normal      0
#define SCHEME_BufferCopy    1
#define SCHEME_BufferSpread  2  /* not used in the case get */

static void _getCoarray(int serno, char *baseAddr, int coindex, char *res,
                        int bytes, int rank, int skip[], int count[]);

static char *_getVectorIter(int serno, char *baseAddr, int bytes,
                            int coindex, char *dst,
                            int loops, int skip[], int count[]);

static void _getVectorByByte(int serno, char *baseAddr, int bytes,
                             int coindex, char *dst);
static void _getVectorByElement(char *desc, int start, int vlength,
                                int coindex, char *dst);


/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_get_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *res, int *rank, ...)
{
  size_t bufsize;
  char *buf, *p;
  int i, nelems;

  /*** temporary ****/
  int scheme = SCHEME_BufferCopy;

  // shortcut for case scalar 
  if (*rank == 0) {   
    char* desc = _XMPF_get_coarrayDesc(*serno);
    int start = _XMPF_get_coarrayStart(*serno, baseAddr);

    switch (scheme) {
    case SCHEME_Normal:
      _getVectorByElement(desc, start, 1, *coindex, res);
      break;

    case SCHEME_BufferCopy:
      buf = malloc((size_t)(*element));
      _getVectorByElement(desc, start, 1, *coindex, buf);
      (void)memcpy(res, buf, *element);
      break;

    default:
      _XMP_fatal("unexpected scheme number in " __FILE__);
    }

    return;
  }

  char *nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, char*);
    skip[i] = nextAddr - baseAddr;
    count[i] = *(va_arg(argList, int*));
  }

  int bytes = _XMPF_get_coarrayElement(*serno);

  switch (scheme) {
  case SCHEME_Normal:
    _getCoarray(*serno, baseAddr, *coindex, res, bytes, *rank, skip, count);
    break;

  case SCHEME_BufferCopy:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    buf = malloc(bufsize);
    _getCoarray(*serno, baseAddr, *coindex, buf, bytes, *rank, skip, count);
    (void)memcpy(res, buf, bufsize);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


void _getCoarray(int serno, char *baseAddr, int coindex, char *res,
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
  char* dst = res;

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

  dst = _getVectorIter(serno, baseAddr, bytes, coindex, dst,
                       rank, skip, count);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end get\n");
  }
}

  
char *_getVectorIter(int serno, char *baseAddr, int bytes,
                     int coindex, char *dst,
                     int loops, int skip[], int count[])
{
  char* src = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _getVectorByByte(serno, src, bytes, coindex, dst);
      dst += bytes;
      src += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      dst = _getVectorIter(serno, baseAddr + i * gap, bytes,
                           coindex, dst,
                           loops - 1, skip, count);
    }
  }
  return dst;
}


void _getVectorByByte(int serno, char *src, int bytes,
                      int coindex, char *dst)
{
  char* desc = _XMPF_get_coarrayDesc(serno);
  int start = _XMPF_get_coarrayStart(serno, src);
  // The element that was recorded when the data was allocated is used.
  int element = _XMPF_get_coarrayElement(serno);
  int vlength = bytes / element;

  _getVectorByElement(desc, start, vlength, coindex, dst);
}


void _getVectorByElement(char *desc, int start, int vlength,
                         int coindex, char *dst)
{
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // coindexed-object
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // result
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_GET_CODE, desc, dst, NULL);
}

