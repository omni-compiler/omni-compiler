/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

// communication schemes
#define SCHEME_Normal      0
#define SCHEME_BufferCopy    1
#define SCHEME_BufferSpread  2
#define SCHEME_RDMABufferCopy    3    /* not implemented yet */
#define SCHEME_RDMABufferSpread  4    /* not implemented yet */

static void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int count[]);

static char *_putVectorIter(int serno, char *baseAddr, int bytes,
                            int coindex, char *src,
                            int loops, int skip[], int count[]);

static void _putVectorByByte(int serno, char *baseAddr, int bytes,
                             int coindex, char* src);
static void _putVectorByElement(char *desc, int start, int vlength,
                                int coindex, char* src);


/***************************************************\
    entry
\***************************************************/

/*
 *  assumed that tha value of emelent is the same as the one recorded previously.
 */
extern void xmpf_coarray_put_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *rhs, int *scheme, int *rank, ...)
{
  size_t bufsize;
  char *buf, *p;
  int i, nelems;

  // shortcut for case scalar 
  if (*rank == 0) {   
    char *desc = _XMPF_get_coarrayDesc(*serno);
    int start = _XMPF_get_coarrayStart(*serno, baseAddr);

    switch (*scheme) {
    case SCHEME_Normal:
      _putVectorByElement(desc, start, 1, *coindex, rhs);
      break;

    case SCHEME_BufferCopy:
    case SCHEME_BufferSpread:
      buf = malloc((size_t)(*element));
      (void)memcpy(buf, rhs, *element);
      _putVectorByElement(desc, start, 1, *coindex, buf);
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
    nextAddr = va_arg(argList, char*);         // nextAddr1, nextAddr2, ...
    skip[i] = nextAddr - baseAddr;
    count[i] = *(va_arg(argList, int*));       // count1, count2, ...
  }

  int bytes = _XMPF_get_coarrayElement(*serno);

  switch (*scheme) {
  case SCHEME_Normal:
    _putCoarray(*serno, baseAddr, *coindex, rhs, bytes, *rank, skip, count);
    break;

  case SCHEME_BufferCopy:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    buf = malloc(bufsize);
    (void)memcpy(buf, rhs, bufsize);
    _putCoarray(*serno, baseAddr, *coindex, buf, bytes, *rank, skip, count);
    break;

  case SCHEME_BufferSpread:
    nelems = 1;
    for (i = 0; i < *rank; i++)
      nelems *= count[i];
    bufsize = nelems * (*element);
    buf = malloc(bufsize);
    for (i = 0, p = buf; i < nelems; i++, p += *element)
      (void)memcpy(p, rhs, *element);
    _putCoarray(*serno, baseAddr, *coindex, buf, bytes, *rank, skip, count);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
                 int bytes, int rank, int skip[], int count[])
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg)
      fprintf(stderr, "**** %d bytes fully contiguous (%s)\n",
              bytes, __FILE__);

    _putVectorByByte(serno, baseAddr, bytes, coindex, rhs);
    return;
  }

  if (bytes == skip[0]) {  // contiguous
    _putCoarray(serno, baseAddr, coindex, rhs,
                bytes * count[0], rank - 1, skip + 1, count + 1);
    return;
  }

  // not contiguous any more
  char* src = rhs;

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

  src = _putVectorIter(serno, baseAddr, bytes, coindex, src,
                       rank, skip, count);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end put\n");
  }
}

  
char *_putVectorIter(int serno, char *baseAddr, int bytes,
                     int coindex, char *src,
                     int loops, int skip[], int count[])
{
  char* dst = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _putVectorByByte(serno, dst, bytes, coindex, src);
      src += bytes;
      dst += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      src = _putVectorIter(serno, baseAddr + i * gap, bytes,
                           coindex, src,
                           loops - 1, skip, count);
    }
  }
  return src;
}


void _putVectorByByte(int serno, char *baseAddr, int bytes,
                      int coindex, char *src)
{
  char* desc = _XMPF_get_coarrayDesc(serno);
  int start = _XMPF_get_coarrayStart(serno, baseAddr);
  // The element that was recorded when the data was allocated is used.
  int element = _XMPF_get_coarrayElement(serno);
  int vlength = bytes / element;

  _putVectorByElement(desc, start, vlength, coindex, src);
}


void _putVectorByElement(char *desc, int start, int vlength,
                         int coindex, char* src)
{
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, src, NULL);
}
