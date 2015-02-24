/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

// communication schemes
#define PUTSCHEME_Normal        0
#define PUTSCHEME_SendBuffer    1

static int _select_putscheme(int condition);

static void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int count[]);

static char *_putVectorIter(int serno, char *baseAddr, int bytes,
                            int coindex, char *src,
                            int loops, int skip[], int count[]);

static void _putVector(int serno, char *baseAddr, int bytes,
                       int coindex, char* src);
#if 0
/* disused */
static void _putVectorByByte(int serno, char *baseAddr, int bytes,
                             int coindex, char* src);
static void _putVectorByElement(char *desc, int start, int vlength,
                                int coindex, char* src);
#endif

/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_put_scalar_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition)
{
  _XMPF_checkIfInTask("scalar coindexed variable");

  int scheme = _select_putscheme(*condition);

  //  char *desc = _XMPF_get_coarrayDesc(*serno);
  // size_t offset = _XMPF_get_coarrayOffset(*serno, baseAddr);

  switch (scheme) {
  case PUTSCHEME_Normal:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select PUTSCHEME_Normal/scalar\n"
                              "  baseAddr=%p, *element=%d\n",
                              baseAddr, *element);
    }
    _putVector(*serno, baseAddr, *element, *coindex, rhs);
    break;
    
  case PUTSCHEME_SendBuffer:
    {
      size_t elementRU = ROUND_UP_BOUNDARY(*element);
      char buf[elementRU];   // could be in RDMA area

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select PUTSCHEME_SendBuffer/scalar\n"
                                "  baseAddr=%p, elementRU=%zd, buf=%p\n",
                                baseAddr, elementRU, buf);
      }
      (void)memcpy(buf, rhs, *element);
      _putVector(*serno, baseAddr, elementRU, *coindex, buf);
    }
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_put_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...)
{
  _XMPF_checkIfInTask("array coindexed variable");

  size_t bufsize;
  char *buf;
  int i;

  int scheme = _select_putscheme(*condition);

  if (*element % BOUNDARY_BYTE != 0) {
    _XMP_fatal("violation of boundary in put communication"
               "xmpf_coarray_put_array_, " __FILE__);
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

  switch (scheme) {
  case PUTSCHEME_Normal:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select PUTSCHEME_Normal/array\n");
    }
    _putCoarray(*serno, baseAddr, *coindex, rhs, *element, *rank, skip, count);
    break;

  case PUTSCHEME_SendBuffer:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select PUTSCHEME_SendBuffer/array\n");
      fprintf(stderr, "  *bufsize=%zd\n", bufsize);
    }
    buf = malloc(bufsize);
    (void)memcpy(buf, rhs, bufsize);
    _putCoarray(*serno, baseAddr, *coindex, buf, *element, *rank, skip, count);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_put_spread_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition,
                                     int *rank, ...)
{
  _XMPF_checkIfInTask("array coindexed variable");

  size_t bufsize;
  char *buf, *p;
  int i, nelems;

  if (*element % BOUNDARY_BYTE != 0) {
    _XMP_fatal("violation of boundary in spread-put communication");
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

  nelems = 1;
  for (i = 0; i < *rank; i++)
    nelems *= count[i];
  bufsize = nelems * (*element);
  buf = malloc(bufsize);
  for (i = 0, p = buf; i < nelems; i++, p += *element)
    (void)memcpy(p, rhs, *element);
  _putCoarray(*serno, baseAddr, *coindex, buf, *element, *rank, skip, count);
}


/* This can be further optimized.
 */
int _select_putscheme(int condition)
{
  int scheme;

#ifdef _XMP_COARRAY_FJRDMA
  scheme = (condition >= 1) ? PUTSCHEME_SendBuffer : PUTSCHEME_Normal;
#else
  scheme = (condition >= 2) ? PUTSCHEME_SendBuffer : PUTSCHEME_Normal;
#endif

  return scheme;
}


void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
                 int bytes, int rank, int skip[], int count[])
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("PUT %d bytes fully contiguous ===\n", bytes);
    }
    _putVector(serno, baseAddr, bytes, coindex, rhs);
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
    sprintf(work, "PUT %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPF_coarrayDebugPrint("%s bytes ===\n", work);
  }

  src = _putVectorIter(serno, baseAddr, bytes, coindex, src,
                       rank, skip, count);
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
      _putVector(serno, dst, bytes, coindex, src);
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


void _putVector(int serno, char *baseAddr, int bytes, int coindex, char *src)
{
  char* desc = _XMPF_get_coarrayDesc(serno);
  size_t offset = _XMPF_get_coarrayOffset(serno, baseAddr);

  _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_node_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, src, NULL);
}


#if 0
/* disused
 */
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

#endif
