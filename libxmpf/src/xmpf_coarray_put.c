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

extern void xmpf_coarray_put_scalar_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition)
{
  char *buf;

  int scheme = _select_putscheme(*condition);

  char *desc = _XMPF_get_coarrayDesc(*serno);
  int start = _XMPF_get_coarrayStart(*serno, baseAddr);

  switch (scheme) {
  case PUTSCHEME_Normal:
    _putVectorByElement(desc, start, 1, *coindex, rhs);
    break;
    
  case PUTSCHEME_SendBuffer:
    buf = malloc((size_t)(*element));
    (void)memcpy(buf, rhs, *element);
    _putVectorByElement(desc, start, 1, *coindex, buf);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_put_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...)
{
  size_t bufsize;
  char *buf, *p;
  int i, nelems;

  int scheme = _select_putscheme(*condition);

  if (*element % BOUNDARY_BYTE != 0) {
    _XMP_fatal("violation of boundary in put communication");
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

  switch (scheme) {
  case PUTSCHEME_Normal:
    _putCoarray(*serno, baseAddr, *coindex, rhs, bytes, *rank, skip, count);
    break;

  case PUTSCHEME_SendBuffer:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    buf = malloc(bufsize);
    (void)memcpy(buf, rhs, bufsize);
    _putCoarray(*serno, baseAddr, *coindex, buf, bytes, *rank, skip, count);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_put_spread_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition,
                                     int *rank, ...)
{
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

  int bytes = _XMPF_get_coarrayElement(*serno);

  nelems = 1;
  for (i = 0; i < *rank; i++)
    nelems *= count[i];
  bufsize = nelems * (*element);
  buf = malloc(bufsize);
  for (i = 0, p = buf; i < nelems; i++, p += *element)
    (void)memcpy(p, rhs, *element);
  _putCoarray(*serno, baseAddr, *coindex, buf, bytes, *rank, skip, count);
}


int _select_putscheme(int condition)
{
  int scheme = PUTSCHEME_Normal;

#ifdef _XMP_COARRAY_FJRDMA
  if (condition == 1)
    scheme = PUTSCHEME_SendBuffer;
#endif

  return scheme;
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

