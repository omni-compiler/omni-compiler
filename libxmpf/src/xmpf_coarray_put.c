/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

static void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int count[],
                        int isSpread);

static char *_putVectorIter(int serno, char *baseAddr, int bytes,
                            int coindex, char *src,
                            int loops, int skip[], int count[],
                            int isSpread);

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
  ///////
  printf("enter xmpf_coarray_put_array_\n");
  printf("  rhs=%p\n", rhs);
  float val = *((float*)rhs);
  printf("  *(float*)rhs=%f\n", val);
  ////////

  // shortcut for case scalar 
  if (*rank == 0) {   
    char* desc = _XMPF_get_coarrayDesc(*serno);
    int start = _XMPF_get_coarrayStart(*serno, baseAddr);
#ifdef _XMP_COARRAY_FJRDMA
    if (scheme == 1) {
      char *buf = malloc((size_t)(*element));
      rhs = memcpy(buf, rhs, *element);
    }
#endif 
    _putVectorByElement(desc, start, 1, *coindex, rhs);
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

#ifdef _XMP_COARRAY_FJRDMA
    if (scheme == 1) {
      ///////
      printf("  here in #ifdef\n");
      /////////
      size_t bufsize = *element;
      for (int i = 0; i < *rank; i++)
        bufsize *= count[i];
      char *buf = malloc(bufsize);
      ///////
      printf("  rhs = %p\n", rhs);
      /////////
      rhs = memcpy(buf, rhs, bufsize);
      ///////
      printf("  rhs = %p\n", rhs);
      /////////
    }
#endif 
  _putCoarray(*serno, baseAddr, *coindex, rhs, 
              bytes, *rank, skip, count, 0 /*isSpread*/);
}


void _putCoarray(int serno, char *baseAddr, int coindex, char *rhs,
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
      _putVectorByByte(serno, baseAddr, bytes, coindex, rhs);
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
                       rank, skip, count, isSpread);

  if (_XMPF_coarrayMsg) {
    fprintf(stderr, "**** end put\n");
  }
}

  
char *_putVectorIter(int serno, char *baseAddr, int bytes,
                     int coindex, char *src,
                     int loops, int skip[], int count[], int isSpread)
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
                           loops - 1, skip, count, isSpread);
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
