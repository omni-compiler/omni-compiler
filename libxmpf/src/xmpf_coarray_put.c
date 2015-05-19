/*
 *   COARRAY PUT
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

// communication schemes
#define SCHEME_DirectPut       10   // RDMA expected
#define SCHEME_BufferPut       11   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectPut  12   // DirectPut with extra data
#define SCHEME_ExtraBufferPut  13   // BufferPut with extra data

static int _select_putscheme_scalar(int condition, int element);
static int _select_putscheme_array(int condition);

static void _putCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int count[]);

static char *_putVectorIter(void *descPtr, char *baseAddr, int bytes,
                            int coindex, char *src,
                            int loops, int skip[], int count[]);

static void _putVector(void *descPtr, char *baseAddr, int bytes,
                       int coindex, char* src);
#if 0
/* disused */
static void _putVectorByByte(void *descPtr, char *baseAddr, int bytes,
                             int coindex, char* src);
static void _putVectorByElement(char *desc, int start, int vlength,
                                int coindex, char* src);
#endif

/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition)
{
  _XMPF_checkIfInTask("scalar coindexed variable");

  int scheme = _select_putscheme_scalar(*condition, *element);

  switch (scheme) {
  case SCHEME_DirectPut:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_DirectPut/scalar\n"
                              "  *baseAddr=%p, *element=%d\n",
                              *baseAddr, *element);
    }
    _putVector(*descPtr, *baseAddr, *element, *coindex, rhs);
    break;
    
  case SCHEME_ExtraDirectPut:
    {
      size_t elementRU = ROUND_UP_BOUNDARY(*element);

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select SCHEME_ExtraDirectPut/scalar\n"
                                "  *baseAddr=%p, *element=%d, elementRU=%zd\n",
                                *baseAddr, *element, elementRU);
      }
      _putVector(*descPtr, *baseAddr, elementRU, *coindex, rhs);
    }
    break;

  case SCHEME_BufferPut:
    {
      char buf[*element];

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select SCHEME_BufferPut/scalar\n"
                                "  *baseAddr=%p, *element=%zd, buf=%p\n",
                                *baseAddr, *element, buf);
      }
      (void)memcpy(buf, rhs, *element);
      _putVector(*descPtr, *baseAddr, *element, *coindex, buf);
    }
    break;

  case SCHEME_ExtraBufferPut:
    {
      size_t elementRU = ROUND_UP_BOUNDARY(*element);
      char buf[elementRU];

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select SCHEME_ExtraBufferPut/scalar\n"
                                "  *baseAddr=%p, elementRU=%zd, buf=%p\n",
                                *baseAddr, elementRU, buf);
      }
      (void)memcpy(buf, rhs, *element);
      _putVector(*descPtr, *baseAddr, elementRU, *coindex, buf);
    }
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}



extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...)
{
  _XMPF_checkIfInTask("array coindexed variable");

  size_t bufsize;
  char *buf;
  int i;

  if (*element % BOUNDARY_BYTE != 0) {
    _XMP_fatal("violation of boundary in put communication"
               "xmpf_coarray_put_array_, " __FILE__);
    return;
  }

  int scheme = _select_putscheme_array(*condition);

  char **nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, char**);         // nextAddr1, nextAddr2, ...
    skip[i] = *nextAddr - *baseAddr;
    count[i] = *(va_arg(argList, int*));       // count1, count2, ...
  }

  switch (scheme) {
  case SCHEME_DirectPut:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_DirectPut/array\n");
    }
    _putCoarray(*descPtr, *baseAddr, *coindex, rhs, *element, *rank, skip, count);
    break;

  case SCHEME_BufferPut:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_BufferPut/array\n");
      fprintf(stderr, "  *bufsize=%zd\n", bufsize);
    }
    buf = malloc(bufsize);
    (void)memcpy(buf, rhs, bufsize);
    _putCoarray(*descPtr, *baseAddr, *coindex, buf, *element, *rank, skip, count);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
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

  char **nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, char**);         // nextAddr1, nextAddr2, ...
    skip[i] = *nextAddr - *baseAddr;
    count[i] = *(va_arg(argList, int*));       // count1, count2, ...
  }

  nelems = 1;
  for (i = 0; i < *rank; i++)
    nelems *= count[i];
  bufsize = nelems * (*element);
  buf = malloc(bufsize);
  for (i = 0, p = buf; i < nelems; i++, p += *element)
    (void)memcpy(p, rhs, *element);
  _putCoarray(*descPtr, *baseAddr, *coindex, buf, *element, *rank, skip, count);
}


/***************************************************\
    sub
\***************************************************/

/* see /XcodeML-Exc-Tools/src/exc/xmpF/XMPtransCoarray.java
 *   condition 1: It may be necessary to use buffer copy.
 *                The address of RHS may not be accessed by FJ-RDMA.
 *   condition 0: Otherwise.
 */

int _select_putscheme_scalar(int condition, int element)
{
#ifdef _XMP_FJRDMA
  // Temporary handling: in spite of condition, BufferPut or 
  // ExtraBufferPut will be selected because judgement of condition
  // seems inaccurate.
  //  if (condition >= 1) { 
  if (condition >= 0) {
   if (element % BOUNDARY_BYTE == 0)
      return SCHEME_BufferPut;
    return SCHEME_ExtraBufferPut;
  }
#endif
  
  if (element % BOUNDARY_BYTE == 0)
    return SCHEME_DirectPut;
  return SCHEME_ExtraDirectPut;
}

int _select_putscheme_array(int condition)
{
#ifdef _XMP_FJRDMA
  if (condition >= 1) {
    return SCHEME_BufferPut;
  }
#endif
  
  return SCHEME_DirectPut;
}


void _putCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                 int bytes, int rank, int skip[], int count[])
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("PUT %d bytes fully contiguous ===\n", bytes);
      fprintf(stderr, "  coindex %d puts to %d\n", XMPF_this_image, coindex);
    }
    _putVector(descPtr, baseAddr, bytes, coindex, rhs);
    return;
  }

  if (bytes == skip[0]) {  // contiguous
    _putCoarray(descPtr, baseAddr, coindex, rhs,
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

  src = _putVectorIter(descPtr, baseAddr, bytes, coindex, src,
                       rank, skip, count);
}

  
char *_putVectorIter(void *descPtr, char *baseAddr, int bytes,
                     int coindex, char *src,
                     int loops, int skip[], int count[])
{
  char* dst = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _putVector(descPtr, dst, bytes, coindex, src);
      src += bytes;
      dst += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      src = _putVectorIter(descPtr, baseAddr + i * gap, bytes,
                           coindex, src,
                           loops - 1, skip, count);
    }
  }
  return src;
}


void _putVector(void *descPtr, char *baseAddr, int bytes, int coindex, char *src)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_image_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, src, NULL);
}


#if 0
/* disused
 */
void _putVectorByByte(void *descPtr, char *baseAddr, int bytes,
                      int coindex, char *src)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  int start = _XMPF_get_coarrayStart(descPtr, baseAddr);
  // The element that was recorded when the data was allocated is used.
  int element = _XMPF_get_coarrayElement(descPtr);
  int vlength = bytes / element;

  _putVectorByElement(desc, start, vlength, coindex, src);
}

void _putVectorByElement(char *desc, int start, int vlength,
                         int coindex, char* src)
{
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // LHS
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // RHS
  _XMP_coarray_rdma_image_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, src, NULL);
}

#endif
