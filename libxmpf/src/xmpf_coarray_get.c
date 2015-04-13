/*
 *   COARRAY GET
 *
 */

#include <stdarg.h>
#include "xmpf_internal.h"

// communication schemes
#define SCHEME_DirectGet       20   // RDMA expected
#define SCHEME_BufferGet       21   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectGet  22   // DirectGet with extra data
#define SCHEME_ExtraBufferGet  23   // BufferGet with extra data

static int _select_getscheme_scalar(int element);
static int _select_getscheme_array(void);

static void _getCoarray(void *descPtr, char *baseAddr, int coindex, char *res,
                        int bytes, int rank, int skip[], int count[]);

static char *_getVectorIter(void *descPtr, char *baseAddr, int bytes,
                            int coindex, char *dst,
                            int loops, int skip[], int count[]);

static void _getVector(void *descPtr, char *baseAddr, int bytes,
                       int coindex, char *dst);
#if 0
/* disused */
static void _getVectorByByte(void *descPtr, char *baseAddr, int bytes,
                             int coindex, char *dst);
static void _getVectorByElement(char *desc, int start, int vlength,
                                int coindex, char *dst);
#endif

/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_get_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *result)
{
  _XMPF_checkIfInTask("a scalar coindexed object");

  int scheme = _select_getscheme_scalar(*element);

  switch (scheme) {
  case SCHEME_DirectGet:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_DirectGet/scalar\n"
                              "  *baseAddr=%p, *element=%d\n",
                              *baseAddr, *element);
    }
    _getVector(*descPtr, *baseAddr, *element, *coindex, result);
    break;

  case SCHEME_BufferGet:
    {
      char buf[*element];

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select SCHEME_BufferGet/scalar\n",
                                "  *baseAddr=%p, *element=%zd, buf=%p\n",
                                *baseAddr, *element, buf);
      }
      _getVector(*descPtr, *baseAddr, *element, *coindex, buf);
      (void)memcpy(result, buf, *element);
    }
    break;

  case SCHEME_ExtraBufferGet:
    {
      size_t elementRU = ROUND_UP_BOUNDARY(*element);
      char buf[elementRU];

      if (_XMPF_coarrayMsg) {
        _XMPF_coarrayDebugPrint("select SCHEME_ExtraBufferGet/scalar\n",
                                "  *baseAddr=%p, elementRU=%zd, buf=%p\n",
                                *baseAddr, elementRU, buf);
      }
      _getVector(*descPtr, *baseAddr, elementRU, *coindex, buf);
      (void)memcpy(result, buf, *element);
    }
    break;

  default:
    _XMP_fatal("undefined scheme number in " __FILE__);
  }
}


extern void xmpf_coarray_get_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *result, int *rank, ...)
{
  _XMPF_checkIfInTask("an array coindexed object");

  size_t bufsize;
  char *buf;
  int i;

  int scheme = _select_getscheme_array();

  char **nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];
  va_list argList;
  va_start(argList, rank);

  if (*element % BOUNDARY_BYTE != 0) {
    _XMP_fatal("violation of boundary in get communication"
               "xmpf_coarray_get_array_, " __FILE__);
    return;
  }

  for (int i = 0; i < *rank; i++) {
    nextAddr = (va_arg(argList, char**));
    skip[i] = *nextAddr - *baseAddr;
    count[i] = *(va_arg(argList, int*));
  }

  switch (scheme) {
  case SCHEME_DirectGet:
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_DirectGet/array\n"
                              "  *baseAddr=%p, *element=%d\n",
                              *baseAddr, *element);
    }
    _getCoarray(*descPtr, *baseAddr, *coindex, result, *element, *rank, skip, count);
    break;

  case SCHEME_BufferGet:
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("select SCHEME_BufferGet/array\n"
                              "  bufsize=%zd\n", bufsize);
    }
    buf = malloc(bufsize);
    _getCoarray(*descPtr, *baseAddr, *coindex, buf, *element, *rank, skip, count);
    (void)memcpy(result, buf, bufsize);
    break;

  default:
    _XMP_fatal("undefined scheme number in " __FILE__);
  }
}


/***************************************************\
    sub
\***************************************************/

int _select_getscheme_scalar(int element)
{
  if (element % BOUNDARY_BYTE > 0)
    return SCHEME_ExtraBufferGet;

  // SCHEME_ExtraDirectGet should not be used because 
  // the extra area overwritten may be valid data.

#ifdef _XMP_FJRDMA
  // The result scalar variable may be invisible to FJ-RDMA.
  return SCHEME_BufferGet;
#else
  return SCHEME_DirectGet;
#endif
}

int _select_getscheme_array(void)
{
#ifdef _XMP_FJRDMA
  // The result array variable may be invisible to FJ-RDMA.
  return SCHEME_BufferGet;
#else
  return SCHEME_DirectGet;
#endif
}


void _getCoarray(void *descPtr, char *baseAddr, int coindex, char *result,
                 int bytes, int rank, int skip[], int count[])
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (_XMPF_coarrayMsg) {
      _XMPF_coarrayDebugPrint("GET %d bytes fully contiguous ===\n", bytes);
    }
    _getVector(descPtr, baseAddr, bytes, coindex, result);
    return;
  }

  if (bytes == skip[0]) {  // contiguous
    _getCoarray(descPtr, baseAddr, coindex, result,
                bytes * count[0], rank - 1, skip + 1, count + 1);
    return;
  }

  // not contiguous any more
  char* dst = result;

  if (_XMPF_coarrayMsg) {
    char work[200];
    char* p;
    sprintf(work, "GET %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPF_coarrayDebugPrint("%s bytes ===\n", work);
  }

  dst = _getVectorIter(descPtr, baseAddr, bytes, coindex, dst,
                       rank, skip, count);
}

  
char *_getVectorIter(void *descPtr, char *baseAddr, int bytes,
                     int coindex, char *dst,
                     int loops, int skip[], int count[])
{
  char* src = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    for (int i = 0; i < n; i++) {
      _getVector(descPtr, src, bytes, coindex, dst);
      dst += bytes;
      src += gap;
    }
  } else {
    for (int i = 0; i < n; i++) {
      dst = _getVectorIter(descPtr, baseAddr + i * gap, bytes,
                           coindex, dst,
                           loops - 1, skip, count);
    }
  }
  return dst;
}


void _getVector(void *descPtr, char *src, int bytes, int coindex, char *dst)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  int offset = _XMPF_get_coarrayOffset(descPtr, src);

  _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // coindexed-object
  _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // result
  _XMP_coarray_rdma_image_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_GET_CODE, desc, dst, NULL);
}


#if 0
/* disused
 */
void _getVectorByByte(void *descPtr, char *src, int bytes,
                      int coindex, char *dst)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  int start = _XMPF_get_coarrayStart(descPtr, src);
  // The element that was recorded when the data was allocated is used.
  int element = _XMPF_get_coarrayElement(descPtr);
  int vlength = bytes / element;

  _getVectorByElement(desc, start, vlength, coindex, dst);
}

void _getVectorByElement(char *desc, int start, int vlength,
                         int coindex, char *dst)
{
  _XMP_coarray_rdma_coarray_set_1(start, vlength, 1);    // coindexed-object
  _XMP_coarray_rdma_array_set_1(0, vlength, 1, 1, 1);    // result
  _XMP_coarray_rdma_image_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_GET_CODE, desc, dst, NULL);
}

#endif
