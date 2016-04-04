/*
 *   COARRAY PUT
 *
 */

#include <assert.h>
#include "xmpf_internal.h"

// communication schemes
#define SCHEME_DirectPut       10   // RDMA expected
#define SCHEME_BufferPut       11   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectPut  12   // DirectPut with extra data
#define SCHEME_ExtraBufferPut  13   // BufferPut with extra data

static int _select_putscheme_scalar(int element, int avail_DMA);
static int _select_putscheme_array(int avail_DMA);

static void _putCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int count[],
                        void *descDMA, size_t offsetDMA, char *rhs_name);

static char *_putVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                            char *src, int loops, int skip[], int count[],
                            void *descDMA, size_t offsetDMA, char *rhs_name);

static void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                              int coindex, char *rhs, int bytes);

static void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                                   int coindex, char *rhs, int bytes);

static void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                           void *descDMA, size_t offsetDMA, char *nameDMA);

static void _spreadCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                           int bytes, int rank, int skip[], int count[],
                           int element);

static char *_spreadVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                               char *src, int loops, int skip[], int count[],
                               int element);

static void _spreadVector_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                                 char *rhs, int element);

#if 0   //obsolete
static void _putVector(void *descPtr, char *baseAddr, int bytes,
                       int coindex, char* src);
#endif

/***************************************************\
    initialization
\***************************************************/

void *_localBuf_desc;           // descriptor of the memory pool
size_t _localBuf_offset;        // offset of the local buffer in the memory pool
char *_localBuf_baseAddr;       // local base address of the local buffer
char *_localBuf_name;           // name of the local buffer


void _XMPF_coarrayInit_put()
{
  _localBuf_desc = _XMPF_get_localBufCoarrayDesc(&_localBuf_baseAddr,
                                                 &_localBuf_offset,
                                                 &_localBuf_name);
}


/***************************************************\
    entry
\***************************************************/

#if PUT_INTERFACE_TYPE == 8
extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs)
#else
extern void xmpf_coarray_put_scalar_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition)
#endif
{
  _XMPF_checkIfInTask("scalar coindexed variable");

  /*--------------------------------------*\
   * Check whether the local address rhs  *
   * is already registered for DMA and,   *
   * if so, get the descriptor, etc.      *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  int avail_DMA;

  descDMA = _XMPF_get_coarrayDescFromAddr(rhs, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? 1 : 0;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_putscheme_scalar(*element, avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  size_t elementRU;

  switch (scheme) {
  case SCHEME_DirectPut:
    _XMPF_coarrayDebugPrint("SCHEME_DirectPut/scalar selected\n");
    assert(avail_DMA);

    _putVector_DMA(*descPtr, *baseAddr, *element, *coindex,
                   descDMA, offsetDMA, nameDMA);
    break;
    
  case SCHEME_ExtraDirectPut:
    elementRU = ROUND_UP_BOUNDARY(*element);
    _XMPF_coarrayDebugPrint("SCHEME_ExtraDirectPut/scalar selected. elementRU=%ud\n",
                            elementRU);
    assert(avail_DMA);

    _putVector_DMA(*descPtr, *baseAddr, elementRU, *coindex,
                   descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferPut:
    _XMPF_coarrayDebugPrint("SCHEME_BufferPut/scalar selected\n");

    _putVector_buffer(*descPtr, *baseAddr, *element, *coindex,
                      rhs, *element);
    break;

  case SCHEME_ExtraBufferPut:
    elementRU = ROUND_UP_BOUNDARY(*element);
    _XMPF_coarrayDebugPrint("SCHEME_ExtraBufferPut/scalar selected. elementRU=%ud\n",
                            elementRU);

    _putVector_buffer(*descPtr, *baseAddr, elementRU, *coindex,
                      rhs, *element);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


#if PUT_INTERFACE_TYPE == 8
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *rhs, int *rank,
                                    int skip[], int skip_rhs[], int count[])
#else
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...)
#endif
{
  _XMPF_checkIfInTask("an array coindexed variable");

  if (*element % ONESIDED_BOUNDARY != 0) {
    _XMP_fatal("violation of boundary writing to a coindexed variable\n"
               "  xmpf_coarray_put_array_, " __FILE__);
    return;
  }

#if PUT_INTERFACE_TYPE != 8
  /*--------------------------------------*\
   *   argument analysis                  *
  \*--------------------------------------*/
  va_list argList;
  va_start(argList, rank);

  char **nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, char**);         // nextAddr1, nextAddr2, ...
    skip[i] = *nextAddr - *baseAddr;
    count[i] = *(va_arg(argList, int*));       // count1, count2, ...
  }
#endif

  /*--------------------------------------*\
   * Check whether the local address rhs  *
   * is already registered for DMA and,   *
   * if so, get the descriptor, etc.      *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  int avail_DMA;

  descDMA = _XMPF_get_coarrayDescFromAddr(rhs, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_putscheme_array(avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  switch (scheme) {
  case SCHEME_DirectPut:
    _XMPF_coarrayDebugPrint("select SCHEME_DirectPut/array\n");
    _putCoarray(*descPtr, *baseAddr, *coindex, rhs,
                *element, *rank, skip, count,
                descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferPut:
    _XMPF_coarrayDebugPrint("select SCHEME_BufferPut/array\n");
    _putCoarray(*descPtr, *baseAddr, *coindex, rhs,
                *element, *rank, skip, count,
                NULL, 0, "(localBuf)");
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


#if PUT_INTERFACE_TYPE == 8
extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *rank,
                                     int skip[], int count[])
#else
extern void xmpf_coarray_put_spread_(void **descPtr, char **baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition,
                                     int *rank, ...)
#endif
{
  _XMPF_checkIfInTask("an array coindexed variable (spread)");

  if (*element % ONESIDED_BOUNDARY != 0) {
    _XMP_fatal("violation of boundary writing a scalar to a coindexed variable\n"
               "   xmpf_coarray_put_spread_, " __FILE__);
    return;
  }

#if PUT_INTERFACE_TYPE != 8
  /*--------------------------------------*\
   *   argument analysis                  *
  \*--------------------------------------*/
  va_list argList;
  va_start(argList, rank);

  char **nextAddr;
  int skip[MAX_RANK];
  int count[MAX_RANK];

  for (int i = 0; i < *rank; i++) {
    nextAddr = va_arg(argList, char**);         // nextAddr1, nextAddr2, ...
    skip[i] = *nextAddr - *baseAddr;
    count[i] = *(va_arg(argList, int*));       // count1, count2, ...
  }
#endif

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  // only BufferPut
  _XMPF_coarrayDebugPrint("select nothing but SCHEME_BufferPut/spread\n");

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  _spreadCoarray(*descPtr, *baseAddr, *coindex, rhs,
                 *element, *rank, skip, count, *element);
}


/***************************************************\
    entry for error messages
\***************************************************/

void xmpf_coarray_put_err_len_(void **descPtr,
                               int *len_mold, int *len_src)
{
  char *name = _XMPF_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("xmpf_coarray_put_err_len_\n"
                          "  coarray name=\'%s\', len(mold)=%d, len(src)=%d\n",
                          name, *len_mold, *len_src);

  _XMPF_coarrayFatal("mismatch length-parameters found in "
                     "put-communication on coarray \'%s\'", name);
}


void xmpf_coarray_put_err_size_(void **descPtr, int *dim,
                                int *size_mold, int *size_src)
{
  char *name = _XMPF_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("xmpf_coarray_put_err_size_\n"
                          "  coarray name=\'%s\', i=%d, size(mold,i)=%d, size(src,i)=%d\n",
                          name, *dim, *size_mold, *size_src);

  _XMPF_coarrayFatal("Mismatch sizes of %d-th dimension found in "
                     "put-communication on coarray \'%s\'", *dim, name);
}


/***************************************************\
    sub
\***************************************************/

/* REMARKING CONDITIONS:
 *  - The result variable may be invisible to FJ-RDMA.
 *  - The length of put/get communication must be divisible by
 *    ONESIDED_BOUNDARY. Else, SCHEME_Extra... should be selected.
 *  - Array element of coarray is divisible by ONESIDED_BOUNDARY
 *    due to a restriction.
 */
int _select_putscheme_scalar(int element, int avail_DMA)
{
  if (avail_DMA)
    if (element % ONESIDED_BOUNDARY == 0)
      return SCHEME_DirectPut;
    else
      return SCHEME_ExtraDirectPut;
  else
    if (element % ONESIDED_BOUNDARY == 0)
      return SCHEME_BufferPut;
    else
      return SCHEME_ExtraBufferPut;
}

int _select_putscheme_array(int avail_DMA)
{
  if (avail_DMA)
    return SCHEME_DirectPut;

  return SCHEME_BufferPut;
}


/***************************************************\
    sub: putCoarray, putVector
\***************************************************/

void _putCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                 int bytes, int rank, int skip[], int count[],
                 void *descDMA, size_t offsetDMA, char *nameDMA)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (descDMA != NULL)
      _putVector_DMA(descPtr, baseAddr, bytes, coindex,
                     descDMA, offsetDMA, nameDMA);
    else
      _putVector_buffer(descPtr, baseAddr, bytes, coindex,
                        rhs, bytes);
    return;
  }

  if (bytes == skip[0]) {  // The first axis is contiguous
    // colapse the axis recursively
    _putCoarray(descPtr, baseAddr, coindex, rhs,
                bytes * count[0], rank - 1, skip + 1, count + 1,
                descDMA, offsetDMA, nameDMA);
    return;
  }

  // not contiguous any more
  char* src = rhs;

  if (_XMPF_get_coarrayMsg()) {
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
                       rank, skip, count,
                       descDMA, offsetDMA, nameDMA);
}

  
char *_putVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                     char *src, int loops, int skip[], int count[],
                     void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* dst = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    if (descDMA != NULL) {   // DMA available
      for (int i = 0; i < n; i++) {
        _putVector_DMA(descPtr, dst, bytes, coindex,
                       descDMA, offsetDMA, nameDMA);
        src += bytes;
        dst += gap;
      }
    } else {     // recursive putVector with static buffer
      for (int i = 0; i < n; i++) {
        _putVector_buffer(descPtr, dst, bytes, coindex,
                          src, bytes);
        src += bytes;
        dst += gap;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    src = _putVectorIter(descPtr, baseAddr + i * gap, bytes,
                         coindex, src,
                         loops - 1, skip, count,
                         descDMA, offsetDMA, nameDMA);
  }
  return src;
}


void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                       int coindex, char *rhs, int bytes)
{
  size_t rest1, rest2, bufSize;
  char *src, *dst;

  if (XMPF_isSafeBufferMode()) {
    _putVector_buffer_SAFE(descPtr, baseAddr, bytesRU,
                           coindex, rhs, bytes);
    return;
  }

  src = rhs;
  dst = baseAddr;
  bufSize = XMPF_get_localBufSize();

  // communication for every buffer size
  for (rest1 = bytesRU, rest2 = bytes;
       rest1 > bufSize;
       rest1 -= bufSize, rest2 -=bufSize) {

    _XMPF_coarrayDebugPrint("MEMCPY %d bytes, continued\n"
                            "  src: addr=%p\n"
                            "  dst: \'%s\'\n",
                            bufSize,
                            src,
                            _localBuf_name);
    (void)memcpy(_localBuf_baseAddr, src, bufSize);

    _putVector_DMA(descPtr, dst, bufSize, coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name);

    src += bufSize;
    dst += bufSize;
  }

  _XMPF_coarrayDebugPrint("MEMCPY %d bytes, final\n"
                          "  src: addr=%p\n"
                          "  dst: \'%s\'\n",
                          rest2,
                          src,
                          _localBuf_name);
  (void)memcpy(_localBuf_baseAddr, src, rest2);

  _putVector_DMA(descPtr, dst, rest1, coindex,
                 _localBuf_desc, _localBuf_offset, _localBuf_name);
}


void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                    void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  _XMPF_coarrayDebugPrint("to [%d] PUT_VECTOR DMA-RDMA, %d bytes\n"
                          "  src (DMA) : \'%s\', offset=%zd\n"
                          "  dst (RDMA): \'%s\', offset=%zd\n",
                          coindex, bytes,
                          nameDMA, offsetDMA,
                          _XMPF_get_coarrayName(descPtr), offset);

  // ACTION
  _XMP_coarray_shortcut_put(coindex,
                            desc,   descDMA,
                            offset, offsetDMA,
                            bytes,  bytes);
}


// SAFE mode without using localBuf
//
void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                            int coindex, char *rhs, int bytes)
{
  char *desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  // MALLOC & MEMCPY
  char *buf = (char*)malloc(sizeof(char) * bytesRU);

  _XMPF_coarrayDebugPrint("MEMCPY, SAFE MODE, %d bytes\n"
                          "  src: addr=%p\n"
                          "  dst: addr=%p\n",
                          bytes,
                          rhs,
                          buf);
  (void)memcpy(buf, rhs, bytes);

  _XMPF_coarrayDebugPrint("to [%d] PUT_VECTOR RDMA, SAFE MODE, %d bytes\n"
                          "  source            : dynamically-allocated buffer, addr=%p\n"
                          "  destination (RDMA): \'%s\', offset=%zd\n",
                          coindex, bytes,
                          buf,
                          _XMPF_get_coarrayName(descPtr), offset);

  // ACTION
  _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // LHS (remote)
  _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // RHS (local)
  _XMP_coarray_rdma_image_set_1(coindex);
  _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, buf, NULL);

  // FREE
  _XMPF_coarrayDebugPrint("FREE, SAFE MODE\n"
                          "  addr=%p\n",
                          buf);
  free(buf);
}


/***************************************************\
    sub: spreadCoarray, spreadVector
\***************************************************/

void _spreadCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                    int bytes, int rank, int skip[], int count[],
                    int element)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    _spreadVector_buffer(descPtr, baseAddr, bytes, coindex, rhs, element);
    return;
  }

  if (bytes == skip[0]) {  // The first axis is contiguous
    // colapse the axis recursively
    _spreadCoarray(descPtr, baseAddr, coindex, rhs,
                   bytes * count[0], rank - 1, skip + 1, count + 1,
                   element);
    return;
  }

  // not contiguous any more
  char* src = rhs;

  if (_XMPF_get_coarrayMsg()) {
    char work[200];
    char* p;
    sprintf(work, "SPREAD %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPF_coarrayDebugPrint("%s bytes ===\n", work);
  }

  src = _spreadVectorIter(descPtr, baseAddr, bytes, coindex, src,
                          rank, skip, count, element);
}

  
char *_spreadVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                        char *src, int loops, int skip[], int count[],
                        int element)
{
  char* dst = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    // DMA is not used.

    for (int i = 0; i < n; i++) {
      _spreadVector_buffer(descPtr, dst, bytes, coindex,
                           src, element);
      src += bytes;
      dst += gap;
    }
  }

  for (int i = 0; i < n; i++) {
    src = _spreadVectorIter(descPtr, baseAddr + i * gap, bytes,
                            coindex, src,
                            loops - 1, skip, count, element);
  }
  return src;
}

void _spreadVector_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                          char *rhs, int element)
{
  size_t rest, bufSize;
  char *src, *dst;

  src = rhs;
  dst = baseAddr;
  bufSize = XMPF_get_localBufSize();

  // communication for every buffer size
  for (rest = bytes;
       rest > bufSize;
       rest -= bufSize) {
    _XMPF_coarrayDebugPrint("SPREAD %d-byte scalar to %d bytes, continued\n"
                            "  src: addr=%p\n"
                            "  dst: \'%s\'\n",
                            element, bufSize,
                            src,
                            _localBuf_name);
    for (char *p = _localBuf_baseAddr;
         p < _localBuf_baseAddr + bufSize;
         p += element)
      (void)memcpy(p, src, element);

    _putVector_DMA(descPtr, dst, bufSize, coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name);

    src += bufSize;
    dst += bufSize;
  }

  _XMPF_coarrayDebugPrint("SPREAD %d-byte scalar to %d bytes\n"
                          "  src: addr=%p\n"
                          "  dst: \'%s\'\n",
                          element, rest,
                          src,
                          _localBuf_name);
  for (char *p = _localBuf_baseAddr;
       p < _localBuf_baseAddr + rest;
       p += element)
    (void)memcpy(p, src, element);

  _putVector_DMA(descPtr, dst, rest, coindex,
                 _localBuf_desc, _localBuf_offset, _localBuf_name);
}



#if 0    
//////////////////////////////// obsolete
    bufsize = *element;
    for (i = 0; i < *rank; i++) {
      bufsize *= count[i];
    }

    //////////////////////////////
    // if (FALSE) {
    //////////////////////////////
    if (bufsize <= XMPF_get_localBufSize()) {
      // using static buffer sharing in the memory pool
      _XMPF_coarrayDebugPrint("select SCHEME_BufferPut-DMA/array\n"
                              "  bufsize=%zd\n", bufsize);

      (void)memcpy(_localBuf_baseAddr, rhs, bufsize);
      _putCoarray(*descPtr, *baseAddr, *coindex, _localBuf_baseAddr,
                  *element, *rank, skip, count,
                  _localBuf_desc, _localBuf_offset, _localBuf_name);
    } else {
      // default: runtime-allocated buffer for large data
      _XMPF_coarrayDebugPrint("select SCHEME_BufferPut/array\n"
                              "  bufsize=%zd\n", bufsize);

      buf = malloc(bufsize);
      (void)memcpy(buf, rhs, bufsize);
      _putCoarray(*descPtr, *baseAddr, *coindex, buf,
                  *element, *rank, skip, count,
                  NULL, 0, "(runtime buffer)");
      free(buf);
    }
    break;
/////////////////////////////////////////
#endif



#if 0   //obsolete
void _putVector(void *descPtr, char *baseAddr, int bytes, int coindex,
                char *src)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  if ((size_t)bytes <= XMPF_get_localBufSize()) {
    _XMPF_coarrayDebugPrint("to [%d] PUT_VECTOR memcpy-DMA-RDMA, %d bytes\n"
                            "  source      (DMA) : static buffer in the pool, offset=%zd\n"
                            "  destination (RDMA): \'%s\', offset=%zd\n",
                            coindex, bytes,
                            _localBuf_offset,
                            _XMPF_get_coarrayName(descPtr), offset);

    // ACTION
    (void)memcpy(_localBuf_baseAddr, src, bytes);
    _XMP_coarray_shortcut_put(coindex,
                              desc,   _localBuf_desc,
                              offset, _localBuf_offset,
                              bytes,  bytes);

  } else {
    _XMPF_coarrayDebugPrint("to [%d] PUT_VECTOR RDMA, %d bytes\n"
                            "  source            : dynamically-allocated buffer, addr=%p\n"
                            "  destination (RDMA): \'%s\', offset=%zd\n",
                            coindex, bytes,
                            src,
                            _XMPF_get_coarrayName(descPtr), offset);

    // ACTION
    _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // LHS (remote)
    _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // RHS (local)
    _XMP_coarray_rdma_image_set_1(coindex);
    _XMP_coarray_rdma_do(COARRAY_PUT_CODE, desc, src, NULL);
  }
}
#endif


