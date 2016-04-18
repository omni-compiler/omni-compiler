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

/* layer 1 */
static void _putCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *rhs,
                            int bytes, int rank, int skip[], int skip_rhs[], int count[],
                            void *descDMA, size_t offsetDMA, char *rhs_name);

static void _putCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *rhs,
                               int bytes, int rank, int skip[], int skip_rhs[], int count[]);

static void _putCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *rhs,
                                   int bytes, int rank, int skip[], int skip_rhs[], int count[]);

/* layer 2 */
static void _putVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                               int rank, int skip[], int skip_kind[], int count[],
                               void *descDMA, size_t offsetDMA, char *rhs_name);

static void _putVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                                  char *src, int rank, int skip[], int skip_kind[], int count[]);

static void _putVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                                      char *rhs, int rank, int skip[], int skip_rhs[], int count[],
                                      int contiguity);

static void _putVectorIter_bufferPack_1(char *rhs, int bytes,
                                        int rank, int skip[], int skip_rhs[], int count[]);

/* layer 3 */
static void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                           void *descDMA, size_t offsetDMA, char *nameDMA);

static void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                              int coindex, char *rhs, int bytes);

static void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                                   int coindex, char *rhs, int bytes);

/* handling local bufer */
static void _init_localBuf(void *descPtr, char *dst, int coindex);
static void _push_localBuf(char *src, int bytes);
static void _flush_localBuf(void);

/* spread-communication */
static void _spreadCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                           int bytes, int rank, int skip[], int count[],
                           int element);

static char *_spreadVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                               char *src, int rank, int skip[], int count[],
                               int element);

static void _spreadVector_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                                 char *rhs, int element);


static void _debugPrint_putCoarray(int bytes, int rank,
                                   int skip[], int skip_rhs[], int count[]);

/***************************************************\
    initialization
\***************************************************/

/* static infos */
void * _localBuf_desc;           // descriptor of the memory pool
size_t _localBuf_offset;         // offset of the local buffer in the memory pool
char * _localBuf_baseAddr;       // local base address of the local buffer
int    _localBuf_size;           // size of the local buffer
char * _localBuf_name;           // name of the local buffer

/* dynamic infos */
int    _localBuf_used;          // length of valid data in localBuf
void * _target_desc;
char * _target_baseAddr;
int    _target_coindex;


void _XMPF_coarrayInit_put()
{
  _localBuf_desc = _XMPF_get_localBufCoarrayDesc(&_localBuf_baseAddr,
                                                 &_localBuf_offset,
                                                 &_localBuf_name);
  _localBuf_size = XMPF_get_localBufSize();
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
  int coindex0 = _XMPF_get_initial_image(*coindex);

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

    _putVector_DMA(*descPtr, *baseAddr, *element, coindex0,
                   descDMA, offsetDMA, nameDMA);
    break;
    
  case SCHEME_ExtraDirectPut:
    elementRU = ROUND_UP_BOUNDARY(*element);
    _XMPF_coarrayDebugPrint("SCHEME_ExtraDirectPut/scalar selected. elementRU=%ud\n",
                            elementRU);
    assert(avail_DMA);

    _putVector_DMA(*descPtr, *baseAddr, elementRU, coindex0,
                   descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferPut:
    _XMPF_coarrayDebugPrint("SCHEME_BufferPut/scalar selected\n");

    _putVector_buffer(*descPtr, *baseAddr, *element, coindex0,
                      rhs, *element);
    break;

  case SCHEME_ExtraBufferPut:
    elementRU = ROUND_UP_BOUNDARY(*element);
    _XMPF_coarrayDebugPrint("SCHEME_ExtraBufferPut/scalar selected. elementRU=%ud\n",
                            elementRU);

    _putVector_buffer(*descPtr, *baseAddr, elementRU, coindex0,
                      rhs, *element);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


#if PUT_INTERFACE_TYPE == 8
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char **rhsAddr, int *rank,
                                    int skip[], int skip_rhs[], int count[])
#else
extern void xmpf_coarray_put_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...)
#endif
{
  int coindex0 = _XMPF_get_initial_image(*coindex);

  if (*element % ONESIDED_BOUNDARY != 0) {
    _XMP_fatal("violation of boundary writing to a coindexed variable\n"
               "  xmpf_coarray_put_array_, " __FILE__);
    return;
  }

#if PUT_INTERFACE_TYPE == 8
  char *rhs = *rhsAddr;

#else
  int *skip_rhs = NULL;    // means RHS is always fully contiguous.

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
    _XMPF_coarrayDebugPrint("SCHEME_DirectPut/array selected\n");
    _putCoarray_DMA(*descPtr, *baseAddr, coindex0, rhs,
                    *element, *rank, skip, skip_rhs, count,
                    descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferPut:
    _XMPF_coarrayDebugPrint("SCHEME_BufferPut/array selected\n");
    _putCoarray_buffer(*descPtr, *baseAddr, coindex0, rhs,
                       *element, *rank, skip, skip_rhs, count);
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
  int coindex0 = _XMPF_get_initial_image(*coindex);

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
  _XMPF_coarrayDebugPrint("SCHEME_BufferPut/spread selected\n");

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  _spreadCoarray(*descPtr, *baseAddr, coindex0, rhs,
                 *element, *rank, skip, count, *element);
}


/***************************************************\
    entry for error messages
\***************************************************/

void xmpf_coarray_put_err_len_(void **descPtr,
                               int *len_mold, int *len_src)
{
  char *name = _XMPF_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("ERROR DETECTED: xmpf_coarray_put_err_len_\n"
                          "  coarray name=\'%s\', len(mold)=%d, len(src)=%d\n",
                          name, *len_mold, *len_src);

  _XMPF_coarrayFatal("mismatch length-parameters found in "
                     "put-communication on coarray \'%s\'", name);
}


void xmpf_coarray_put_err_size_(void **descPtr, int *dim,
                                int *size_mold, int *size_src)
{
  char *name = _XMPF_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("ERROR DETECTED: xmpf_coarray_put_err_size_\n"
                          "  coarray name=\'%s\', i=%d, size(mold,i)=%d, size(src,i)=%d\n",
                          name, *dim, *size_mold, *size_src);

  _XMPF_coarrayFatal("Mismatch sizes of %d-th dimension found in "
                     "put-communication on coarray \'%s\'", *dim, name);
}


/***************************************************\
    layer 1: putCoarray
    collapsing contiguous axes
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


void _putCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *rhs,
                     int bytes, int rank, int skip[], int skip_rhs[], int count[],
                     void *descDMA, size_t offsetDMA, char *nameDMA)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    _putVector_DMA(descPtr, baseAddr, bytes, coindex,
                   descDMA, offsetDMA, nameDMA);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_rhs[0]) {   // The first axis of RHS is contiguous
      // colapse the axis recursively
      _putCoarray_DMA(descPtr, baseAddr, coindex, rhs,
                      bytes * count[0], rank - 1, skip + 1, skip_rhs + 1, count + 1,
                      descDMA, offsetDMA, nameDMA);
      return;
    }
  }

  // Coarray or RHS is non-contiguous

  if (_XMPF_get_coarrayMsg()) {
    char work[200];
    char* p;
    sprintf(work, "DMA-RDMA PUT %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPF_coarrayDebugPrint("=%s bytes ===\n", work);
  }

  _putVectorIter_DMA(descPtr, baseAddr, bytes, coindex,
                     rank, skip, skip_rhs, count,
                     descDMA, offsetDMA, nameDMA);
}

  
void _putCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int skip_rhs[], int count[])
{
  _XMPF_coarrayDebugPrint("=ENTER _putCoarray_buffer(rank=%d)\n", rank);

  if (rank == 0) {  // fully contiguous after perfect collapsing
    _putVector_buffer(descPtr, baseAddr, bytes, coindex,
                      rhs, bytes);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_rhs[0]) {   // The first axis of RHS is contiguous
      // colapse the axis recursively
      _putCoarray_buffer(descPtr, baseAddr, coindex, rhs,
                         bytes * count[0], rank - 1, skip + 1, skip_rhs + 1, count + 1);
      return;
    }
  }

  // Coarray or RHS is non-contiguous

  // select buffer-RDMA or packing buffer-RDMA
  if (bytes != skip[0] || bytes * 2 > _localBuf_size) {

    // Buffer-RDMA scheme selected because:
    //  - the collapsed coarray has no more contiguity between the array elements, or
    //  - the array element is large enough compared with the local buffer.
    if (_XMPF_get_coarrayMsg()) {
      _XMPF_coarrayDebugPrint("=SELECTED Buffer-RDMA\n");
      _debugPrint_putCoarray(bytes, rank, skip, skip_rhs, count);
    }

    _putVectorIter_buffer(descPtr, baseAddr, bytes, coindex, rhs,
                          rank, skip, skip_rhs, count);

  } else {

    // Packing Buffer-RDMA scheme selected because:
    //  - the collapsed coarray still has contiguity between the array elements, and
    //  - The local buffer has room for two or more array elements.

    if (_XMPF_get_coarrayMsg()) {
      _XMPF_coarrayDebugPrint("=SELECTED Packing Buffer-RDMA\n");
      _debugPrint_putCoarray(bytes, rank, skip, skip_rhs, count);
    }

    _putCoarray_bufferPack(descPtr, baseAddr, coindex, rhs,
                           bytes, rank, skip, skip_rhs, count);
  }
}

  
/* Assumption:
 *   - bytes == skip[0], i.e., at least the first dimension of the coarray is contiguous.
 *   - bytes * 2 <= _localBuf_size, i.e., the element is smaller enough than localBuf.
 */
static void _putCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *rhs,
                                   int bytes, int rank, int skip[], int skip_rhs[], int count[])
{
  int k, contiguity, size;

  size = bytes;
  for (k = contiguity = 0; k < rank; k++) {
    if (size != skip[k])    // size < skip[k] if the stride is negative.
      break;
    ++contiguity;
    size *= count[k];
  }

  _XMPF_coarrayDebugPrint("=CALLING _putVectorIter_bufferPack, rank=%d, contiguity=%d\n",
                          rank, contiguity);

  _putVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                            rhs, rank, skip, skip_rhs, count,
                            contiguity);
}


static void _debugPrint_putCoarray(int bytes, int rank,
                                   int skip[], int skip_rhs[], int count[])
{
  char work[200];
  char *p;

  sprintf(work, "src: %d bytes", bytes);
  p = work + strlen(work);
  for (int i = 0; i < rank; i++) {
    sprintf(p, " (stride %d) * %d", skip_rhs[i], count[i]);
    p += strlen(p);
  }
  _XMPF_coarrayDebugPrint("*** %s\n", work);

  sprintf(work, "dst: %d bytes", bytes);
  p = work + strlen(work);
  for (int i = 0; i < rank; i++) {
    sprintf(p, " (stride %d) * %d", skip[i], count[i]);
    p += strlen(p);
  }
  _XMPF_coarrayDebugPrint("*** %s\n", work);
}


/***************************************************\
    layer 2: put iterative vector
\***************************************************/

void _putVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                        int rank, int skip[], int skip_rhs[], int count[],
                        void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_rhs = skip_rhs ? skip_rhs[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _putVector_DMA(descPtr, dst, bytes, coindex,
                     descDMA, offsetDMA, nameDMA);
      dst += gap;
      offsetDMA += gap_rhs;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _putVectorIter_DMA(descPtr, dst, bytes, coindex,
                       rank - 1, skip, skip_rhs, count,
                       descDMA, offsetDMA, nameDMA);
    dst += gap;
    offsetDMA += gap_rhs;
  }

  return;
}


void _putVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                           char *src, int rank, int skip[], int skip_rhs[], int count[])
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_rhs = skip_rhs ? skip_rhs[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _putVector_buffer(descPtr, dst, bytes, coindex,
                        src, bytes);
      dst += gap;
      src += gap_rhs;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _putVectorIter_buffer(descPtr, dst, bytes,
                          coindex, src,
                          rank - 1, skip, skip_rhs, count);
    dst += gap;
    src += gap_rhs;
  }
}


void _putVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                                char *rhs, int rank, int skip[], int skip_rhs[], int count[],
                                int contiguity)
{
  assert(rank >= 1);

  //  _XMPF_coarrayDebugPrint("==PUT VECTOR-ITER Packing-buffer, recursive call (rank=%d)\n"
  //                          "  contiguity=%d, baseAddr=%p, rhs=%p\n",
  //                          rank, contiguity, baseAddr, rhs);

  if (contiguity == rank) {     // the collapsed coarray is fully contiguous.
    _init_localBuf(descPtr, baseAddr, coindex);
    _putVectorIter_bufferPack_1(rhs, bytes,
                                rank, skip, skip_rhs, count);
    _flush_localBuf();

    return;
  }

  // recursive call
  int n = count[rank - 1];
  int gap_rhs = skip_rhs[rank - 1];

  char *src = rhs;
  for (int i = 0; i < n; i++) {
    _putVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                              src, rank - 1, skip, skip_rhs, count,
                              contiguity);
    src += gap_rhs;
  }
}
  

/* Assumption:
 *   - The coarray is fully contiguous in this range of rank.
 * Local buffer is being used.
 */
void _putVectorIter_bufferPack_1(char *rhs, int bytes,
                                 int rank, int skip[], int skip_rhs[], int count[])
{
  char *src = rhs;
    
  if (rank == 1) {
    for (int i = 0; i < count[0]; i++) {
      _push_localBuf(src, bytes);
      src += skip_rhs[0];
    }
    return;
  }

  int n = count[rank - 1];
  int gap_rhs =  skip_rhs[rank - 1];

  for (int i = 0; i < n; i++) {
    _putVectorIter_bufferPack_1(src, bytes,
                                rank - 1, skip, skip_rhs, count);
    src += gap_rhs;
  }
}


/***************************************************\
    layer 3: putVector
\***************************************************/

void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                    void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  _XMPF_coarrayDebugPrint("===PUT_VECTOR DMA-RDMA to[%d], %d bytes\n"
                          "  local : \'%s\', offset=%zd\n"
                          "  remote: \'%s\', offset=%zd\n",
                          coindex, bytes,
                          nameDMA, offsetDMA,
                          _XMPF_get_coarrayName(descPtr), offset);

  // ACTION
  _XMP_coarray_shortcut_put(coindex,
                            desc,   descDMA,
                            offset, offsetDMA,
                            bytes,  bytes);
}


void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                       int coindex, char *rhs, int bytes)
{
  if (XMPF_isSafeBufferMode()) {
    _putVector_buffer_SAFE(descPtr, baseAddr, bytesRU,
                           coindex, rhs, bytes);
    return;
  }

  _init_localBuf(descPtr, baseAddr, coindex);
  _push_localBuf(rhs, bytes);
  _flush_localBuf();

  /***********************************************************
  size_t rest1, rest2, bufSize;
  char *src, *dst;

  src = rhs;
  dst = baseAddr;
  bufSize = _localBuf_size;

  // communication for every buffer size
  for (rest1 = bytesRU, rest2 = bytes;
       rest1 > bufSize;
       rest1 -= bufSize, rest2 -=bufSize) {

    _XMPF_coarrayDebugPrint("===MEMCPY %d bytes, cont\'d\n"
                            "  from: addr=%p\n"
                            "  to  : \'%s\'\n",
                            bufSize,
                            src,
                            _localBuf_name);

    (void)memcpy(_localBuf_baseAddr, src, bufSize);

    _putVector_DMA(descPtr, dst, bufSize, coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name);

    src += bufSize;
    dst += bufSize;
  }

  _XMPF_coarrayDebugPrint("===MEMCPY %d bytes, final\n"
                          "  from: addr=%p\n"
                          "  to  : \'%s\'\n",
                          rest2,
                          src,
                          _localBuf_name);
  (void)memcpy(_localBuf_baseAddr, src, rest2);

  _putVector_DMA(descPtr, dst, rest1, coindex,
                 _localBuf_desc, _localBuf_offset, _localBuf_name);

  ***********************************************/
}


// SAFE mode without using localBuf
//
void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                            int coindex, char *rhs, int bytes)
{
  char *desc = _XMPF_get_coarrayDesc(descPtr);
  size_t offset = _XMPF_get_coarrayOffset(descPtr, baseAddr);

  // MALLOC & MEMCPY
  char *buf = (char*)_XMP_alloc(sizeof(char) * bytesRU);

  _XMPF_coarrayDebugPrint("===MEMCPY, SAFE MODE, %d bytes\n"
                          "  from: addr=%p\n"
                          "  to  : addr=%p\n",
                          bytes,
                          rhs,
                          buf);
  (void)memcpy(buf, rhs, bytes);

  _XMPF_coarrayDebugPrint("===PUT_VECTOR RDMA to [%d], SAFE MODE, %d bytes\n"
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
  _XMPF_coarrayDebugPrint("===DO NOT FREE in SAFE MODE\n"
                          "  addr=%p\n",
                          buf);
  //_XMP_free(buf);  // for safe, changing the address every time
}


/***************************************************\
    spread communication
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
    _XMPF_coarrayDebugPrint("=%s bytes\n", work);
  }

  src = _spreadVectorIter(descPtr, baseAddr, bytes, coindex, src,
                          rank, skip, count, element);
}

  
char *_spreadVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                        char *src, int rank, int skip[], int count[],
                        int element)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];

  if (rank == 1) {
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
                            rank - 1, skip, count, element);
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
  bufSize = _localBuf_size;

  // communication for every buffer size
  for (rest = bytes;
       rest > bufSize;
       rest -= bufSize) {
    _XMPF_coarrayDebugPrint("===SPREAD %d-byte scalar to %d bytes, continued\n"
                            "  from: addr=%p\n"
                            "  to  : \'%s\'\n",
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

  _XMPF_coarrayDebugPrint("===SPREAD %d-byte scalar to %d bytes\n"
                          "  from: addr=%p\n"
                          "  to  : \'%s\'\n",
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



/***************************************************\
    handling local buffer
\***************************************************/

void _init_localBuf(void *descPtr, char *dst, int coindex)
{
  _target_desc = descPtr;
  _target_baseAddr = dst;
  _target_coindex = coindex;
  _localBuf_used = 0;
}


void _push_localBuf(char *src0, int bytes0)
{
  char *src = src0;
  int bytes = bytes0;

  if (_localBuf_used + bytes >= _localBuf_size) {
      _flush_localBuf();

      // for huge data
      while (bytes > _localBuf_size) {
        _XMPF_coarrayDebugPrint("===MEMCPY to localBuf, %d bytes (cont\'d)\n"
                                "  from: addr=%p\n"
                                "  to  : localBuf, offset=0\n",
                                _localBuf_size, src);

        (void)memcpy(_localBuf_baseAddr, src, _localBuf_size);
        _localBuf_used = _localBuf_size;

        _flush_localBuf();

        src += _localBuf_size;
        bytes -= _localBuf_size;
      }
  }    

  if (bytes == 0)
    return;

  _XMPF_coarrayDebugPrint("===MEMCPY to localBuf, %d bytes (final)\n"
                          "  from: addr=%p\n"
                          "  to  : localBuf, offset=%d\n",
                          _localBuf_size, src, _localBuf_used);

  (void)memcpy(_localBuf_baseAddr + _localBuf_used, src, bytes);
  _localBuf_used += bytes;
}


void _flush_localBuf()
{
  if (_localBuf_used > 0) {
    _putVector_DMA(_target_desc, _target_baseAddr, _localBuf_used, _target_coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name);
    _target_baseAddr += _localBuf_used;
    _localBuf_used = 0;
  }
}


