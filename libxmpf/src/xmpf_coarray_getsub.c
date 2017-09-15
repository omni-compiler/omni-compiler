/*
 *   COARRAY GET SUBROUTINE VERSION (for optimization)
 *
 */

#include <assert.h>
#include "xmpf_internal_coarray.h"

// communication schemes
#define SCHEME_DirectGetsub       10   // RDMA expected
#define SCHEME_BufferGetsub       11   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectGetsub  12   // DirectGetsub with extra data
#define SCHEME_ExtraBufferGetsub  13   // BufferGetsub with extra data

static int _select_getsubscheme_array(int avail_DMA);

/* layer 1 */
static void _getsubCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *local,
                            int bytes, int rank, int skip[], int skip_local[], int count[],
                            void *descDMA, size_t offsetDMA, char *local_name);

static void _getsubCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *local,
                               int bytes, int rank, int skip[], int skip_local[], int count[]);

static void _getsubCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *local,
                                   int bytes, int rank, int skip[], int skip_local[],
                                   int count[]);

/* layer 2 */
static void _getsubVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                               int rank, int skip[], int skip_kind[], int count[],
                               void *descDMA, size_t offsetDMA, char *local_name);

static void _getsubVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                                  char *src, int rank, int skip[], int skip_kind[],
                                  int count[]);

static void _getsubVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                                      char *local, int rank, int skip[], int skip_local[], int count[],
                                      int contiguity);

static void _getsubVectorIter_bufferPack_1(char *local, int bytes,
                                        int rank, int skip[], int skip_local[],
                                        int count[]);

/* layer 3 */
/* defined in xmpf_coarray_get.c */
/* _XMPF_getVector_DMA, _XMPF_getVector_buffer */


/* handling local bufer */
static void _init_localBuf(void *descPtr, char *dst, int coindex);
static void _push_localBuf(char *src, int bytes);
static void _flush_localBuf(void);

static void _debugPrint_getsubCoarray(int bytes, int rank,
                                   int skip[], int skip_local[], int count[]);

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
void * _remote_desc;
char * _remote_baseAddr;
int    _remote_coindex;


void _XMPF_coarrayInit_getsub()
{
  _localBuf_desc = _XMP_CO_get_localBufCoarrayDesc(&_localBuf_baseAddr,
                                                 &_localBuf_offset,
                                                 &_localBuf_name);
  _localBuf_size = XMPF_get_localBufSize();
}



/***************************************************\
    entry
\***************************************************/

extern void xmpf_coarray_getsub_array_(void **descPtr, char **baseAddr, int *element,
                                    int *coindex, char **localAddr, int *rank,
                                    int skip[], int skip_local[], int count[])
{
  int coindex0 = _XMPF_get_initial_image_withDescPtr(*coindex, *descPtr);

  if (*element % COMM_UNIT != 0) {
    _XMP_fatal("violation of boundary writing to a coindexed variable\n"
               "  xmpf_coarray_getsub_array_, " __FILE__);
    return;
  }

  char *local = *localAddr;

  /*--------------------------------------*\
   * Check whether the local address local  *
   * is already registered for DMA and,   *
   * if so, get the descriptor, etc.      *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  BOOL avail_DMA;

  descDMA = XMPF_isEagerCommMode() ? NULL :
      _XMP_CO_get_coarrayDescFromAddr(local, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_getsubscheme_array(avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  switch (scheme) {
  case SCHEME_DirectGetsub:
    _XMPF_coarrayDebugPrint("SCHEME_DirectGetsub/array selected\n");
    _getsubCoarray_DMA(*descPtr, *baseAddr, coindex0, local,
                    *element, *rank, skip, skip_local, count,
                    descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferGetsub:
    _XMPF_coarrayDebugPrint("SCHEME_BufferGetsub/array selected\n");
    _getsubCoarray_buffer(*descPtr, *baseAddr, coindex0, local,
                       *element, *rank, skip, skip_local, count);
    break;

  default:
    _XMP_fatal("unexpected scheme number in " __FILE__);
  }
}


/***************************************************\
    entry for error messages
\***************************************************/

void xmpf_coarray_getsub_err_len_(void **descPtr,
                                  int *len_mold, int *len_src)
{
  char *name = _XMP_CO_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("ERROR DETECTED: xmpf_coarray_getsub_err_len_\n"
                          "  coarray name=\'%s\', len(mold)=%d, len(src)=%d\n",
                          name, *len_mold, *len_src);

  _XMPF_coarrayFatal("mismatch length-parameters found in "
                     "optimized get-communication on coarray \'%s\'", name);
}


void xmpf_coarray_getsub_err_size_(void **descPtr, int *dim,
                                   int *size_mold, int *size_src)
{
  char *name = _XMP_CO_get_coarrayName(*descPtr);

  _XMPF_coarrayDebugPrint("ERROR DETECTED: xmpf_coarray_getsub_err_size_\n"
                          "  coarray name=\'%s\', i=%d, size(mold,i)=%d, size(src,i)=%d\n",
                          name, *dim, *size_mold, *size_src);

  _XMPF_coarrayFatal("Mismatch sizes of %d-th dimension found in "
                     "optimized get-communication on coarray \'%s\'", *dim, name);
}


/***************************************************\
    layer 1: getsubCoarray
    collapsing contiguous axes
\***************************************************/

/* REMARKING CONDITIONS:
 *  - The length of getsub communication must be divisible by
 *    COMM_UNIT. Else, SCHEME_Extra... should be selected.
 *  - Array element of coarray is divisible by COMM_UNIT
 *    due to a restriction.
 */

int _select_getsubscheme_array(int avail_DMA)
{
  if (avail_DMA)
    return SCHEME_DirectGetsub;

  return SCHEME_BufferGetsub;
}


void _getsubCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *local,
                     int bytes, int rank, int skip[], int skip_local[], int count[],
                     void *descDMA, size_t offsetDMA, char *nameDMA)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    _XMPF_getVector_DMA(descPtr, baseAddr, bytes, coindex,
                   descDMA, offsetDMA, nameDMA);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_local[0]) {   // The first axis of LOCAL is contiguous
      // colapse the axis recursively
      _getsubCoarray_DMA(descPtr, baseAddr, coindex, local,
                      bytes * count[0], rank - 1, skip + 1, skip_local + 1, count + 1,
                      descDMA, offsetDMA, nameDMA);
      return;
    }
  }

  // Coarray or LOCAL is non-contiguous

  if (_XMPF_get_coarrayMsg()) {
    char work[200];
    char* p;
    sprintf(work, "DMA-RDMA GETSUB %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPF_coarrayDebugPrint("=%s bytes ===\n", work);
  }

  _getsubVectorIter_DMA(descPtr, baseAddr, bytes, coindex,
                     rank, skip, skip_local, count,
                     descDMA, offsetDMA, nameDMA);
}

  
void _getsubCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *local,
                        int bytes, int rank, int skip[], int skip_local[],
                        int count[])
{
  _XMPF_coarrayDebugPrint("=ENTER _getsubCoarray_buffer(rank=%d)\n", rank);

  if (rank == 0) {  // fully contiguous after perfect collapsing
    _XMPF_getVector_buffer(descPtr, baseAddr, bytes, coindex,
                      local, bytes);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_local[0]) {   // The first axis of LOCAL is contiguous
      // colapse the axis recursively
      _getsubCoarray_buffer(descPtr, baseAddr, coindex, local,
                         bytes * count[0], rank - 1, skip + 1, skip_local + 1,
                         count + 1);
      return;
    }
  }

  // Coarray or LOCAL is non-contiguous

  // select buffer-RDMA or packing buffer-RDMA
  if (bytes != skip[0] || bytes * 2 > _localBuf_size) {

    // Buffer-RDMA scheme selected because:
    //  - the collapsed coarray has no more contiguity between the array elements, or
    //  - the array element is large enough compared with the local buffer.
    if (_XMPF_get_coarrayMsg()) {
      _XMPF_coarrayDebugPrint("=SELECTED Buffer-RDMA\n");
      _debugPrint_getsubCoarray(bytes, rank, skip, skip_local, count);
    }

    _getsubVectorIter_buffer(descPtr, baseAddr, bytes, coindex, local,
                          rank, skip, skip_local, count);

  } else {

    // Packing Buffer-RDMA scheme selected because:
    //  - the collapsed coarray still has contiguity between the array elements, and
    //  - The local buffer has room for two or more array elements.

    if (_XMPF_get_coarrayMsg()) {
      _XMPF_coarrayDebugPrint("=SELECTED Packing Buffer-RDMA\n");
      _debugPrint_getsubCoarray(bytes, rank, skip, skip_local, count);
    }

    _getsubCoarray_bufferPack(descPtr, baseAddr, coindex, local,
                           bytes, rank, skip, skip_local, count);
  }
}

  
/* Assumption:
 *   - bytes == skip[0], i.e., at least the first dimension of the coarray is contiguous.
 *   - bytes * 2 <= _localBuf_size, i.e., the element is smaller enough than localBuf.
 */
static void _getsubCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *local,
                                   int bytes, int rank, int skip[], int skip_local[],
                                   int count[])
{
  int k, contiguity, size;

  size = bytes;
  for (k = contiguity = 0; k < rank; k++) {
    if (size != skip[k])    // size < skip[k] if the stride is negative.
      break;
    ++contiguity;
    size *= count[k];
  }

  _XMPF_coarrayDebugPrint("=CALLING _getsubVectorIter_bufferPack, rank=%d, contiguity=%d\n",
                          rank, contiguity);

  _getsubVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                            local, rank, skip, skip_local, count,
                            contiguity);
}


static void _debugPrint_getsubCoarray(int bytes, int rank,
                                   int skip[], int skip_local[], int count[])
{
  char work[200];
  char *p;

  sprintf(work, "src: %d bytes", bytes);
  p = work + strlen(work);
  for (int i = 0; i < rank; i++) {
    sprintf(p, " (stride %d) * %d", skip_local[i], count[i]);
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
    layer 2: getsub iterative vector
\***************************************************/

void _getsubVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                           int rank, int skip[], int skip_local[], int count[],
                           void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_local = skip_local ? skip_local[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _XMPF_getVector_DMA(descPtr, dst, bytes, coindex,
                     descDMA, offsetDMA, nameDMA);
      dst += gap;
      offsetDMA += gap_local;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _getsubVectorIter_DMA(descPtr, dst, bytes, coindex,
                       rank - 1, skip, skip_local, count,
                       descDMA, offsetDMA, nameDMA);
    dst += gap;
    offsetDMA += gap_local;
  }

  return;
}


void _getsubVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                           char *src, int rank, int skip[], int skip_local[],
                           int count[])
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_local = skip_local ? skip_local[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _XMPF_getVector_buffer(descPtr, dst, bytes, coindex,
                        src, bytes);
      dst += gap;
      src += gap_local;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _getsubVectorIter_buffer(descPtr, dst, bytes,
                          coindex, src,
                          rank - 1, skip, skip_local, count);
    dst += gap;
    src += gap_local;
  }
}


void _getsubVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                               char *local, int rank, int skip[], int skip_local[], int count[],
                               int contiguity)
{
  assert(rank >= 1);

  //  _XMPF_coarrayDebugPrint("==GETSUB VECTOR-ITER Packing-buffer, recursive call (rank=%d)\n"
  //                          "  contiguity=%d, baseAddr=%p, local=%p\n",
  //                          rank, contiguity, baseAddr, local);

  if (contiguity == rank) {     // the collapsed coarray is fully contiguous.
    _init_localBuf(descPtr, baseAddr, coindex);
    _getsubVectorIter_bufferPack_1(local, bytes,
                                rank, skip, skip_local, count);
    _flush_localBuf();

    return;
  }

  // recursive call
  int n = count[rank - 1];
  int gap_local = skip_local[rank - 1];

  char *src = local;
  for (int i = 0; i < n; i++) {
    _getsubVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                              src, rank - 1, skip, skip_local, count,
                              contiguity);
    src += gap_local;
  }
}
  

/* Assumption:
 *   - The coarray is fully contiguous in this range of rank.
 * Local buffer is being used.
 */
void _getsubVectorIter_bufferPack_1(char *local, int bytes,
                                 int rank, int skip[], int skip_local[],
                                 int count[])
{
  char *src = local;
    
  if (rank == 1) {
    for (int i = 0; i < count[0]; i++) {
      _push_localBuf(src, bytes);
      src += skip_local[0];
    }
    return;
  }

  int n = count[rank - 1];
  int gap_local =  skip_local[rank - 1];

  for (int i = 0; i < n; i++) {
    _getsubVectorIter_bufferPack_1(src, bytes,
                                rank - 1, skip, skip_local, count);
    src += gap_local;
  }
}


/***************************************************\
    layer 3: getVector
\***************************************************/

/* see _XMPF_getVector_* in xmpf_coarray_get.c
 */



/***************************************************\
    handling local buffer
\***************************************************/

void _init_localBuf(void *descPtr, char *dst, int coindex)
{
  _remote_desc = descPtr;
  _remote_baseAddr = dst;
  _remote_coindex = coindex;
  _localBuf_used = 0;
}


void _push_localBuf(char *src0, int bytes0)
{
  char *src = src0;
  int bytes = bytes0;
  int copySize;

  if (_localBuf_used + bytes >= _localBuf_size) {
      _flush_localBuf();

      // for huge data
      while (bytes > _localBuf_size) {
        copySize = _localBuf_size;      
        _XMPF_coarrayDebugPrint("===MEMCPY %d of %d bytes to localBuf (cont\'d)\n"
                                "  from: addr=%p\n"
                                "  to  : localBuf\n",
                                copySize, bytes,
                                src);

        (void)memcpy(_localBuf_baseAddr, src, copySize);
        _localBuf_used = copySize;

        _flush_localBuf();

        src += copySize;
        bytes -= copySize;
      }
  }    

  if (bytes == 0)
    return;
  copySize = bytes;

  _XMPF_coarrayDebugPrint("===MEMCPY %d bytes to localBuf (final)\n"
                          "  from: addr=%p\n"
                          "  to  : localBuf + offset(%d bytes)\n",
                          copySize,
                          src,
                          _localBuf_used);

  (void)memcpy(_localBuf_baseAddr + _localBuf_used, src, copySize);
  _localBuf_used += copySize;
}


void _flush_localBuf()
{
  if (_localBuf_used > 0) {
    _XMPF_getVector_DMA(_remote_desc, _remote_baseAddr, _localBuf_used, _remote_coindex,
                        _localBuf_desc, _localBuf_offset, _localBuf_name);
    _remote_baseAddr += _localBuf_used;
    _localBuf_used = 0;
  }
}


