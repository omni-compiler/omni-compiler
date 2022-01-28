/*
 *   COARRAY PUT
 *
 */

#include <assert.h>
#include "xmpco_internal.h"
#include "_xmpco_putget.h"


// communication schemes
#define SCHEME_DirectPut       10   // RDMA expected
#define SCHEME_BufferPut       11   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectPut  12   // DirectPut with extra data
#define SCHEME_ExtraBufferPut  13   // BufferPut with extra data

static int _select_scheme_put_scalar(int element, int avail_DMA);
static int _select_scheme_put_array(int avail_DMA);

/* layer 1 */
static void _putCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *rhs,
                            int bytes, int rank, int skip[], int skip_rhs[], int count[],
                            void *descDMA, size_t offsetDMA, char *rhs_name,
			    SyncMode sync_mode);

static void _putCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *rhs,
                               int bytes, int rank, int skip[], int skip_rhs[],
			       int count[], SyncMode sync_mode);

static void _putCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *rhs,
                                   int bytes, int rank, int skip[], int skip_rhs[],
                                   int count[], SyncMode sync_mode);

/* layer 2 */
static void _putVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                               int rank, int skip[], int skip_kind[], int count[],
                               void *descDMA, size_t offsetDMA, char *rhs_name,
                               SyncMode sync_mode);

static void _putVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                                  char *src, int rank, int skip[], int skip_kind[],
                                  int count[], SyncMode sync_mode);

static void _putVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                                      char *rhs, int rank, int skip[], int skip_rhs[], int count[],
                                      int contiguity, SyncMode sync_mode);

static void _putVectorIter_bufferPack_1(char *rhs, int bytes,
                                        int rank, int skip[], int skip_rhs[],
                                        int count[], SyncMode sync_mode);

/* layer 3 */
static void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                           void *descDMA, size_t offsetDMA, char *nameDMA,
                           SyncMode sync_mode);

static void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                              int coindex, char *rhs, int bytes,
			      SyncMode sync_mode);

static void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                                   int coindex, char *rhs, int bytes);

/* handling local bufer */
static void _init_localBuf(void *descPtr, char *dst, int coindex);
static void _push_localBuf(char *src, int bytes, SyncMode sync_mode);
static void _flush_localBuf(SyncMode sync_mode);

/* spread-communication */
static void _spreadCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                           int bytes, int rank, int skip[], int count[],
                           int element, SyncMode sync_mode);

static char *_spreadVectorIter(void *descPtr, char *baseAddr,
                               int bytes, int coindex,
                               char *src, int rank, int skip[], int count[],
                               int element, SyncMode sync_mode);

static void _spreadVector_buffer(void *descPtr, char *baseAddr,
                                 int bytes, int coindex,
                                 char *rhs, int element, SyncMode sync_mode);


static void _debugPrint_putCoarray(int bytes, int rank,
                                   int skip[], int skip_rhs[], int count[]);

/***************************************************\
    initialization
\***************************************************/

/* static infos */
static void * _localBuf_desc;           // descriptor of the memory pool
static size_t _localBuf_offset;         // offset of the local buffer in the memory pool
static char * _localBuf_baseAddr;       // local base address of the local buffer
static int    _localBuf_size;           // size of the local buffer
static char * _localBuf_name;           // name of the local buffer

/* dynamic infos */
static int    _localBuf_used;          // length of valid data in localBuf
static void * _target_desc;
static char * _target_baseAddr;
static int    _target_coindex;


void _XMPCO_coarrayInit_put()
{
  _localBuf_desc = _XMPCO_get_infoOfLocalBuf(&_localBuf_baseAddr,
                                              &_localBuf_offset,
                                              &_localBuf_name);
  _localBuf_size = _XMPCO_get_localBufSize();
}


/***************************************************\
    entry
\***************************************************/

void XMPCO_PUT_scalarStmt(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                          int coindex, char *rhs, SyncMode sync_mode)
{
  int coindex0 = _XMPCO_get_initial_image_withDescPtr(coindex, descPtr);

  /*--------------------------------------*\
   * Check whether the local address rhs  *
   * is already registered for DMA and,   *
   * if so, get the descriptor, etc.      *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  BOOL avail_DMA;

  descDMA = _XMPCO_get_isEagerCommMode() ? NULL :
    _XMPCO_get_desc_fromLocalAddr(rhs, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_scheme_put_scalar(element, avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  size_t elementRU;

  switch (scheme) {
  case SCHEME_DirectPut:
    _XMPCO_debugPrint("SCHEME_DirectPut/scalar selected\n");
    assert(avail_DMA);

    _putVector_DMA(descPtr, baseAddr, element, coindex0,
                   descDMA, offsetDMA, nameDMA, sync_mode);
    break;
    
  case SCHEME_ExtraDirectPut:
    elementRU = ROUND_UP_COMM(element);
    _XMPCO_debugPrint("SCHEME_ExtraDirectPut/scalar selected. elementRU=%ud\n",
                      elementRU);
    assert(avail_DMA);

    _putVector_DMA(descPtr, baseAddr, elementRU, coindex0,
                   descDMA, offsetDMA, nameDMA, sync_mode);
    break;

  case SCHEME_BufferPut:
    _XMPCO_debugPrint("SCHEME_BufferPut/scalar selected\n");

    _putVector_buffer(descPtr, baseAddr, element, coindex0,
                      rhs, element, sync_mode);
    break;

  case SCHEME_ExtraBufferPut:
    elementRU = ROUND_UP_COMM(element);
    _XMPCO_debugPrint("SCHEME_ExtraBufferPut/scalar selected. elementRU=%ud\n",
                      elementRU);

    _putVector_buffer(descPtr, baseAddr, elementRU, coindex0,
                      rhs, element, sync_mode);
    break;

  default:
    _XMPCO_fatal("unexpected scheme number in " __FILE__);
  }
}


void XMPCO_PUT_arrayStmt(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                         int coindex, char *rhsAddr, int rank,
                         int skip[], int skip_rhs[], int count[],
                         SyncMode sync_mode)
{
  int coindex0 = _XMPCO_get_initial_image_withDescPtr(coindex, descPtr);

  if (element % COMM_UNIT != 0) {
    _XMPCO_fatal("violation of boundary writing to a coindexed variable\n"
                 "  xmpf_coarray_put_array_, " __FILE__);
    return;
  }

  /*--------------------------------------*\
   * Check whether the local address rhs  *
   * is already registered for DMA and,   *
   * if so, get the descriptor, etc.      *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  BOOL avail_DMA;
  int scheme;

  descDMA = _XMPCO_get_isEagerCommMode() ? NULL :
      _XMPCO_get_desc_fromLocalAddr(rhsAddr, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  scheme = _select_scheme_put_array(avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  switch (scheme) {
  case SCHEME_DirectPut:
    _XMPCO_debugPrint("SCHEME_DirectPut/array selected\n");
    _putCoarray_DMA(descPtr, baseAddr, coindex0, rhsAddr,
                    element, rank, skip, skip_rhs, count,
                    descDMA, offsetDMA, nameDMA, sync_mode);
    break;

  case SCHEME_BufferPut:
    _XMPCO_debugPrint("SCHEME_BufferPut/array selected\n");
    _putCoarray_buffer(descPtr, baseAddr, coindex0, rhsAddr,
                       element, rank, skip, skip_rhs, count, sync_mode);
    break;

  default:
    _XMPCO_fatal("unexpected scheme number in " __FILE__);
  }
}


void XMPCO_PUT_spread(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                      int coindex, char *rhs, int rank,
                      int skip[], int count[], SyncMode sync_mode)
{
  int coindex0 = _XMPCO_get_initial_image_withDescPtr(coindex, descPtr);

  if (element % COMM_UNIT != 0) {
    _XMPCO_fatal("violation of boundary writing a scalar to a coindexed variable\n"
               "   xmpf_coarray_put_spread_, " __FILE__);
    return;
  }


  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  // only BufferPut
  _XMPCO_debugPrint("SCHEME_BufferPut/spread selected\n");

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  _spreadCoarray(descPtr, baseAddr, coindex0, rhs,
                 element, rank, skip, count, element, sync_mode);
}


/***************************************************\
    entry for error messages
\***************************************************/

void xmpf_coarray_put_err_len_(void **descPtr,
                               int *len_mold, int *len_src)
{
  char *name = _XMPCO_get_nameOfCoarray(*descPtr);

  _XMPCO_debugPrint("ERROR DETECTED: xmpf_coarray_put_err_len_\n"
                          "  coarray name=\'%s\', len(mold)=%d, len(src)=%d\n",
                          name, *len_mold, *len_src);

  _XMPCO_fatal("mismatch length-parameters found in "
                     "put-communication on coarray \'%s\'", name);
}


void xmpf_coarray_put_err_size_(void **descPtr, int *dim,
                                int *size_mold, int *size_src)
{
  char *name = _XMPCO_get_nameOfCoarray(*descPtr);

  _XMPCO_debugPrint("ERROR DETECTED: xmpf_coarray_put_err_size_\n"
                          "  coarray name=\'%s\', i=%d, size(mold,i)=%d, size(src,i)=%d\n",
                          name, *dim, *size_mold, *size_src);

  _XMPCO_fatal("Mismatch sizes of %d-th dimension found in "
                     "put-communication on coarray \'%s\'", *dim, name);
}


/***************************************************\
    layer 1: putCoarray
    collapsing contiguous axes
\***************************************************/

/* REMARKING CONDITIONS:
 *  - The length of put communication must be divisible by
 *    COMM_UNIT. Else, SCHEME_Extra... should be selected.
 *  - Array element of coarray is divisible by COMM_UNIT
 *    due to a restriction.
 */

int _select_scheme_put_scalar(int element, int avail_DMA)
{
  if (avail_DMA)
    if (element % COMM_UNIT == 0)
      return SCHEME_DirectPut;
    else
      return SCHEME_ExtraDirectPut;
  else
    if (element % COMM_UNIT == 0)
      return SCHEME_BufferPut;
    else
      return SCHEME_ExtraBufferPut;
}

int _select_scheme_put_array(int avail_DMA)
{
  if (avail_DMA)
    return SCHEME_DirectPut;

  return SCHEME_BufferPut;
}


void _putCoarray_DMA(void *descPtr, char *baseAddr, int coindex, char *rhs,
                     int bytes, int rank, int skip[], int skip_rhs[], int count[],
                     void *descDMA, size_t offsetDMA, char *nameDMA,
		     SyncMode sync_mode)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    _putVector_DMA(descPtr, baseAddr, bytes, coindex,
                   descDMA, offsetDMA, nameDMA, sync_mode);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_rhs[0]) {   // The first axis of RHS is contiguous
      // colapse the axis recursively
      _putCoarray_DMA(descPtr, baseAddr, coindex, rhs,
                      bytes * count[0], rank - 1, skip + 1, skip_rhs + 1, count + 1,
                      descDMA, offsetDMA, nameDMA, sync_mode);
      return;
    }
  }

  // Coarray or RHS is non-contiguous

  if (_XMPCO_get_isMsgMode()) {
    char work[200];
    char* p;
    sprintf(work, "DMA-RDMA PUT %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPCO_debugPrint("=%s bytes ===\n", work);
  }

  _putVectorIter_DMA(descPtr, baseAddr, bytes, coindex,
                     rank, skip, skip_rhs, count,
                     descDMA, offsetDMA, nameDMA, sync_mode);
}

  
void _putCoarray_buffer(void *descPtr, char *baseAddr, int coindex, char *rhs,
                        int bytes, int rank, int skip[], int skip_rhs[],
                        int count[], SyncMode sync_mode)
{
  _XMPCO_debugPrint("=ENTER _putCoarray_buffer(coindex=%d, bytes=%d, rank=%d, sync_mode=%d)\n",
		    coindex, bytes, rank, sync_mode);

  if (rank == 0) {  // fully contiguous after perfect collapsing
    _putVector_buffer(descPtr, baseAddr, bytes, coindex,
                      rhs, bytes, sync_mode);
    return;
  }

  if (bytes == skip[0]) {      // The first axis of the coarray is contiguous
    if (bytes == skip_rhs[0]) {   // The first axis of RHS is contiguous
      // colapse the axis recursively
      _putCoarray_buffer(descPtr, baseAddr, coindex, rhs,
                         bytes * count[0], rank - 1, skip + 1, skip_rhs + 1,
                         count + 1, sync_mode);
      return;
    }
  }

  // Coarray or RHS is non-contiguous

  // select buffer-RDMA or packing buffer-RDMA
  if (bytes != skip[0] || bytes * 2 > _localBuf_size) {

    // Buffer-RDMA scheme selected because:
    //  - the collapsed coarray has no more contiguity between the array elements, or
    //  - the array element is large enough compared with the local buffer.
    if (_XMPCO_get_isMsgMode()) {
      _XMPCO_debugPrint("=SELECTED Buffer-RDMA\n");
      _debugPrint_putCoarray(bytes, rank, skip, skip_rhs, count);
    }

    _putVectorIter_buffer(descPtr, baseAddr, bytes, coindex, rhs,
                          rank, skip, skip_rhs, count, sync_mode);

  } else {

    // Packing Buffer-RDMA scheme selected because:
    //  - the collapsed coarray still has contiguity between the array elements, and
    //  - The local buffer has room for two or more array elements.

    if (_XMPCO_get_isMsgMode()) {
      _XMPCO_debugPrint("=SELECTED Packing Buffer-RDMA\n");
      _debugPrint_putCoarray(bytes, rank, skip, skip_rhs, count);
    }

    _putCoarray_bufferPack(descPtr, baseAddr, coindex, rhs,
                           bytes, rank, skip, skip_rhs, count, sync_mode);
  }
}

  
/* Assumption:
 *   - bytes == skip[0], i.e., at least the first dimension of the coarray is contiguous.
 *   - bytes * 2 <= _localBuf_size, i.e., the element is smaller enough than localBuf.
 */
static void _putCoarray_bufferPack(void *descPtr, char *baseAddr, int coindex, char *rhs,
                                   int bytes, int rank, int skip[], int skip_rhs[],
                                   int count[], SyncMode sync_mode)
{
  int k, contiguity, size;

  size = bytes;
  for (k = contiguity = 0; k < rank; k++) {
    if (size != skip[k])    // size < skip[k] if the stride is negative.
      break;
    ++contiguity;
    size *= count[k];
  }

  _XMPCO_debugPrint("=CALLING _putVectorIter_bufferPack, rank=%d, contiguity=%d\n",
                          rank, contiguity);

  _putVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                            rhs, rank, skip, skip_rhs, count,
                            contiguity, sync_mode);
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
  _XMPCO_debugPrint("*** %s\n", work);

  sprintf(work, "dst: %d bytes", bytes);
  p = work + strlen(work);
  for (int i = 0; i < rank; i++) {
    sprintf(p, " (stride %d) * %d", skip[i], count[i]);
    p += strlen(p);
  }
  _XMPCO_debugPrint("*** %s\n", work);
}


/***************************************************\
    layer 2: put iterative vector
\***************************************************/

void _putVectorIter_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                        int rank, int skip[], int skip_rhs[], int count[],
                        void *descDMA, size_t offsetDMA, char *nameDMA,
                        SyncMode sync_mode)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_rhs = skip_rhs ? skip_rhs[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _putVector_DMA(descPtr, dst, bytes, coindex,
                     descDMA, offsetDMA, nameDMA, sync_mode);
      dst += gap;
      offsetDMA += gap_rhs;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _putVectorIter_DMA(descPtr, dst, bytes, coindex,
                       rank - 1, skip, skip_rhs, count,
                       descDMA, offsetDMA, nameDMA, sync_mode);
    dst += gap;
    offsetDMA += gap_rhs;
  }

  return;
}


void _putVectorIter_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                           char *src, int rank, int skip[], int skip_rhs[],
                           int count[], SyncMode sync_mode)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];
  int gap_rhs = skip_rhs ? skip_rhs[rank - 1] : bytes;

  if (rank == 1) {
    for (int i = 0; i < n; i++) {
      _putVector_buffer(descPtr, dst, bytes, coindex,
                        src, bytes, sync_mode);
      dst += gap;
      src += gap_rhs;
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    _putVectorIter_buffer(descPtr, dst, bytes,
                          coindex, src,
                          rank - 1, skip, skip_rhs, count, sync_mode);
    dst += gap;
    src += gap_rhs;
  }
}


void _putVectorIter_bufferPack(void *descPtr, char *baseAddr, int bytes, int coindex,
                               char *rhs, int rank, int skip[], int skip_rhs[], int count[],
                               int contiguity, SyncMode sync_mode)
{
  assert(rank >= 1);

  if (contiguity == rank) {     // the collapsed coarray is fully contiguous.
    _init_localBuf(descPtr, baseAddr, coindex);
    _putVectorIter_bufferPack_1(rhs, bytes,
                                rank, skip, skip_rhs, count, sync_mode);
    _flush_localBuf(sync_mode);

    return;
  }

  // recursive call
  int n = count[rank - 1];
  int gap_rhs = skip_rhs[rank - 1];

  char *src = rhs;
  for (int i = 0; i < n; i++) {
    _putVectorIter_bufferPack(descPtr, baseAddr, bytes, coindex,
                              src, rank - 1, skip, skip_rhs, count,
                              contiguity, sync_mode);
    src += gap_rhs;
  }
}
  

/* Assumption:
 *   - The coarray is fully contiguous in this range of rank.
 * Local buffer is being used.
 */
void _putVectorIter_bufferPack_1(char *rhs, int bytes,
                                 int rank, int skip[], int skip_rhs[],
                                 int count[], SyncMode sync_mode)
{
  char *src = rhs;
    
  if (rank == 1) {
    for (int i = 0; i < count[0]; i++) {
      _push_localBuf(src, bytes, sync_mode);
      src += skip_rhs[0];
    }
    return;
  }

  int n = count[rank - 1];
  int gap_rhs =  skip_rhs[rank - 1];

  for (int i = 0; i < n; i++) {
    _putVectorIter_bufferPack_1(src, bytes,
                                rank - 1, skip, skip_rhs, count, sync_mode);
    src += gap_rhs;
  }
}


/***************************************************\
    layer 3: putVector
\***************************************************/

void _putVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                    void *descDMA, size_t offsetDMA, char *nameDMA,
                    SyncMode sync_mode)
{
  char* desc = _XMPCO_get_descForMemoryChunk(descPtr);
  size_t offset = _XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr);

  _XMPCO_debugPrint("===putVector_DMA(coindex=%d, bytes=%d, sync_mode=%s)\n"
		    "  local : \'%s\', offset=%zd\n"
		    "  remote: \'%s\', offset=%zd\n",
		    coindex, bytes, sync_mode,
                    (sync_mode == syncNONBLOCK) ? "syncNONBLOCK" :
                    (sync_mode == syncBLOCK) ? "syncBLOCK" :
                    (sync_mode == syncATOMIC) ? "syncATOMIC" :
                    (sync_mode == syncRUNTIME) ? "syncRUNTIME" : "*BROKEN*",
		    nameDMA, offsetDMA,
		    _XMPCO_get_nameOfCoarray(descPtr), offset);

  switch (sync_mode) {

  case syncATOMIC:
    //-------------------------------
    // ATCION: atomic_define
    //-------------------------------
    if (offset % 4 != 0) {
      _XMPCO_fatal("RESTRICSION: boundary error: "
		   "the 1-st argument of atomic_define");
    }
    _XMP_atomic_define_1(desc, offset / 4, coindex-1, 0,
                         descDMA, offsetDMA, 4);
    break;

  case syncNONBLOCK:

  case syncBLOCK:

  case syncRUNTIME:
    //-------------------------------
    // ATCION: PUT buffer-to-RDMA
    //-------------------------------
    _XMP_coarray_contiguous_put(coindex-1,
				desc,   descDMA,
				offset, offsetDMA,
				bytes,  bytes);
    break;
  }
}


void _putVector_buffer(void *descPtr, char *baseAddr, int bytesRU,
                       int coindex, char *rhs, int bytes, SyncMode sync_mode)
{
  if (_XMPCO_get_isSafeBufferMode()) {
    if (sync_mode == syncATOMIC) {
      _XMPCO_debugPrint("SafeBufferMode does not support for atomic_define()");
    } else {
      _putVector_buffer_SAFE(descPtr, baseAddr, bytesRU,
			     coindex, rhs, bytes);
      return;
    }
  }

  _init_localBuf(descPtr, baseAddr, coindex);
  _push_localBuf(rhs, bytes, sync_mode);
  _flush_localBuf(sync_mode);
}


// SAFE mode without using localBuf
//
void _putVector_buffer_SAFE(void *descPtr, char *baseAddr, int bytesRU,
                            int coindex, char *rhs, int bytes)
{
  char *desc = _XMPCO_get_descForMemoryChunk(descPtr);
  size_t offset = _XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr);

  // MALLOC & MEMCPY
  char *buf = (char*)_XMP_alloc(sizeof(char) * bytesRU);

  _XMPCO_debugPrint("===MEMCPY, SAFE MODE, %d bytes\n"
                          "  from: addr=%p\n"
                          "  to  : addr=%p\n",
                          bytes,
                          rhs,
                          buf);
  (void)memcpy(buf, rhs, bytes);

  _XMPCO_debugPrint("===PUT_VECTOR RDMA to [%d], SAFE MODE, %d bytes\n"
                          "  source            : dynamically-allocated buffer, addr=%p\n"
                          "  destination (RDMA): \'%s\', offset=%zd\n",
                          coindex, bytes,
                          buf,
                          _XMPCO_get_nameOfCoarray(descPtr), offset);

  // ACTION
  _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // LHS (remote)
  _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // RHS (local)
  _XMP_coarray_rdma_image_set_1(coindex-1);
  _XMP_coarray_put(desc, buf, NULL);

  // NOT FREE for safe
  _XMPCO_debugPrint("===DO NOT FREE every local buffer in SAFE MODE\n"
                          "  addr=%p\n",
                          buf);
  //_XMP_free(buf);
}


/***************************************************\
    spread communication
\***************************************************/

void _spreadCoarray(void *descPtr, char *baseAddr, int coindex, char *rhs,
                    int bytes, int rank, int skip[], int count[],
                    int element, SyncMode sync_mode)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    _spreadVector_buffer(descPtr, baseAddr, bytes, coindex, rhs, element,
                         sync_mode);
    return;
  }

  if (bytes == skip[0]) {  // The first axis is contiguous
    // colapse the axis recursively
    _spreadCoarray(descPtr, baseAddr, coindex, rhs,
                   bytes * count[0], rank - 1, skip + 1, count + 1,
                   element, sync_mode);
    return;
  }

  // not contiguous any more
  char* src = rhs;

  if (_XMPCO_get_isMsgMode()) {
    char work[200];
    char* p;
    sprintf(work, "SPREAD %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPCO_debugPrint("=%s bytes\n", work);
  }

  src = _spreadVectorIter(descPtr, baseAddr, bytes, coindex, src,
                          rank, skip, count, element, sync_mode);
}

  
char *_spreadVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                        char *src, int rank, int skip[], int count[],
                        int element, SyncMode sync_mode)
{
  char* dst = baseAddr;
  int n = count[rank - 1];
  int gap = skip[rank - 1];

  if (rank == 1) {
    // DMA is not used.

    for (int i = 0; i < n; i++) {
      _spreadVector_buffer(descPtr, dst, bytes, coindex,
                           src, element, sync_mode);
      src += bytes;
      dst += gap;
    }
  }

  for (int i = 0; i < n; i++) {
    src = _spreadVectorIter(descPtr, baseAddr + i * gap, bytes,
                            coindex, src,
                            rank - 1, skip, count, element, sync_mode);
  }
  return src;
}


void _spreadVector_buffer(void *descPtr, char *baseAddr, int bytes, int coindex,
                          char *rhs, int element, SyncMode sync_mode)
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
    _XMPCO_debugPrint("===SPREAD %d-byte scalar to %d bytes, continued\n"
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
                   _localBuf_desc, _localBuf_offset, _localBuf_name,
                   sync_mode);

    src += bufSize;
    dst += bufSize;
  }

  _XMPCO_debugPrint("===SPREAD %d-byte scalar to %d bytes\n"
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
                 _localBuf_desc, _localBuf_offset, _localBuf_name,
                 sync_mode);
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


void _push_localBuf(char *src0, int bytes0, SyncMode sync_mode)
{
  char *src = src0;
  int bytes = bytes0;
  int copySize;

  if (_localBuf_used + bytes >= _localBuf_size) {
      _flush_localBuf(sync_mode);

      // for huge data
      while (bytes > _localBuf_size) {
        copySize = _localBuf_size;      
        _XMPCO_debugPrint("===MEMCPY %d of %d bytes to localBuf (cont\'d)\n"
                                "  from: addr=%p\n"
                                "  to  : localBuf\n",
                                copySize, bytes,
                                src);

        (void)memcpy(_localBuf_baseAddr, src, copySize);
        _localBuf_used = copySize;

        _flush_localBuf(sync_mode);

        src += copySize;
        bytes -= copySize;
      }
  }    

  if (bytes == 0)
    return;
  copySize = bytes;

  _XMPCO_debugPrint("===MEMCPY %d bytes to localBuf (final)\n"
                          "  from: addr=%p\n"
                          "  to  : localBuf + offset(%d bytes)\n",
                          copySize,
                          src,
                          _localBuf_used);

  (void)memcpy(_localBuf_baseAddr + _localBuf_used, src, copySize);
  _localBuf_used += copySize;
}


void _flush_localBuf(SyncMode sync_mode)
{
  int state;

  if (_localBuf_used > 0) {
    _putVector_DMA(_target_desc, _target_baseAddr, _localBuf_used, _target_coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name, sync_mode);
    _target_baseAddr += _localBuf_used;
    _localBuf_used = 0;
  }

  if (_XMPCO_get_isSyncPutMode()) {
      xmp_sync_memory(&state);
      _XMPCO_debugPrint("SYNC MEMORY caused by SYNCPUT MODE (stat=%d)\n", state);
  }
}


