/*
 *   COARRAY GET
 *
 */

#include <assert.h>
#include "xmpco_internal.h"
#include "_xmpco_putget.h"

// communication schemes
#define SCHEME_DirectGet       20   // RDMA expected
#define SCHEME_BufferGet       21   // to get visible to FJ-RDMA
#define SCHEME_ExtraDirectGet  22   // DirectGet with extra data
#define SCHEME_ExtraBufferGet  23   // BufferGet with extra data

static int _select_scheme_get_scalar(int element, BOOL avail_DMA);
static int _select_scheme_get_array(BOOL avail_DMA);

static void _getCoarray(void *descPtr, char *baseAddr, int coindex, char *res,
                        int bytes, int rank, int skip[], int count[],
                        void *descDMA, size_t offsetDMA, char *nameDMA);

static char *_getVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                            char *dst, int loops, int skip[], int count[],
                            void *descDMA, size_t offsetDMA, char *nameDMA);


/***************************************************\
    initialization
\***************************************************/

/* copy of static infos */
void *_localBuf_desc;           // descriptor of the memory pool
size_t _localBuf_offset;        // offset of the local buffer in the memory pool
char *_localBuf_baseAddr;       // local base address of the local buffer
char *_localBuf_name;           // name of the local buffer


void _XMPCO_coarrayInit_get()
{
  _localBuf_desc = _XMPCO_get_infoOfLocalBuf(&_localBuf_baseAddr,
                                              &_localBuf_offset,
                                              &_localBuf_name);
}



/***************************************************\
    entry
\***************************************************/

void XMPCO_GET_scalarExpr(CoarrayInfo_t *descPtr, char *baseAddr, 
                          int element, int coindex, char *result)
{
  int coindex0 = _XMPCO_get_initial_image_withDescPtr(coindex, descPtr);

  /*--------------------------------------*\
   * Confirm that *baseAddr is located in *
   * the memory chunk corresponding to    *
   * descptr.                             *
  \*--------------------------------------*/
  BOOL pass = _XMPCO_isAddrInMemoryChunk(baseAddr, descPtr);
  if (!pass) {
    _XMPCO_fatal("INTERNAL: illegal baseAddr of mold coarray (in xmpf_coarray_get_scalar_)\n"
                       "  possibly because of unexpected copy-in interface\n"
                       "  coarray_name=\"%s\", chunk_size=%ud; offset_in_chunk=%ud",
                       _XMPCO_get_nameOfCoarray(descPtr),
                       (unsigned)_XMPCO_get_sizeOfMemoryChunk(descPtr),
                       (unsigned)_XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr));
  }

  /*--------------------------------------*\
   * Check whether the local address      *
   * result is already registered for DMA *
   * and, if so, get the descriptor, etc. *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  BOOL avail_DMA;

  descDMA = _XMPCO_get_isEagerCommMode() ? NULL :
      _XMPCO_get_desc_fromLocalAddr(result, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_scheme_get_scalar(element, avail_DMA);

  switch (scheme) {
  case SCHEME_DirectGet:
    _XMPCO_debugPrint("SCHEME_DirectGet/scalar selected\n");

    assert(avail_DMA);
    _XMPCO_getVector_DMA(descPtr, baseAddr, element, coindex0,
                   descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferGet:
    _XMPCO_debugPrint("select SCHEME_BufferGet/scalar\n");

    _XMPCO_getVector_buffer(descPtr, baseAddr, element, coindex0,
                      result, element);
    break;

  case SCHEME_ExtraBufferGet:
    {
      size_t elementRU = ROUND_UP_COMM(element);

      _XMPCO_debugPrint("select SCHEME_ExtraBufferGet/scalar. elementRU=%ud\n",
                              elementRU);

      _XMPCO_getVector_buffer(descPtr, baseAddr, elementRU, coindex0,
                        result, element);
    }
    break;

  default:
    _XMPCO_fatal("undefined scheme number in " __FILE__);
  }
}


void XMPCO_GET_arrayExpr(CoarrayInfo_t *descPtr, char *baseAddr,
                         int element, int coindex, char *result,
                         int rank, int skip[], int count[])
{
  int coindex0 = _XMPCO_get_initial_image_withDescPtr(coindex, descPtr);

  if (element % COMM_UNIT != 0) {
    _XMPCO_fatal("violation of boundary in reference of a coindexed object\n"
              "  xmpf_coarray_get_array_, " __FILE__);
    return;
  }

  /*--------------------------------------*\
   * Confirm that baseAddr is located in  *
   * the memory chunk corresponding to    *
   * descptr.                             *
  \*--------------------------------------*/
  BOOL pass = _XMPCO_isAddrInMemoryChunk(baseAddr, descPtr);
  if (!pass) {
    _XMPCO_fatal("INTERNAL : illegal baseAddr of mold coarray (in xmpf_coarray_get_array_)\n"
                       "  possibly because of unexpected copy-in interface\n"
                       "  coarray_name=\"%s\", chunk_size=%ud; offset_in_chunk=%ud",
                       _XMPCO_get_nameOfCoarray(descPtr),
                       (unsigned)_XMPCO_get_sizeOfMemoryChunk(descPtr),
                       (unsigned)_XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr));
  }

  /*--------------------------------------*\
   * Check whether the local address      *
   * result is already registered for DMA *
   * and, if so, get the descriptor, etc. *
  \*--------------------------------------*/
  void *descDMA;
  size_t offsetDMA;
  char *orgAddrDMA;
  char *nameDMA;
  BOOL avail_DMA;

  descDMA = _XMPCO_get_isEagerCommMode() ? NULL :
      _XMPCO_get_desc_fromLocalAddr(result, &orgAddrDMA, &offsetDMA, &nameDMA);
  avail_DMA = descDMA ? TRUE : FALSE;

  /*--------------------------------------*\
   * select scheme                        *
  \*--------------------------------------*/
  int scheme = _select_scheme_get_array(avail_DMA);

  /*--------------------------------------*\
   * action                               *
  \*--------------------------------------*/
  switch (scheme) {
  case SCHEME_DirectGet:
    _XMPCO_debugPrint("select SCHEME_DirectGet/array\n");
    _getCoarray(descPtr, baseAddr, coindex0, result,
                element, rank, skip, count,
                descDMA, offsetDMA, nameDMA);
    break;

  case SCHEME_BufferGet:
    _XMPCO_debugPrint("select SCHEME_BufferGet/array\n");
    _getCoarray(descPtr, baseAddr, coindex0, result,
                element, rank, skip, count,
                NULL, 0, "(localBuf)");
    break;

  default:
    _XMPCO_fatal("undefined scheme number in " __FILE__);
  }
}



/***************************************************\
    select schemes
\***************************************************/

/* REMARKING CONDITIONS:
 *  - The result variable may be invisible to FJ-RDMA.
 *  - The length of get communication must be divisible by
 *    COMM_UNIT. Else, SCHEME_Extra... should be selected.
 *  - SCHEME_ExtraDirectGet should not be used because the extra
 *    copy to local destination might overwrite neighboring data.
 *  - Array element of coarray is divisible by COMM_UNIT
 *    due to a restriction.
 */

int _select_scheme_get_scalar(int element, BOOL avail_DMA)
{
  if (element % COMM_UNIT > 0)
    return SCHEME_ExtraBufferGet;

  if (avail_DMA)
    return SCHEME_DirectGet;

  return SCHEME_BufferGet;
}

int _select_scheme_get_array(BOOL avail_DMA)
{
  if (avail_DMA)
    return SCHEME_DirectGet;

  return SCHEME_BufferGet;
}


/***************************************************\
    sub
\***************************************************/

void _getCoarray(void *descPtr, char *baseAddr, int coindex, char *result,
                 int bytes, int rank, int skip[], int count[],
                 void *descDMA, size_t offsetDMA, char *nameDMA)
{
  if (rank == 0) {  // fully contiguous after perfect collapsing
    if (descDMA != NULL)   // DMA available
      _XMPCO_getVector_DMA(descPtr, baseAddr, bytes, coindex,
                     descDMA, offsetDMA, nameDMA);
    else
      _XMPCO_getVector_buffer(descPtr, baseAddr, bytes, coindex,
                        result, bytes);
    return;
  }

  if (bytes == skip[0]) {  // The first axis is contiguous
    // colapse the axis recursively
    _getCoarray(descPtr, baseAddr, coindex, result,
                bytes * count[0], rank - 1, skip + 1, count + 1,
                descDMA, offsetDMA, nameDMA);
    return;
  }

  // not contiguous any more
  char* dst = result;

  if (_XMPCO_get_isMsgMode()) {
    char work[200];
    char* p;
    sprintf(work, "GET %d", bytes);
    p = work + strlen(work);
    for (int i = 0; i < rank; i++) {
      sprintf(p, " (%d-byte stride) * %d", skip[i], count[i]);
      p += strlen(p);
    }
    _XMPCO_debugPrint("%s bytes ===\n", work);
  }

  dst = _getVectorIter(descPtr, baseAddr, bytes, coindex, dst,
                       rank, skip, count,
                       descDMA, offsetDMA, nameDMA);
}

  
char *_getVectorIter(void *descPtr, char *baseAddr, int bytes, int coindex,
                     char *dst, int loops, int skip[], int count[],
                     void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* src = baseAddr;
  int n = count[loops - 1];
  int gap = skip[loops - 1];

  if (loops == 1) {
    if (descDMA != NULL) {  // DMA available
      for (int i = 0; i < n; i++) {
        _XMPCO_getVector_DMA(descPtr, src, bytes, coindex,
                             descDMA, offsetDMA, nameDMA);
        dst += bytes;
        src += gap;
      }
    } else {    // recursive getVector with static buffer
      for (int i = 0; i < n; i++) {
        _XMPCO_getVector_buffer(descPtr, src, bytes, coindex,
                                dst, bytes);
        dst += bytes;
        src += gap;
      }
    }
    return dst;
  }

  for (int i = 0; i < n; i++) {
    dst = _getVectorIter(descPtr, baseAddr + i * gap, bytes,
                         coindex, dst,
                         loops - 1, skip, count,
                         descDMA, offsetDMA, nameDMA);
  }
  return dst;
}



/***************************************************\
    common
\***************************************************/

void _XMPCO_getVector_DMA(void *descPtr, char *baseAddr, int bytes, int coindex,
                          void *descDMA, size_t offsetDMA, char *nameDMA)
{
  char* desc = _XMPCO_get_descForMemoryChunk(descPtr);
  size_t offset = _XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr);

  _XMPCO_debugPrint("=e=GET_VECTOR RDMA-DMA %d bytes\n"
		    "  from: remote[%d] \'%s\' + %zd\n"
		    "  to  : %s + %zd\n",
		    bytes,
		    coindex, _XMPCO_get_nameOfCoarray(descPtr), offset,
		    nameDMA, offsetDMA);

  // ACTION
  _XMP_coarray_contiguous_get(coindex-1,
                            descDMA,   desc,
                            offsetDMA, offset,
                            bytes,     bytes);
}


void _XMPCO_getVector_buffer(void *descPtr, char *baseAddr, int bytesRU, int coindex,
                             char *result, int bytes)
{
  size_t rest1, rest2, bufSize;
  char *src, *dst;

  src = baseAddr;
  dst = result;
  bufSize = _XMPCO_get_localBufSize();

  // communication for every buffer size
  for (rest1 = bytesRU, rest2 = bytes;
       rest1 > bufSize;
       rest1 -= bufSize, rest2 -=bufSize) {
    _XMPCO_getVector_DMA(descPtr, src, bufSize, coindex,
                   _localBuf_desc, _localBuf_offset, _localBuf_name);

    _XMPCO_debugPrint("=e=MEMCPY %d bytes, cont\'d\n"
                            "  from: \'%s\'\n"
                            "  to  : addr=%p\n",
                            bufSize,
                            _localBuf_name,
                            dst);
    (void)memcpy(dst, _localBuf_baseAddr, bufSize);

    src += bufSize;
    dst += bufSize;
  }

  _XMPCO_getVector_DMA(descPtr, src, rest1, coindex,
                 _localBuf_desc, _localBuf_offset, _localBuf_name);

  _XMPCO_debugPrint("=e=MEMCPY %d bytes, final\n"
                          "  from: \'%s\'\n"
                          "  to  : addr=%p\n",
                          rest2,
                          _localBuf_name,
                          dst);
  (void)memcpy(dst, _localBuf_baseAddr, rest2);
}



#if 0   //obsolete
void _getVector(void *descPtr, char *baseAddr, int bytes, int coindex,
                char *dst)
{
  char* desc = _XMPCO_get_descForMemoryChunk(descPtr);
  size_t offset = _XMPCO_get_offsetInMemoryChunk(descPtr, baseAddr);

  if ((size_t)bytes <= _XMPCO_get_localBufSize()) {
    _XMPCO_debugPrint("to [%d] GET_VECTOR, RDMA-DMA-memcpy, %d bytes\n"
                            "  source      (RDMA): \'%s\', offset=%zd\n"
                            "  destination (DMA) : static buffer in the pool, offset=%zd\n",
                            coindex, bytes,
                            _XMPCO_get_nameOfCoarray(descPtr), offset,
                            _localBuf_offset);

    // ACTION
    _XMP_coarray_contiguous_get(coindex-1,
                              _localBuf_desc,   desc,
                              _localBuf_offset, offset,
                              bytes,            bytes);
    (void)memcpy(dst, _localBuf_baseAddr, bytes);

  } else {
    _XMPCO_debugPrint("to [%d] GET_VECTOR, regmem&RDMA, %d bytes\n"
                            "  source      (RDMA): \'%s\', offset=%zd\n"
                            "  destination       : dynamically-allocated buffer, addr=%p\n",
                            coindex, bytes,
                            _XMPCO_get_nameOfCoarray(descPtr), offset,
                            dst);

    // ACTION
    _XMP_coarray_rdma_coarray_set_1(offset, bytes, 1);    // coindexed-object
    _XMP_coarray_rdma_array_set_1(0, bytes, 1, 1, 1);    // result
    _XMP_coarray_rdma_image_set_1(coindex-1);
    _XMP_coarray_rdma_do(COARRAY_GET_CODE, desc, dst, NULL);
  }
}


#endif
