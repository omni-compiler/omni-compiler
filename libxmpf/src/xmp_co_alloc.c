#include <string.h>
#include <stdarg.h>
#include "xmpf_internal_coarray.h"
#include "_xmp_co_alloc.h"

/*****************************************\
  static variables
\*****************************************/
// 
static MemoryChunk_t *pool_chunk = NULL;
static size_t pool_totalSize = 0;
static char *pool_currentAddr;

// sorted chunk table (STRUCTURE-III)
static SortedChunkTable_t *_sortedChunkTable;
static size_t _sortedChunkTableMallocSize;
static size_t _sortedChunkTableSize;


/*****************************************\
  static declarations
\*****************************************/
// access functions for resource set
static ResourceSet_t *_newResourceSet(char *name, int namelen);
static void _freeResourceSet(ResourceSet_t *rset);

// access functions for memory chunk
static MemoryChunk_t *_newMemoryChunk(void *desc, char *orgAddr, size_t nbytes);
static MemoryChunk_t *_newMemoryChunk_empty(void);
static void _addMemoryChunkInResourceSet(ResourceSet_t *rset, MemoryChunk_t *chunk);
static void _unlinkMemoryChunk(MemoryChunk_t *chunk);
static void _unlinkMemoryChunkInResourceSet(MemoryChunk_t *chunk);
static void _freeMemoryChunk(MemoryChunk_t *chunk);
static void _freeMemoryChunk_empty(MemoryChunk_t *chunk);
static char *_dispMemoryChunk(MemoryChunk_t *chunk);

// access functions for coarray info
static CoarrayInfo_t *_newCoarrayInfo_empty(void);
static void _freeCoarrayInfo_empty(CoarrayInfo_t *cinfo);
static CoarrayInfo_t *_newCoarrayInfo(char *baseAddr, size_t nbytes);
static void _setSimpleCoshapeToCoarrayInfo(CoarrayInfo_t *cinfo);
static void _addCoarrayInfo(MemoryChunk_t *chunk, CoarrayInfo_t *cinfo2);
static void _unlinkCoarrayInfo(CoarrayInfo_t *cinfo2);
static void _freeCoarrayInfo(CoarrayInfo_t *cinfo);
static char *_dispCoarrayInfo(CoarrayInfo_t *cinfo);

// allocation and deallocation
static size_t _roundUpElementSize(int count, size_t element, char *name, int namelen);
static MemoryChunk_t *_mallocMemoryChunk(int count, size_t element);
static MemoryChunk_t *_mallocMemoryChunk_core(unsigned nbytes);
static MemoryChunk_t *_regmemMemoryChunk_core(void *baseAddr, unsigned nbytes);
static MemoryChunk_t *_constructMemoryChunk(void *baseAddr, unsigned nbytes);

// historical order
static MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk);
static void _unlinkMemoryChunkOrder(MemoryChunkOrder_t *chunkP);
static void _freeMemoryChunkOrder(MemoryChunkOrder_t *chunkP);

static CoarrayInfo_t *_getShareOfStaticCoarray(size_t thisSize);
static CoarrayInfo_t *_allocLargeStaticCoarray(size_t nbytesRU);
static CoarrayInfo_t *_regmemStaticCoarray(void *baseAddr, size_t nbytesRU);

// STRUCTURE-II
//  management of the history of malloc/free
static void _initMallocHistory(void);
static void _addMemoryChunkToMallocHistory(MemoryChunk_t *chunk);

// STRUCTURE-III
//  management of sorted memory-chunk table
static void _initSortedChunkTable(void);
static void _addMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk);
static void _delMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk);
static MemoryChunk_t *_findMemoryChunkInSortedChunkTable(char *addr);
static int _searchSortedChunkTable(unsigned long addrKey, BOOL *found);


// utility
static char* _xmp_strndup(char *name, const int namelen);

struct {
  MemoryChunkOrder_t  *head;
  MemoryChunkOrder_t  *tail;
} _mallocStack;

// CoarrayInfo for control data area
static CoarrayInfo_t *_cinfo_ctrlData;
// CoarrayInfo for the static communication buffer
static CoarrayInfo_t *_cinfo_localBuf;  


/***********************************************\
  hidden inquire functions
\***********************************************/
static size_t _mallocSize = (size_t)0;

size_t xmp_coarray_malloc_bytes()
{
  return _mallocSize;
}

size_t xmp_coarray_allocated_bytes()
{
  MemoryChunkOrder_t *chunkp;
  MemoryChunk_t *chunk;
  size_t size, size1;

  // sum all sizes of MemoryChunks
  size = 0;
  forallMemoryChunkOrder(chunkp) {
    chunk = chunkp->chunk;
    size1 = size + chunk->nbytes;
    if (size1 < size)
      _XMPF_coarrayFatal("More than %llu-bytes of memory required for static coarrays\n",
                         ~(size_t)0 );
    size = size1;
  }

  // subtract the size of the localBuf and ctrlData CoarrayInfos
  // because this is not allocated by the user
  size -= _cinfo_ctrlData->size + _cinfo_localBuf->size;

  return size;
}


size_t xmp_coarray_garbage_bytes()
{
  MemoryChunkOrder_t *chunkp;
  MemoryChunk_t *chunk;
  size_t size;

  size = 0;
  forallMemoryChunkOrder(chunkp) {
    chunk = chunkp->chunk;
    if (chunk->isGarbage)
      size += chunk->nbytes;
  }

  return size;
}


/***********************************************\
  Allocation
\***********************************************/

CoarrayInfo_t *
_XMP_CO_malloc_coarray(char **addr, int count, size_t element,
                       ResourceSet_t *rset)
{
  size_t nbytes = count * element;

  _XMPF_coarrayDebugPrint("_XMP_CO_MALLOC_COARRAY\n");

  // malloc
  MemoryChunk_t *chunk = _mallocMemoryChunk(count, element);
  _XMPF_coarrayDebugPrint("*** new MemoryChunk %s\n",
                          _dispMemoryChunk(chunk));

  if (rset != NULL) {
    _addMemoryChunkInResourceSet(rset, chunk);

    _XMPF_coarrayDebugPrint("*** MemoryChunk %s added to rset=%p\n",
                            _dispMemoryChunk(chunk), rset);
  }

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytes);

  _addCoarrayInfo(chunk, cinfo);

  *addr = cinfo->baseAddr;   // == chunk->orgAddr
  return cinfo;
}


CoarrayInfo_t *
_XMP_CO_regmem_coarray(void *var, int count, size_t element,
                       ResourceSet_t *rset)
{
  size_t nbytes = count * element;

  _XMPF_coarrayDebugPrint("_XMP_CO_REGMEM_COARRAY\n");

  // regmem
  MemoryChunk_t *chunk = _regmemMemoryChunk_core(var, nbytes);
  _XMPF_coarrayDebugPrint("*** new MemoryChunk for RegMem variable %s\n",
                          _dispMemoryChunk(chunk));

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytes);

  _addCoarrayInfo(chunk, cinfo);
  return cinfo;
}



/**
 * have a share of memory in the pool (if not larger than threshold)
 * or allocate individually (if larger than threshold)
 *    out: return : pointer to descriptor CoarrayInfo_t
 *         addr   : address of the coarray object to be allocated
 *    in:  count  : count of elements
 *         element: element size
 *         namelen: character length of name (for debugging)
 *         name   : name of the coarray (for debugging)
 */
CoarrayInfo_t *
_XMP_CO_malloc_staticCoarray(char **addr, int count, size_t element,
                             int namelen, char *name)
{
  size_t nbytes = count * element;
  CoarrayInfo_t *cinfo;

  _XMPF_coarrayDebugPrint("_XMP_CO_ALLOC_STATIC_COARRAY varname=\'%.*s\'\n"
                          "  count=%d, element=%d, nbytes=%u\n",
                          namelen, name, count, element, nbytes);

  if (nbytes > XMPF_get_poolThreshold()) {
    _XMPF_coarrayDebugPrint("*** LARGER (%u bytes) than _XMPF_poolThreshold\n", nbytes);
    cinfo = _allocLargeStaticCoarray(nbytes);
  } else {
    size_t nbytesRU = ROUND_UP_MALLOC(nbytes);
    _XMPF_coarrayDebugPrint("*** Not LARGER (%u bytes) than _XMPF_poolThreshold\n", nbytesRU);
    cinfo = _getShareOfStaticCoarray(nbytesRU);
  }
  cinfo->name = _xmp_strndup(name, namelen);

  *addr = cinfo->baseAddr;
  return cinfo;
}


/**
 * Similar to _alloc_static_coarray() except that the coarray is 
 * allocated not by the runtime but by the Fortran system.
 *    out: return  : pointer to descriptor CoarrayInfo_t
 *    in:  var     : pointer to the coarray
 *         count   : count of elements
 *         element : element size
 *         name    : name of the coarray (for debugging)
 *         namelen : character length of name (for debugging)
 */
CoarrayInfo_t *
_XMP_CO_regmem_staticCoarray(void *var, int count, size_t element,
                             int namelen, char *name)
{
  CoarrayInfo_t *cinfo;

  _XMPF_coarrayDebugPrint("_XMP_CO_REGMEM_STATIC_COARRAY varname=\'%.*s\'\n"
                          "  count=%d, element=%ud\n",
                          namelen, name, count, element);

  // boundary check
  if ((size_t)var % MALLOC_UNIT != 0) {  // check base address
    /* restriction */
    _XMPF_coarrayFatal("boundary violation detected for coarray \'%.*s\'\n"
                       "  var=%p\n",
                       namelen, name, var);
  }

  size_t nbytes = count * element;
  //size_t nbytesRU = ROUND_UP_MALLOC(nbytes);

  _XMPF_coarrayDebugPrint("COARRAY_REGMEM_STATIC_ varname=\'%.*s\'\n",
                          namelen, name);

  cinfo = _regmemStaticCoarray(var, nbytes);
  //cinfo = _regmemStaticCoarray(var, nbytesRU);
  cinfo->name = _xmp_strndup(name, namelen);

  return cinfo;
}


CoarrayInfo_t *_regmemStaticCoarray(void *baseAddr, size_t nbytes)
{
  _XMPF_checkIfInTask("memory registration of static coarray");

  _XMPF_coarrayDebugPrint("*** _regmemStaticCoarray (%u bytes)\n", nbytes);

  // get memory-chunk and set baseAddr
  MemoryChunk_t *chunk = _regmemMemoryChunk_core(baseAddr, nbytes);

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytes);
  _addCoarrayInfo(chunk, cinfo);

  return cinfo;
}


CoarrayInfo_t *_allocLargeStaticCoarray(size_t nbytesRU)
{
  _XMPF_checkIfInTask("allocation of static coarray");

  // malloc memory-chunk
  MemoryChunk_t *chunk = _mallocMemoryChunk_core(nbytesRU);
  _XMPF_coarrayDebugPrint("*** MemoryChunk %s malloc-ed\n",
                          _dispMemoryChunk(chunk));

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytesRU);
  _addCoarrayInfo(chunk, cinfo);

  return cinfo;
}


CoarrayInfo_t *_getShareOfStaticCoarray(size_t nbytesRU)
{
  _XMPF_checkIfInTask("_getShareOfStaticCoarray() called in a task");

  // allocate and set _coarrayInfo
  CoarrayInfo_t *cinfo = _newCoarrayInfo(pool_currentAddr, nbytesRU);
  _addCoarrayInfo(pool_chunk, cinfo);
  
  // check: lack of memory pool
  if (pool_currentAddr + nbytesRU > pool_chunk->orgAddr + pool_totalSize) {
    _XMPF_coarrayFatal("INTERNAL ERROR: "
                       "insufficient memory pool for static coarray: "
                       "_getShareOfStaticCoarray() in %s", __FILE__);
  }

  _XMPF_coarrayDebugPrint("*** memory share %u bytes from the pool\n", nbytesRU);

  pool_currentAddr += nbytesRU;

  return cinfo;
}


/***********************************************\
  Deallocation/Deregistration
  - to keep the reverse order of allocation,
    freeing memory is delayed until garbage collection.
\***********************************************/

void _XMP_CO_free_coarray(CoarrayInfo_t *cinfo)
{
  MemoryChunk_t *chunk = cinfo->parent;

  // SYNCALL_AUTO
  xmpf_sync_all_auto_();

  _XMPF_coarrayDebugPrint("_XMP_CO_FREE_COARRAY for MemoryChunk %s\n",
                          _dispMemoryChunk(chunk));

  // unlink and free CoarrayInfo keeping MemoryChunk
  _unlinkCoarrayInfo(cinfo);
  _freeCoarrayInfo(cinfo);

  if (IsEmptyMemoryChunk(chunk)) {
    // unlink this memory chunk as a garbage
    _unlinkMemoryChunk(chunk);
    // now chance to cellect and free garbages
    _XMP_CO_garbageCollectMallocHistory();
  }
}


void _XMP_CO_deregmem_coarray(CoarrayInfo_t *cinfo)
{
  MemoryChunk_t *chunk = cinfo->parent;

  // SYNCALL_AUTO
  xmpf_sync_all_auto_();

  _XMPF_coarrayDebugPrint("_XMP_CO_DEREGMEM_COARRAY for MemoryChunk %s\n",
                          _dispMemoryChunk(chunk));

  // unlink and free CoarrayInfo keeping MemoryChunk
  _unlinkCoarrayInfo(cinfo);
  _freeCoarrayInfo(cinfo);

  if (IsEmptyMemoryChunk(chunk)) {
    // unlink this memory chunk
    _unlinkMemoryChunk(chunk);
  }
}


/***********************************************\
   parts
\***********************************************/

MemoryChunk_t *_mallocMemoryChunk(int count, size_t element)
{
  MemoryChunk_t *chunk;
  size_t nbytes = count * element;
  size_t nbytesRU = ROUND_UP_MALLOC(nbytes);

  // make memory-chunk even if size nbyte=0
  chunk = _mallocMemoryChunk_core(nbytesRU);

  return chunk;
}

MemoryChunk_t *_mallocMemoryChunk_core(unsigned nbytesRU)
{
  return _constructMemoryChunk(NULL, nbytesRU);
}

MemoryChunk_t *_regmemMemoryChunk_core(void *baseAddr, unsigned nbytesRU)
{
  return _constructMemoryChunk(baseAddr, nbytesRU);
}

MemoryChunk_t *_constructMemoryChunk(void *baseAddr, unsigned nbytes)
{
  void *desc;
  char *orgAddr;
  MemoryChunk_t *chunk;

  // _XMP_coarray_malloc() and set mallocInfo
  _XMP_coarray_malloc_info_1(nbytes, (size_t)1);   // set shape
  _XMP_coarray_malloc_image_info_1();              // set coshape
  if (baseAddr == NULL) {
    _XMP_coarray_malloc(&desc, &orgAddr);         // malloc
    chunk = _newMemoryChunk(desc, orgAddr, nbytes);
  } else {
    _XMP_coarray_regmem(&desc, baseAddr);         // register memory
    chunk = _newMemoryChunk(desc, baseAddr, nbytes);
  }

  _XMPF_coarrayDebugPrint("*** MemoryChunk %s was made. (%u bytes)\n",
                          _dispMemoryChunk(chunk), nbytes);

  // stack to mallocHistory (STRUCTURE-II)
  _addMemoryChunkToMallocHistory(chunk);

  // add to _sortedChunkTable (STRUCTURE-III)
  _addMemoryChunkInSortedChunkTable(chunk);

  return chunk;
}


/*****************************************\
  Initialization/Finalization
  Handling memory pool
\*****************************************/
void _XMP_CO_malloc_pool()
{
  size_t ctrlDataSize = sizeof(int) * 8;
  size_t localBufSize = XMPF_get_localBufSize();

  _XMPF_coarrayDebugPrint("_XMP_CO_MALLOC_POOL_ contains:\n"
                          "  system-defined local buffer :%10u bytes\n"
                          "  system-defined control data :%10u bytes\n"
                          "  user-defined coarays        :%10u bytes\n",
                          localBufSize, ctrlDataSize, pool_totalSize);

  pool_totalSize += localBufSize + ctrlDataSize;

  // init malloc/free history (STRUCTURE-II)
  _initMallocHistory();

  // init sorted chunk table (STRUCTURE-III)
  _initSortedChunkTable();

  // malloc the pool
  pool_chunk = _mallocMemoryChunk(1, pool_totalSize);
  pool_currentAddr = pool_chunk->orgAddr;

  // share communication buffer in the pool
  _cinfo_localBuf = _getShareOfStaticCoarray(localBufSize);
  _cinfo_localBuf->name = "(localBuf)";
  _setSimpleCoshapeToCoarrayInfo(_cinfo_localBuf);

  // share control data area in the pool
  _cinfo_ctrlData = _getShareOfStaticCoarray(ctrlDataSize);
  _cinfo_ctrlData->name = "(ctrlData)";
  _setSimpleCoshapeToCoarrayInfo(_cinfo_ctrlData);

  // init library internal
  _XMPF_coarrayInit_get();
  _XMPF_coarrayInit_getsub();
  _XMPF_coarrayInit_put();
}

static void _setSimpleCoshapeToCoarrayInfo(CoarrayInfo_t *cinfo)
{
  int size;

  cinfo->corank = 1;
  cinfo->lcobound = (int*)MALLOC(sizeof(int));
  cinfo->ucobound = (int*)MALLOC(sizeof(int));
  cinfo->cosize = (int*)MALLOC(sizeof(int));

  size = _XMPF_num_images_current();
  cinfo->lcobound[0] = 1;
  cinfo->ucobound[0] = size;
  cinfo->cosize[0] = size;
}



void _XMP_CO_count_size(int count, size_t element)
{
  size_t thisSize = count * element;
  size_t mallocSize = ROUND_UP_MALLOC(thisSize);

  if (mallocSize > XMPF_get_poolThreshold()) {
    _XMPF_coarrayDebugPrint("XMP_CO_COUNT_SIZE_: no count because of the large size\n"
                            "  pooling threshold :%10u bytes\n"
                            "  data size         :%10u bytes\n",
                            XMPF_get_poolThreshold(), mallocSize);
    return;
  }

  pool_totalSize += mallocSize;
  _XMPF_coarrayDebugPrint("XMP_CO_COUNT_SIZE_: count up\n"
                          "  %u bytes, totally %u bytes\n",
                          mallocSize, pool_totalSize);
}



MPI_Comm _XMP_CO_communicatorFromCoarrayInfo(CoarrayInfo_t *cinfo)
{
  if (cinfo == NULL)
    return MPI_COMM_NULL;

  _XMP_nodes_t *nodes = cinfo->nodes;

  if (nodes == NULL)
    return MPI_COMM_NULL;

  return *(MPI_Comm*)(nodes->comm);
}


void _XMP_CO_prolog(void **tag, int namelen, char *name)
{
  ResourceSet_t *rset = _newResourceSet(name, namelen);

  _XMPF_coarrayDebugPrint("XMP_CO_PROLOG (name=\'%s\', rset=%p)\n", rset->name, rset);

  *tag = (void*)rset;
}


void _XMP_CO_epilog(void **tag)
{
  if (*tag == NULL)
    return;

  ResourceSet_t *rset = (ResourceSet_t*)(*tag);

  _XMPF_coarrayDebugPrint("_XMP_CO_EPILOG_ (name=\'%s\', rset=%p)\n", rset->name, rset);

  _freeResourceSet(rset);     // with or without automatic SYNCALL

  *tag = NULL;
}



/*****************************************\
  Find descriptor from the local address
\*****************************************/

/** generate and return a descriptor for a coarray DUMMY ARGUMENT
 *   1. find the memory chunk that contains the coarray data object,
 *   2. generate coarrayInfo for the coarray dummy argument and link it 
 *      to the memory chunk, and
 *   3. return coarrayInfo as descPtr
 */
void* _XMP_CO_find_descptr(char *addr, int namelen, char *name)
{
  MemoryChunk_t *myChunk;

  _XMPF_coarrayDebugPrint("_XMF_CO_FIND_DESCPTR_ "
                          "(varname=\'%.*s\')\n",
                          namelen, name);

  // generate a new descPtr for an allocatable dummy coarray
  CoarrayInfo_t *cinfo = _newCoarrayInfo_empty();

  // search my MemoryChunk only from addr
  myChunk = _findMemoryChunkInSortedChunkTable(addr);

  if (myChunk != NULL) {
    _XMPF_coarrayDebugPrint("*** found my home MemoryChunk %s\n",
                            _dispMemoryChunk(myChunk));
    // return coarrayInfo as descPtr
    _addCoarrayInfo(myChunk, cinfo);
    return (void*)cinfo;
  }

  _XMPF_coarrayDebugPrint("*** found no MemoryChunk of mine\n");
  return NULL;
}



/*****************************************\
  set attributes of CoarrayInfo
\*****************************************/

void _XMP_CO_set_corank(CoarrayInfo_t *cp, int corank)
{
  cp->corank = corank;
  cp->lcobound = (int*)MALLOC(sizeof(int) * corank);
  cp->ucobound = (int*)MALLOC(sizeof(int) * corank);
  cp->cosize = (int*)MALLOC(sizeof(int) * corank);

  _XMPF_coarrayDebugPrint("*** set corank of CoarrayInfo %s, corank=%d\n",
                          _dispCoarrayInfo(cp), cp->corank);
}


void _XMP_CO_set_codim_withBOUNDS(CoarrayInfo_t *cp, int dim, int lb, int ub)
{
  int i, count, n_images, size;

  if (dim < cp->corank - 1) {        // not last dimension
    cp->lcobound[dim] = lb;
    cp->ucobound[dim] = ub;
    size = ub - lb + 1;
    cp->cosize[dim] = size;
    if (cp->cosize[dim] <= 0)
      _XMPF_coarrayFatal("upper cobound less than lower cobound");
  }
  else if (dim == cp->corank - 1) {   // last dimension
    cp->lcobound[dim] = lb;
    n_images = _XMPF_num_images_current();
    for (i = 0, count = 1; i < cp->corank - 1; i++)
      count *= cp->cosize[i];
    size = DIV_CEILING(n_images, count);
    cp->cosize[dim] = size;
    cp->ucobound[dim] = lb + size - 1;
  }
  else {                               // illegal
    _XMPF_coarrayFatal("spedified dim (%d) >= pre-specified corank (%d)\n",
                       dim, cp->corank);
  }
}

void _XMP_CO_set_varname(CoarrayInfo_t *cp, int namelen, char *name)
{
  cp->name = _xmp_strndup(name, namelen);
  _XMPF_coarrayDebugPrint("*** set name of CoarrayInfo %s\n",
                          _dispCoarrayInfo(cp));
}


CoarrayInfo_t* _XMP_CO_set_nodes(CoarrayInfo_t *cinfo,
                                 _XMP_nodes_t *nodes)
{
  if (cinfo != NULL) {
    cinfo->nodes = nodes;
    return cinfo;
  }

  CoarrayInfo_t* cinfo1 = _newCoarrayInfo_empty();
  cinfo1->nodes = nodes;
  return cinfo1;
}


/***********************************************\
   inquire functions (open interface)
\***********************************************/

char *_XMP_CO_get_coarrayName(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->name;
}

char *_XMP_CO_get_coarrayBaseAddr(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->baseAddr;
}

size_t _XMP_CO_get_coarraySize(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->size;
}

size_t _XMP_CO_get_coarrayOffset(void *descPtr, char *addr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  size_t offset = addr - cinfo->baseAddr;
  return offset;
}


void *_XMP_CO_get_coarrayChunkDesc(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->parent->desc;
}

char *_XMP_CO_get_coarrayChunkOrgAddr(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->parent->orgAddr;
}

size_t _XMP_CO_get_coarrayChunkSize(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->parent->nbytes;
}

size_t _XMP_CO_get_coarrayChunkOffset(void *descPtr, char *addr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  char* orgAddr = cinfo->parent->orgAddr;
  size_t offset = addr - orgAddr;
  return offset;
}


void *_XMP_CO_get_ctrlDataCoarrayDesc(char **baseAddr, size_t *offset,
                                    char **name)
{
  MemoryChunk_t *chunk = _cinfo_ctrlData->parent;
  char *orgAddr = chunk->orgAddr;                    // origin address of the memory pool

  *baseAddr = _cinfo_ctrlData->baseAddr;             // base address of the control data
  *offset = orgAddr - *baseAddr;                     // offset of the control data in the memory pool
  *name = _cinfo_ctrlData->name;                     // name of the control data
  return chunk->desc;                                // descriptor of the memory pool
}

void *_XMP_CO_get_localBufCoarrayDesc(char **baseAddr, size_t *offset,
                                    char **name)
{
  MemoryChunk_t *chunk = _cinfo_localBuf->parent;
  char *orgAddr = chunk->orgAddr;                    // origin address of the memory pool

  *baseAddr = _cinfo_localBuf->baseAddr;             // base address of the local buffer
  *offset = orgAddr - *baseAddr;                     // offset of the local buffer in the memory pool
  *name = _cinfo_localBuf->name;                     // name of the local buffer
  return chunk->desc;                                // descriptor of the memory pool
}

BOOL _XMP_CO_isAddrInCoarrayChunk(char *localAddr, void *descPtr)
{
  char *orgAddr = _XMP_CO_get_coarrayChunkOrgAddr(descPtr);
  size_t size = _XMP_CO_get_coarrayChunkSize(descPtr);
  size_t offset = localAddr - orgAddr;
  BOOL result = (offset < size);
  return result;
}

void *_XMP_CO_get_coarrayDescFromAddr(char *localAddr, char **orgAddr,
                                    size_t *offset, char **name)
{
  MemoryChunk_t* chunk = _findMemoryChunkInSortedChunkTable(localAddr);
  if (chunk == NULL) {
    *orgAddr = NULL;
    *offset = 0;
    *name = "(not found)";
    return NULL;
  }

  *orgAddr = chunk->orgAddr;
  *offset = localAddr - *orgAddr;
  *name = chunk->headCoarray->next->name;
  return chunk->desc;
}

/*  return MPI_COMM_NULL if communicatior is not specified for descPtr
 */
MPI_Comm _XMP_CO_get_communicatorFromDescPtr(void *descPtr)
{
  return _XMP_CO_communicatorFromCoarrayInfo((CoarrayInfo_t *)descPtr);
}


/*****************************************\
  access functions for ResourceSet_t
\*****************************************/

ResourceSet_t *_newResourceSet(char *name, int namelen)
{
  ResourceSet_t *rset =
    (ResourceSet_t*)MALLOC(sizeof(ResourceSet_t));

  rset->headChunk = _newMemoryChunk_empty();
  rset->tailChunk = _newMemoryChunk_empty();
  rset->headChunk->next = rset->tailChunk;
  rset->tailChunk->prev = rset->headChunk;
  rset->headChunk->parent = rset;
  rset->tailChunk->parent = rset;
  rset->name = _xmp_strndup(name, namelen);
  return rset;
}

void _freeResourceSet(ResourceSet_t *rset)
{
  MemoryChunk_t *chunk;

  if (IsEmptyResourceSet(rset)) {
    // avoid automatic syncall (ID=465)
    _XMPF_coarrayDebugPrint("*** omitted automatic syncall and garbage collection\n");
  }

  else {
    // SYNCALL_AUTO
    xmpf_sync_all_auto_();

    forallMemoryChunk (chunk, rset) {
      // unlink memory chunk as a garbage
      _unlinkMemoryChunk(chunk);
    }

    // now chance of garbabe collection
    _XMP_CO_garbageCollectMallocHistory();
  }

  FREE_string(rset->name);
  FREE_ResourceSet_t(rset);
}


/*****************************************\
  access functions for MemoryChunk_t
\*****************************************/

MemoryChunk_t *_newMemoryChunk_empty(void)
{
  MemoryChunk_t *chunk =
    (MemoryChunk_t*)MALLOC(sizeof(MemoryChunk_t));

  chunk->prev = NULL;
  chunk->next = NULL;
  chunk->headCoarray = _newCoarrayInfo_empty();
  chunk->tailCoarray = _newCoarrayInfo_empty();
  chunk->headCoarray->next = chunk->tailCoarray;
  chunk->tailCoarray->prev = chunk->headCoarray;
  chunk->headCoarray->parent = chunk;
  chunk->tailCoarray->parent = chunk;
  chunk->isGarbage = FALSE;

  return chunk;
}


MemoryChunk_t *_newMemoryChunk(void *desc, char *orgAddr, size_t nbytes)
{
  MemoryChunk_t *chunk = _newMemoryChunk_empty();

  chunk->desc = desc;
  chunk->orgAddr = orgAddr;
  chunk->nbytes = nbytes;

  return chunk;
}


void _addMemoryChunkInResourceSet(ResourceSet_t *rset, MemoryChunk_t *chunk2)
{
  MemoryChunk_t *chunk3 = rset->tailChunk;
  MemoryChunk_t *chunk1 = chunk3->prev;

  chunk1->next = chunk2;
  chunk3->prev = chunk2;

  chunk2->prev = chunk1;
  chunk2->next = chunk3;
  chunk2->parent = rset;
}


void _unlinkMemoryChunk(MemoryChunk_t *chunk)
{
  // STRUCTURE-II
  chunk->isGarbage = TRUE;

  // STRUCTURE-III
  _delMemoryChunkInSortedChunkTable(chunk);

  // STRUCTURE-I
  _unlinkMemoryChunkInResourceSet(chunk);
}


void _unlinkMemoryChunkInResourceSet(MemoryChunk_t *chunk2)
{
  MemoryChunk_t *chunk1 = chunk2->prev;
  MemoryChunk_t *chunk3 = chunk2->next;

  _XMPF_coarrayDebugPrint("*** MemoryChunk %s unlinking from parent %p\n",
                          _dispMemoryChunk(chunk2), chunk2->parent);

  if (chunk1 != NULL) {
    chunk1->next = chunk3;
    chunk2->prev = NULL;
  }

  if (chunk3 != NULL) {
    chunk3->prev = chunk1;
    chunk2->next = NULL;
  }

  chunk2->parent = NULL;
}


void _freeMemoryChunk_empty(MemoryChunk_t *chunk)
{
  _freeCoarrayInfo_empty(chunk->headCoarray);
  _freeCoarrayInfo_empty(chunk->tailCoarray);
  FREE_MemoryChunk_t(chunk);
}


void _freeMemoryChunk(MemoryChunk_t *chunk)
{
  CoarrayInfo_t *cinfo;

  forallCoarrayInfo (cinfo, chunk) {
    _unlinkCoarrayInfo(cinfo);
    _freeCoarrayInfo(cinfo);
  }

  _XMPF_coarrayDebugPrint("*** MemoryChunk %s freeing\n",
                          _dispMemoryChunk(chunk));

  // free the last memory chunk object
  _XMP_coarray_lastly_deallocate();

  _freeMemoryChunk_empty(chunk);
}


char *_dispMemoryChunk(MemoryChunk_t *chunk)
{
  static char work[30+30*4];
  CoarrayInfo_t *cinfo;
  int count;

  (void)sprintf(work, "<%p %u bytes ", chunk, (unsigned)chunk->nbytes);

  count = 0;
  forallCoarrayInfo(cinfo, chunk) {
    if (++count == 6) {
      strcat(work, "...");
      break;
    } 
    if (cinfo->name) {
      strcat(work, "\'");
      strncat(work, cinfo->name, 20);
      strcat(work, "\'");
    } else {
      strcat(work, "(null)");
    }
  }

  strcat(work, ">");

  return work;
}



/*****************************************\
  access functions for CoarrayInfo_t
\*****************************************/

static CoarrayInfo_t *_newCoarrayInfo_empty(void)
{
  CoarrayInfo_t *cinfo =
    (CoarrayInfo_t*)CALLOC(1, sizeof(CoarrayInfo_t));
  return cinfo;
}

static void _freeCoarrayInfo_empty(CoarrayInfo_t *cinfo)
{
  FREE_CoarrayInfo_t(cinfo);
}


static CoarrayInfo_t *_newCoarrayInfo(char *baseAddr, size_t size)
{
  CoarrayInfo_t *cinfo = _newCoarrayInfo_empty();
  cinfo->baseAddr = baseAddr;
  cinfo->size = size;

  _XMPF_coarrayDebugPrint("*** new CoarrayInfo %s\n",
                          _dispCoarrayInfo(cinfo));
  return cinfo;
}


void _addCoarrayInfo(MemoryChunk_t *parent, CoarrayInfo_t *cinfo2)
{
  CoarrayInfo_t *cinfo3 = parent->tailCoarray;
  CoarrayInfo_t *cinfo1 = cinfo3->prev;

  cinfo1->next = cinfo2;
  cinfo3->prev = cinfo2;
  cinfo2->prev = cinfo1;
  cinfo2->next = cinfo3;
  cinfo2->parent = parent;

  _XMPF_coarrayDebugPrint("*** CoarrayInfo %s added to MemoryChunk %s\n",
                          _dispCoarrayInfo(cinfo2), _dispMemoryChunk(parent));
}

void _unlinkCoarrayInfo(CoarrayInfo_t *cinfo2)
{
  CoarrayInfo_t *cinfo1 = cinfo2->prev;
  CoarrayInfo_t *cinfo3 = cinfo2->next;

  cinfo1->next = cinfo3;
  cinfo3->prev = cinfo1;
  cinfo2->prev = NULL;
  cinfo2->next = NULL;
  cinfo2->parent = NULL;

  _XMPF_coarrayDebugPrint("*** CoarrayInfo %s unlinked from MemoryChunk %s\n",
                          _dispCoarrayInfo(cinfo2),
                          _dispMemoryChunk(cinfo1->parent));
}

void _freeCoarrayInfo(CoarrayInfo_t *cinfo)
{
  FREE_string(cinfo->name);
  int n = cinfo->corank;
  FREE_int_n(cinfo->lcobound, n);
  FREE_int_n(cinfo->ucobound, n);
  FREE_int_n(cinfo->cosize, n);
  FREE_CoarrayInfo_t(cinfo);
}


char *_dispCoarrayInfo(CoarrayInfo_t *cinfo)
{
  static char work[300];

  char *name = cinfo->name;
  if (name) {
    if (strlen(name) > 280)
      (void)sprintf(work, "<%p (too-long-name)>", cinfo);
    else
      (void)sprintf(work, "<%p \'%s\'>", cinfo, name);
  } else {
    (void)sprintf(work, "<%p (noname)>", cinfo);
  }
  return work;
}


/*****************************************\
   STRUCTURE-II
   management of the history of malloc/free
\*****************************************/

void _initMallocHistory()
{
  _mallocStack.head = _newMemoryChunkOrder(NULL);
  _mallocStack.tail = _newMemoryChunkOrder(NULL);
  _mallocStack.head->next = _mallocStack.tail;
  _mallocStack.tail->prev = _mallocStack.head;
}


void _addMemoryChunkToMallocHistory(MemoryChunk_t *chunk)
{
  MemoryChunkOrder_t *chunkP2 = _newMemoryChunkOrder(chunk);
  MemoryChunkOrder_t *chunkP3 = _mallocStack.tail;
  MemoryChunkOrder_t *chunkP1 = chunkP3->prev;

  chunkP1->next = chunkP2;
  chunkP3->prev = chunkP2;
  chunkP2->prev = chunkP1;
  chunkP2->next = chunkP3;
}


/*  free deallocated coarry data objects as much as possible,
 *  keeping the reverse order of allocations.
 */
void _XMP_CO_garbageCollectMallocHistory()
{
  MemoryChunkOrder_t *chunkP;

  _XMPF_coarrayDebugPrint("[[[GARBAGE COLLECTION]]] starts\n");

  forallMemoryChunkOrderRev(chunkP) {
    if (!chunkP->chunk->isGarbage)
      break;

    // unlink and free MemoryChunkOrder linkage
    _unlinkMemoryChunkOrder(chunkP);
    _freeMemoryChunkOrder(chunkP);
  }

  _XMPF_coarrayDebugPrint("[[[GARBAGE COLLECTION]]] ends\n");
}



MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk)
{
  MemoryChunkOrder_t *chunkP =
    (MemoryChunkOrder_t*)CALLOC(1, sizeof(MemoryChunkOrder_t));
  chunkP->chunk = chunk;

  return chunkP;
}

void _unlinkMemoryChunkOrder(MemoryChunkOrder_t *chunkP2)
{
  MemoryChunkOrder_t *chunkP1 = chunkP2->prev;
  MemoryChunkOrder_t *chunkP3 = chunkP2->next;

  chunkP1->next = chunkP3;
  chunkP3->prev = chunkP1;
  chunkP2->next = chunkP2->prev = NULL;
}

void _freeMemoryChunkOrder(MemoryChunkOrder_t *chunkP)
{
  // including freeing coarray data object
  _freeMemoryChunk(chunkP->chunk);

  FREE_MemoryChunkOrder_t(chunkP);
}


/*****************************************\
   STRUCTURE-III
   management of sorted memory-chunk table
\*****************************************/

void _initSortedChunkTable(void)
{
  _sortedChunkTableMallocSize = _SortedChunkTableInitSize;

  _sortedChunkTable = (SortedChunkTable_t*)
    MALLOC(sizeof(SortedChunkTable_t) * _sortedChunkTableMallocSize);

  _sortedChunkTableSize = 2;
  _sortedChunkTable[0].orgAddr = 0L;
  _sortedChunkTable[0].chunk = NULL;
  _sortedChunkTable[1].orgAddr = ~0L;
  _sortedChunkTable[1].chunk = NULL;
}


void _addMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk)
{
  int idx;
  BOOL found;
  unsigned long addrKey;

  // condition check
  if (chunk == NULL || chunk->nbytes == 0)
    return;      // empty memory-chunk structure

  // adjust table size
  if (_sortedChunkTableSize == _sortedChunkTableMallocSize) {
    _sortedChunkTableMallocSize *= 2;
    _sortedChunkTable = (SortedChunkTable_t*)
      realloc(_sortedChunkTable,
              sizeof(SortedChunkTable_t) * _sortedChunkTableMallocSize);
  }

  // search
  addrKey = (unsigned long)(chunk->orgAddr);
  idx = _searchSortedChunkTable(addrKey, &found);
  if (found) {
    _XMPF_coarrayFatal("_addMemoryChunkInSortedChunkTable() failed\n"
                       "because Memory-chunk (including coarray \'%s\') was "
                       "already booked in _sortedChunkTable\n",
                       chunk->headCoarray->next->name);
  }

  // shift
  for (int i = _sortedChunkTableSize - 1; i > idx; i--) {
    _sortedChunkTable[i+1].orgAddr = _sortedChunkTable[i].orgAddr;
    _sortedChunkTable[i+1].chunk = _sortedChunkTable[i].chunk;
  }

  // add
  _sortedChunkTable[idx+1].orgAddr = addrKey;
  _sortedChunkTable[idx+1].chunk = chunk;

  ++_sortedChunkTableSize;
}


void _delMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk)
{
  int idx;
  BOOL found;
  unsigned long addrKey;

  // condition check
  if (chunk == NULL || chunk->nbytes == 0)
    return;      // empty memory-chunk structure

  // adjust malloc size
  if (_sortedChunkTableSize*4 <= _sortedChunkTableMallocSize &&
      _sortedChunkTableMallocSize > _SortedChunkTableInitSize) {
    _sortedChunkTableMallocSize /= 2;
    _sortedChunkTable = (SortedChunkTable_t*)
      realloc(_sortedChunkTable,
              sizeof(SortedChunkTable_t) * _sortedChunkTableMallocSize);
  }

  // search
  addrKey = (unsigned long)(chunk->orgAddr);
  idx = _searchSortedChunkTable(addrKey, &found);
  if (!found) {
    _XMPF_coarrayFatal("_delMemoryChunkInSortedChunkTable() failed\n"
                       "because Memory-chunk (including coarray \'%s\') is "
                       "not booked in _sortedChunkTable\n",
                       chunk->headCoarray->next->name);
  }

  --_sortedChunkTableSize;

  // overwrite and shift
  for (int i = idx; i < _sortedChunkTableSize; i++) {
    _sortedChunkTable[i].orgAddr = _sortedChunkTable[i+1].orgAddr;
    _sortedChunkTable[i].chunk = _sortedChunkTable[i+1].chunk;
  }
}

/*  NULL if not found
 */
MemoryChunk_t *_findMemoryChunkInSortedChunkTable(char *addr)
{
  int idx;
  BOOL found;
  unsigned long addrKey = (unsigned long)addr;

  idx = _searchSortedChunkTable(addrKey, &found);
  if (!found)
    return NULL;

  return _sortedChunkTable[idx].chunk;
}

/* lowest-level routine using binary search
 */
int _searchSortedChunkTable(unsigned long addrKey, BOOL *found)
{
  int idx_low = 0;
  int idx_over = _sortedChunkTableSize - 1;
  int idx;
  unsigned long addrKey_idx;
  MemoryChunk_t *chunk;


  // binary search
  // initial condition:
  //   _sortedChunkTable[idx_low].orgAddr==0L
  //   _sortedChunkTable[idx_over].orgAddr==~0L
  for (idx = (idx_low + idx_over) / 2;
       idx > idx_low;
       idx = (idx_low + idx_over) / 2) {
    addrKey_idx = _sortedChunkTable[idx].orgAddr;
    if (addrKey == addrKey_idx) {
      // found the same address
      *found = TRUE;
      return idx;
    }
    else if (addrKey < addrKey_idx)
      // It is less than idx.
      idx_over = idx;
    else
      // It is greater than or equal to idx.
      idx_low = idx;
  }

  // check if the address is included in the memory-chunk
  chunk = _sortedChunkTable[idx].chunk;
  if (chunk == NULL)
    *found = FALSE;
  else 
    *found = (addrKey < (unsigned long)(chunk->orgAddr + chunk->nbytes));
  return idx;
}


/***********************************************\
   utils
\***********************************************/

/*
 * End of string in Fortran may not have '\0'.
 * However, strndup() of gcc-4.8.4 assumes '\0' at end of string.
 * Therefore, we define a new function _xmp_strndup() instead of strndup().
 */
char* _xmp_strndup(char *name, const int namelen)
{
  char *buf = (char *)MALLOC(namelen + 1);
  memcpy(buf, name, namelen);
  buf[namelen] = '\0';
  return buf;
}

/*  The size of the scalar variable will be rounded up to the size
 *  that the communication library can handle.
 */
size_t _roundUpElementSize(int count, size_t element, char *name, int namelen)
{
  size_t elementRU;

  // boundary check and recovery
  if (element % COMM_UNIT == 0) {
    elementRU = element;
  } else if (count == 1) {              // scalar or one-element array
    /* round up */
    elementRU = ROUND_UP_COMM(element);
    _XMPF_coarrayDebugPrint("round-up size of scalar variable "
                            "%d to %u (name=\"%.*s\")\n",
                            element, elementRU, namelen, name);
  } else {
    /* restriction */
    _XMPF_coarrayFatal("boundary violation detected in coarray allocation\n"
                       "  element size %d (name=\"%.*s\")\n",
                       element, namelen, name);
  }

  return elementRU;
}


/***********************************************\
  malloc/free wrapper
\***********************************************/
void *MALLOC(size_t size)
{
  _mallocSize += size;
  return malloc(size);
}

void *CALLOC(size_t nmemb, size_t size)
{
  _mallocSize += nmemb * size;
  return calloc(nmemb, size);
}

void _FREE(void *ptr)
{
  free(ptr);
}

void FREE_ResourceSet_t(ResourceSet_t *rset)
{
  _mallocSize -= sizeof(ResourceSet_t);
  _FREE(rset);
}

void FREE_MemoryChunkOrder_t(MemoryChunkOrder_t *chunkp)
{
  _mallocSize -= sizeof(MemoryChunkOrder_t);
  _FREE(chunkp);
}

void FREE_MemoryChunk_t(MemoryChunk_t *chunk)
{
  _mallocSize -= sizeof(MemoryChunk_t);
  _FREE(chunk);
}

void FREE_CoarrayInfo_t(CoarrayInfo_t *cinfo)
{
  _mallocSize -= sizeof(CoarrayInfo_t);
  _FREE(cinfo);
}


void FREE_string(char *name)
{
  _mallocSize -= strlen(name) + 1;
  _FREE(name);
}

void FREE_int_n(int *intp, int n)
{
  _mallocSize -= sizeof(*intp) * n;
  _FREE(intp);
}


