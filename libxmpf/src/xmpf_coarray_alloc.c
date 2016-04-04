#include <string.h>
#include "xmpf_internal.h"

/*****************************************\
  macro definitions
\*****************************************/
#define _SortedChunkTableInitSize 256

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

#define forallMemoryChunkOrder(cp)  for(MemoryChunkOrder_t *_cp1_=((cp)=_mallocStack.head->next)->next; \
                                    _cp1_ != NULL;                     \
                                    (cp)=_cp1_, _cp1_=_cp1_->next)

#define forallMemoryChunkOrderRev(cp)  for(MemoryChunkOrder_t *_cp1_=((cp)=_mallocStack.tail->prev)->prev; \
                                       _cp1_ != NULL;                   \
                                       (cp)=_cp1_, _cp1_=_cp1_->prev)


#define forallResourceSet(rs)  for(ResourceSet_t *_rs1_=((rs)=_headResourceSet->next)->next; \
                                   _rs1_ != NULL;                       \
                                   (rs)=_rs1_, _rs1_=_rs1_->next)

#define forallResourceSetRev(rs)  for(ResourceSet_t *_rs1_=((rs)=_tailResourceSet->prev)->prev; \
                                      _rs1_ != NULL;                    \
                                      (rs)=_rs1_, _rs1_=_rs1_->prev)

#define forallMemoryChunk(chk,rs) for(MemoryChunk_t *_chk1_=((chk)=(rs)->headChunk->next)->next; \
                                      _chk1_ != NULL;                   \
                                      (chk)=_chk1_, _chk1_=_chk1_->next)

#define forallMemoryChunkRev(chk,rs) for(MemoryChunk_t *_chk1_=((chk)=(rs)->tailChunk->prev)->prev; \
                                         _chk1_ != NULL;                \
                                         (chk) = _chk1_, _chk1_=_chk1_->prev)

#define forallCoarrayInfo(ci,chk) for(CoarrayInfo_t *_ci1_ = ((ci)=(chk)->headCoarray->next)->next; \
                                      _ci1_ != NULL;                    \
                                      (ci) = _ci1_, _ci1_=_ci1_->next)

#define IsFirstCoarrayInfo(ci)  ((ci)->prev->prev == NULL)
#define IsLastCoarrayInfo(ci)   ((ci)->next->next == NULL)
#define IsOnlyCoarrayInfo(ci)   (IsFirstCoarrayInfo(ci) && IsLastCoarrayInfo(ci))

#define IsFirstMemoryChunk(chk)  ((chk)->prev->prev == NULL)
#define IsLastMemoryChunk(chk)   ((chk)->next->next == NULL)
#define IsEmptyMemoryChunk(chk)  ((chk)->headCoarray->next->next == NULL)

#define IsEmptyResourceSet(rs)   ((rs)->headChunk->next->next == NULL)


/*****************************************\
  typedef
\*****************************************/

// MEMORY MANAGEMENT STRUCTURE-I (for automatic deallocation)
typedef struct _resourceSet_t  ResourceSet_t;
typedef struct _memoryChunk_t  MemoryChunk_t;
typedef struct _coarrayInfo_t  CoarrayInfo_t;

// MEMORY MANAGEMENT STRUCTURE-II (for dynamic ALLOCATE/DEALLOCATE stmts.)
typedef struct _memoryChunkStack_t  MemoryChunkStack_t;
typedef struct _memoryChunkOrder_t  MemoryChunkOrder_t;

// MEMORY MANAGEMENT STRUCTURE-III (for binary search in memory chunks)
typedef struct _sortedChunkTable_t  SortedChunkTable_t;


/*****************************************\
  inernal structures
\*****************************************/

/** MEMORY MANAGEMENT STRUCTURE-I (for automatic deallocation)
 *  runtime resource corresponding to a procedure or to the entire program.
 *  A tag, cast of the address of a resource-set, is an interface to Fortran.
 */
struct _resourceSet_t {
  char            *name;        // procedure name (for debug message)
  MemoryChunk_t   *headChunk;
  MemoryChunk_t   *tailChunk;
};


/** structure for each malloc/free call
 *  Every memory chunk is linked both:
 *   - from a resource set until it is deallocated in the program, and
 *   - from _mallocHistory in order of malloc until it is actually be freed.
 */
struct _memoryChunk_t {
  MemoryChunk_t   *prev;
  MemoryChunk_t   *next;
  ResourceSet_t   *parent;
  BOOL             isGarbage;    // true if already encountered DEALLOCATE stmt
  char            *orgAddr;      // local address of the allocated memory
  unsigned         nbytes;       // allocated size of memory [bytes]
  void            *desc;         // address of the lower layer's descriptor 
  CoarrayInfo_t   *headCoarray;
  CoarrayInfo_t   *tailCoarray;
};

/** structure for each coarray variable
 *  One or more coarrays can be linked from a single memory chunk and be
 *  malloc'ed and be free'd together.
 */
struct _coarrayInfo_t {
  CoarrayInfo_t  *prev;
  CoarrayInfo_t  *next;
  MemoryChunk_t  *parent;
  char           *name;      // name of the variable (for debug message)
  char           *baseAddr;  // local address of the coarray (cray pointer)
  size_t          nbytes;    // size of the coarray [bytes]
  int             corank;    // number of codimensions
  int            *lcobound;  // array of lower cobounds [0..(corank-1)]
  int            *ucobound;  // array of upper cobounds [0..(corank-1)]
  int            *cosize;    // cosize[k] = max(ucobound[k]-lcobound[k]+1, 0)
};



/** MEMORY MANAGEMENT STRUCTURE-II (for dynamic ALLOCATE/DEALLOCATE stmts.)
 *  structure to manage the history of malloc/free
 */
struct _memoryChunkStack_t {
  MemoryChunkOrder_t  *head;
  MemoryChunkOrder_t  *tail;
};

struct _memoryChunkOrder_t {
  MemoryChunkOrder_t  *prev;
  MemoryChunkOrder_t  *next;
  MemoryChunk_t       *chunk;
};


/** MEMORY MANAGEMENT STRUCTURE-III (for quick search of memory chunks)
 *  table of memory chunks sorted in order of local address
 */
struct _sortedChunkTable_t {
  unsigned long   orgAddr;
  MemoryChunk_t  *chunk;
};


/*****************************************\
  static declarations
\*****************************************/

// access functions for resource set
static ResourceSet_t *_newResourceSet(char *name, int namelen);
static void _freeResourceSet(ResourceSet_t *rset);

// access functions for memory chunk
static MemoryChunk_t *_newMemoryChunk(void *desc, char *orgAddr, size_t nbytes);
static void _addMemoryChunkInResourceSet(ResourceSet_t *rset, MemoryChunk_t *chunk);
static void _unlinkMemoryChunk(MemoryChunk_t *chunk);
static void _unlinkMemoryChunkInResourceSet(MemoryChunk_t *chunk);
static void _freeMemoryChunk(MemoryChunk_t *chunk);
static char *_dispMemoryChunk(MemoryChunk_t *chunk);
static MemoryChunk_t *_getMemoryChunkFromLocalAddress(char *addr);

static MemoryChunk_t *pool_chunk = NULL;
static size_t pool_totalSize = 0;
static char *pool_currentAddr;

// access functions for coarray info
static CoarrayInfo_t *_newCoarrayInfo_empty(void);
static CoarrayInfo_t *_newCoarrayInfo(char *baseAddr, size_t nbytes);
static void _addCoarrayInfo(MemoryChunk_t *chunk, CoarrayInfo_t *cinfo2);
static void _unlinkCoarrayInfo(CoarrayInfo_t *cinfo2);
static void _freeCoarrayInfo(CoarrayInfo_t *cinfo);
static char *_dispCoarrayInfo(CoarrayInfo_t *cinfo);

static CoarrayInfo_t *_getShareOfStaticCoarray(size_t thisSize,
                                               size_t elementRU);
static CoarrayInfo_t *_allocLargeStaticCoarray(size_t thisSize,
                                               size_t elementRU);
static CoarrayInfo_t *_regmemStaticCoarray(void *baseAddr, size_t thisSize,
                                           size_t elementRU);

// CoarrayInfo for the static communication buffer
static CoarrayInfo_t *_cinfo_localBuf;  

// allocation and deallocation
static MemoryChunk_t *_mallocMemoryChunk(int count, size_t element);
static MemoryChunk_t *_mallocMemoryChunk_core(unsigned nbytes,
                                              size_t elementRU);
static MemoryChunk_t *_regmemMemoryChunk_core(void *baseAddr, unsigned nbytes,
                                              size_t elementRU);
static MemoryChunk_t *_constructMemoryChunk(void *baseAddr, unsigned nbytes,
                                            size_t elementRU);

// malloc/free history (STRUCTURE-II)
static MemoryChunkStack_t _mallocStack;

static void _initMallocHistory(void);
static void _addMemoryChunkToMallocHistory(MemoryChunk_t *chunk);
static void _garbageCollectMallocHistory(void);

// historical order
static MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk);
static void _unlinkMemoryChunkOrder(MemoryChunkOrder_t *chunkP);
static void _freeMemoryChunkOrder(MemoryChunkOrder_t *chunkP);

static char* _toOrdinalNumberString(int n);

// sorted chunk table (STRUCTURE-III)
static SortedChunkTable_t *_sortedChunkTable;
static size_t _sortedChunkTableMallocSize;
static size_t _sortedChunkTableSize;

static void _initSortedChunkTable(void);
static void _addMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk);
static void _delMemoryChunkInSortedChunkTable(MemoryChunk_t *chunk);
static MemoryChunk_t *_findMemoryChunkInSortedChunkTable(char *addr);
static int _searchSortedChunkTable(unsigned long addrKey, BOOL *found);

/***********************************************\
  hidden utility functions
\***********************************************/

int xmpf_coarray_allocated_bytes_()
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

  // subtract the size of the localBuf CoarrayInfo
  // because this is not allocated by the user
  size -= _cinfo_localBuf->nbytes;

  return size;
}

int xmpf_coarray_garbage_bytes_()
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
  ALLOCATE statement
  Type-1: alloc/free by the low-level library
  Steps:
    1. malloc calling the lower-level library
    2. make a MemoryChunk having a CoarrayInfo
\***********************************************/

void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                          int *count, int *element, void **tag)
{
  _XMPF_checkIfInTask("allocatable coarray allocation");
  ResourceSet_t *rset;

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MALLOC\n");

  // malloc
  MemoryChunk_t *chunk = _mallocMemoryChunk(*count, (size_t)(*element));
  _XMPF_coarrayDebugPrint("*** new MemoryChunk %s\n",
                          _dispMemoryChunk(chunk));

  if (*tag != NULL) {
    rset = (ResourceSet_t*)(*tag);
    _addMemoryChunkInResourceSet(rset, chunk);

    _XMPF_coarrayDebugPrint("*** MemoryChunk %s added to rset=%p\n",
                            _dispMemoryChunk(chunk), rset);
  }

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr,
                                         (*count) * (size_t)(*element));
  _addCoarrayInfo(chunk, cinfo);

  // output #1, #2
  *descPtr = (void*)cinfo;
  *crayPtr = cinfo->baseAddr;   // == chunk->orgAddr

  // SYNCALL_AUTO
  xmpf_sync_all_auto_();
}


size_t _roundUpElementSize(int count, size_t element, char *name, int namelen)
{
  size_t elementRU;

  // boundary check and recovery
  if (element % ONESIDED_BOUNDARY == 0) {
    elementRU = element;
  } else if (count == 1) {              // scalar or one-element array
    /* round up */
    elementRU = ROUND_UP_BOUNDARY(element);
    _XMPF_coarrayDebugPrint("round-up element size %d to %u (name=\"%*s\")\n",
                            element, elementRU, namelen, name);
  } else {
    /* restriction */
    _XMPF_coarrayFatal("boundary violation detected in coarray allocation\n"
                       "  element size %d (name=\"%*s\")\n",
                       element, namelen, name);
  }

  return elementRU;
}


MemoryChunk_t *_mallocMemoryChunk(int count, size_t element)
{
  MemoryChunk_t *chunk;
  static char *name = "(unk)";
  static const int namelen = 5;

  size_t elementRU = _roundUpElementSize(count, element, name, namelen);
  unsigned nbytes = (unsigned)count * elementRU;

  // make memory-chunk even if size nbyte=0
  chunk = _mallocMemoryChunk_core(nbytes, elementRU);

  return chunk;
}

MemoryChunk_t *_mallocMemoryChunk_core(unsigned nbytes, size_t elementRU)
{
  return _constructMemoryChunk(NULL, nbytes, elementRU);
}

MemoryChunk_t *_regmemMemoryChunk_core(void *baseAddr, unsigned nbytes,
                                       size_t elementRU)
{
  return _constructMemoryChunk(baseAddr, nbytes, elementRU);
}

MemoryChunk_t *_constructMemoryChunk(void *baseAddr, unsigned nbytes,
                                     size_t elementRU)
{
  void *desc;
  char *orgAddr;
  MemoryChunk_t *chunk;

  // _XMP_coarray_malloc() and set mallocInfo
  _XMP_coarray_malloc_info_1(nbytes, (size_t)1);   // set shape
  _XMP_coarray_malloc_image_info_1();              // set coshape
  if (baseAddr == NULL) {
    _XMP_coarray_malloc_do(&desc, &orgAddr);         // malloc
    chunk = _newMemoryChunk(desc, orgAddr, nbytes);
  } else {
    _XMP_coarray_regmem_do(&desc, baseAddr);         // register memory
    chunk = _newMemoryChunk(desc, baseAddr, nbytes);
  }

  _XMPF_coarrayDebugPrint("*** MemoryChunk %s was made.\n"
                          "  (%u bytes, elementRU=%u)\n",
                          _dispMemoryChunk(chunk), nbytes, elementRU);

  // stack to mallocHistory (STRUCTURE-II)
  _addMemoryChunkToMallocHistory(chunk);

  // add to _sortedChunkTable (STRUCTURE-III)
  _addMemoryChunkInSortedChunkTable(chunk);

  return chunk;
}


/***********************************************\
  DEALLOCATE statement
  Type-1: alloc/free by the low-level library
  Type-1a: to keep the reverse order of allocation,
     actual deallocation is delayed until garbage collection.
\***********************************************/

void xmpf_coarray_free_(void **descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  MemoryChunk_t *chunk = cinfo->parent;

  // SYNCALL_AUTO
  xmpf_sync_all_auto_();

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_FREE for MemoryChunk %s\n",
                          _dispMemoryChunk(chunk));

  // unlink and free CoarrayInfo keeping MemoryChunk
  _unlinkCoarrayInfo(cinfo);
  _freeCoarrayInfo(cinfo);

  if (IsEmptyMemoryChunk(chunk)) {
    // unlink this memory chunk as a garbage
    _unlinkMemoryChunk(chunk);
    // now chance to cellect and free garbages
    _garbageCollectMallocHistory();
  }
}


/*****************************************\
  handling memory pool
   for static coarrays
\*****************************************/

void xmpf_coarray_malloc_pool_(void)
{
  int one = 1;

  size_t localBufSize = XMPF_get_localBufSize();

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MALLOC_POOL\n"
                          "  static buffer (localBuf) allocated   : %u bytes\n",
                          "  user-defined static coarays allocated: %u bytes\n",
                          localBufSize, pool_totalSize);

  pool_totalSize += localBufSize;

  // init malloc/free history (STRUCTURE-II)
  _initMallocHistory();

  // init sorted chunk table (STRUCTURE-III)
  _initSortedChunkTable();

  // malloc the pool
  pool_chunk = _mallocMemoryChunk(1, pool_totalSize);
  pool_currentAddr = pool_chunk->orgAddr;

  // share communication buffer in the pool
  //-- instead of xmpf_coarray_alloc_static_
  _cinfo_localBuf = _getShareOfStaticCoarray(localBufSize,  // nbytes
                                             localBufSize); // elementRU
  _cinfo_localBuf->name = "(localBuf)";
  //-- call of xmpf_coarray_set_coshape_
  xmpf_coarray_set_coshape_((void **)(&_cinfo_localBuf), &one, &one);

  // init library internal
  _XMPF_coarrayInit_get();
  _XMPF_coarrayInit_put();
}

/*
 * End of string in Fortran may not have '\0'.
 * However, strndup() of gcc-4.8.4 assumes '\0' at end of string.
 * Therefore, we define a new function _xmp_strndup() instead of strndup().
 */
static char* _xmp_strndup(char *name, const int namelen)
{
  char *buf = (char *)malloc(namelen + 1);
  memcpy(buf, name, namelen);
  buf[namelen] = '\0';
  return buf;
}

/*
 * have a share of memory in the pool (if smaller)
 * or allocate individually (if larger)
 *    out: descPtr: pointer to descriptor CoarrayInfo_t
 *         crayPtr: cray pointer to the coarray object
 *    in:  count  : count of elements
 *         element: element size
 *         name   : name of the coarray (for debugging)
 *         namelen: character length of name (for debugging)
 */
void xmpf_coarray_alloc_static_(void **descPtr, char **crayPtr,
                                int *count, int *element,
                                char *name, int *namelen)
{
  size_t elementRU = _roundUpElementSize(*count, (size_t)(*element),
                                         name, *namelen);
  size_t nbytes = (size_t)(*count) * elementRU;

  CoarrayInfo_t *cinfo;

  _XMPF_coarrayDebugPrint("COARRAY_ALLOC_STATIC varname=\'%*s\'\n"
                          "  *count=%d, *element=%d, nbytes=%u, elementRU=%u\n",
                          *namelen, name, *count, *element, nbytes, elementRU);

  if (nbytes > XMPF_get_poolThreshold()) {
    _XMPF_coarrayDebugPrint("*** LARGER (%u bytes) than _XMPF_poolThreshold\n", nbytes);
    cinfo = _allocLargeStaticCoarray(nbytes, elementRU);
  } else {
    _XMPF_coarrayDebugPrint("*** SMALLER (%u bytes) than or eq to _XMPF_poolThreshold\n", nbytes);
    cinfo = _getShareOfStaticCoarray(nbytes, elementRU);
  }
  cinfo->name = _xmp_strndup(name, *namelen);

  *descPtr = (void*)cinfo;
  *crayPtr = cinfo->baseAddr;
}


/*
 * Similar to xmpf_coarray_alloc_static_() except that the coarray is 
 * allocated not by the runtime but by the Fortran system.
 *    out: descPtr : pointer to descriptor CoarrayInfo_t
 *    in:  baseAddr: local base address of a coarray
 *         count   : count of elements
 *         element : element size
 *         name    : name of the coarray (for debugging)
 *         namelen : character length of name (for debugging)
 */
void xmpf_coarray_regmem_static_(void **descPtr, void **baseAddr,
                                int *count, int *element,
                                 char *name, int *namelen)
{
  CoarrayInfo_t *cinfo;

  // boundary check
  if ((size_t)(*baseAddr) % ONESIDED_BOUNDARY != 0 ||
      *element % ONESIDED_BOUNDARY != 0) {
    /* restriction */
    _XMPF_coarrayFatal("boundary violation detected for coarray \'%s\'\n"
                       "  element size %d\n",
                       namelen, name, element);
  }

  size_t elementRU = (size_t)(*element);
  size_t nbytes = (size_t)(*count) * elementRU;

  _XMPF_coarrayDebugPrint("COARRAY_REGMEM_STATIC varname=\'%*s\'\n",
                          *namelen, name);

  cinfo = _regmemStaticCoarray(*baseAddr, nbytes, elementRU);
  cinfo->name = _xmp_strndup(name, *namelen);

  *descPtr = (void*)cinfo;
}


CoarrayInfo_t *_regmemStaticCoarray(void *baseAddr, size_t nbytes,
                                    size_t elementRU)
{
  _XMPF_checkIfInTask("allocation of static coarray");

  _XMPF_coarrayDebugPrint("*** REG-MEM (%u bytes)\n", nbytes);

  // get memory-chunk and set baseAddr
  MemoryChunk_t *chunk = _regmemMemoryChunk_core(baseAddr, nbytes, elementRU);

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytes);
  _addCoarrayInfo(chunk, cinfo);

  return cinfo;
}


CoarrayInfo_t *_allocLargeStaticCoarray(size_t nbytes, size_t elementRU)
{
  _XMPF_checkIfInTask("allocation of static coarray");

  // malloc memory-chunk
  MemoryChunk_t *chunk = _mallocMemoryChunk_core(nbytes, elementRU);
  _XMPF_coarrayDebugPrint("*** MemoryChunk %s malloc-ed\n",
                          _dispMemoryChunk(chunk));

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo(chunk->orgAddr, nbytes);
  _addCoarrayInfo(chunk, cinfo);

  return cinfo;
}


CoarrayInfo_t *_getShareOfStaticCoarray(size_t nbytes, size_t elementRU)
{
  _XMPF_checkIfInTask("share of static coarray");

  // allocate and set _coarrayInfo
  CoarrayInfo_t *cinfo = _newCoarrayInfo(pool_currentAddr, nbytes);
  _addCoarrayInfo(pool_chunk, cinfo);

  // check: lack of memory pool
  if (pool_currentAddr + nbytes > pool_chunk->orgAddr + pool_totalSize) {
    _XMPF_coarrayFatal("INTERNAL ERROR: "
                       "insufficient memory pool for static coarrays: "
                       "xmpf_coarray_share_pool_() in %s", __FILE__);
  }

  _XMPF_coarrayDebugPrint("*** memory share %u bytes from the pool <%p>\n",
                          nbytes, pool_currentAddr);

  pool_currentAddr += nbytes;

  return cinfo;
}


void xmpf_coarray_count_size_(int *count, int *element)
{
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);

  if (mallocSize > XMPF_get_poolThreshold()) {
    _XMPF_coarrayDebugPrint("XMPF_COARRAY_COUNT_SIZE: no count\n"
                            "  larger than pooling threshold: %u bytes\n",
                            mallocSize);
    return;
  }

  pool_totalSize += mallocSize;
  _XMPF_coarrayDebugPrint("XMPF_COARRAY_COUNT_SIZE: count up\n"
                          "  %u bytes, totally %u bytes\n",
                          mallocSize, pool_totalSize);
}


void xmpf_coarray_prolog_(void **tag, char *name, int *namelen)
{
  ResourceSet_t *rset = _newResourceSet(name, *namelen);

  _XMPF_coarrayDebugPrint("PROLOG CODE. name=\'%s\', rset=%p\n", rset->name, rset);

  *tag = (void*)rset;
}


void xmpf_coarray_epilog_(void **tag)
{
  if (*tag == NULL)
    return;

  ResourceSet_t *rset = (ResourceSet_t*)(*tag);

  _XMPF_coarrayDebugPrint("EPILOG CODE. name=\'%s\', rset=%p\n", rset->name, rset);

  _freeResourceSet(rset);     // with or without automatic SYNCALL

  *tag = NULL;
}


/*****************************************\
   entries
\*****************************************/

/** generate and return a descriptor for a coarray DUMMY ARGUMENT
 *   1. find the memory chunk that contains the coarray data object,
 *   2. generate coarrayInfo for the coarray dummy argument and link it 
 *      to the memory chunk, and
 *   3. return coarrayInfo as descPtr
 */
void xmpf_coarray_find_descptr_(void **descPtr, char *baseAddr,
                                void **tag, char *name, int *namelen)
{
  ResourceSet_t *rset = (ResourceSet_t*)(*tag);
  MemoryChunk_t *myChunk;

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_FIND_DESCPTR varname=\'%*s\'\n",
                          *namelen, name);

  if (rset == NULL)
    rset = _newResourceSet("(POOL)", strlen("(POOL)"));

  // generate a new descPtr for an allocatable dummy coarray
  CoarrayInfo_t *cinfo = _newCoarrayInfo_empty();

  // search my MemoryChunk only from baseAddr
  myChunk = _getMemoryChunkFromLocalAddress(baseAddr);

  if (myChunk == NULL) {
    _XMPF_coarrayDebugPrint("*** ILLEGAL: home MemoryChunk was not found. "
                            "baseAddr=%p\n", baseAddr);
    _XMPF_coarrayFatal("The actual argument corresponding to \'%*s\' "
                       "should be a coarray.\n", *namelen, name);
  }


  _XMPF_coarrayDebugPrint("*** found my home MemoryChunk %s\n",
                          _dispMemoryChunk(myChunk));

  _addCoarrayInfo(myChunk, cinfo);

  // return coarrayInfo as descPtr
  *descPtr = (void*)cinfo;
}


/*****************************************\
   STRUCTURE-III
   management of sorted memory-chunk table
\*****************************************/

void _initSortedChunkTable(void)
{
  _sortedChunkTableMallocSize = _SortedChunkTableInitSize;

  _sortedChunkTable = (SortedChunkTable_t*)
    malloc(sizeof(SortedChunkTable_t) * _sortedChunkTableMallocSize);

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

  // adjust malloc size
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


/*****************************************\
   STRUCTURE-II
   management of the history of malloc/free
\*****************************************/

void _initMallocHistory(void)
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


MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk)
{
  MemoryChunkOrder_t *chunkP =
    (MemoryChunkOrder_t*)calloc(1, sizeof(MemoryChunkOrder_t));
  chunkP->chunk = chunk;

  return chunkP;
}


/*  free deallocated coarry data objects as much as possible,
 *  keeping the reverse order of allocations.
 */
void _garbageCollectMallocHistory()
{
  MemoryChunkOrder_t *chunkP;

  _XMPF_coarrayDebugPrint("GARBAGE COLLECTION starts\n");

  forallMemoryChunkOrderRev(chunkP) {
    if (!chunkP->chunk->isGarbage)
      break;

    // unlink and free MemoryChunkOrder linkage
    _unlinkMemoryChunkOrder(chunkP);
    _freeMemoryChunkOrder(chunkP);
  }
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

  free(chunkP);
}



/*****************************************\
   management of dynamic attribute:
     current coshapes
\*****************************************/

/* translate m.
 * set the current lower and upper cobounds
 */
void xmpf_coarray_set_coshape_(void **descPtr, int *corank, ...)
{
  int i, n, count, n_images;
  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  va_list args;
  va_start(args, corank);

  cp->corank = n = *corank;
  cp->lcobound = (int*)malloc(sizeof(int) * n);
  cp->ucobound = (int*)malloc(sizeof(int) * n);
  cp->cosize = (int*)malloc(sizeof(int) * n);

  // axis other than the last
  for (count = 1, i = 0; i < n - 1; i++) {
    cp->lcobound[i] = *va_arg(args, int*);
    cp->ucobound[i] = *va_arg(args, int*);
    cp->cosize[i] = cp->ucobound[i] - cp->lcobound[i] + 1;
    if (cp->cosize[i] <= 0)
      _XMPF_coarrayFatal("upper cobound less than lower cobound");
    count *= cp->cosize[i];
  }

  // the last axis specified as lcobound:*
  n_images = num_images_();
  cp->lcobound[n-1] = *va_arg(args, int*);
  cp->cosize[n-1] = DIV_CEILING(n_images, count);
  cp->ucobound[n-1] = cp->lcobound[n-1] + cp->cosize[n-1] - 1;

  va_end(args);

  _XMPF_coarrayDebugPrint("*** set shape of CoarrayInfo %s, corank=%d\n",
                          _dispCoarrayInfo(cp), cp->corank);
}


/* translate n.
 * set the name of coarray object
 */
void xmpf_coarray_set_varname_(void **descPtr, char *name, int *namelen)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  cinfo->name = _xmp_strndup(name, *namelen);

  _XMPF_coarrayDebugPrint("*** set name of CoarrayInfo %s\n",
                          _dispCoarrayInfo(cinfo));
}



/*****************************************\
  access functions for ResourceSet_t
\*****************************************/

ResourceSet_t *_newResourceSet(char *name, int namelen)
{
  ResourceSet_t *rset =
    (ResourceSet_t*)malloc(sizeof(ResourceSet_t));

  rset->headChunk = _newMemoryChunk(NULL, NULL, 0);
  rset->tailChunk = _newMemoryChunk(NULL, NULL, 0);
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
    _XMPF_coarrayDebugPrint("*** avoid automatic syncall and garbage collection\n");
  }

  else {
    // SYNCALL_AUTO
    xmpf_sync_all_auto_();

    forallMemoryChunk (chunk, rset) {
      // unlink memory chunk as a garbage
      _unlinkMemoryChunk(chunk);
    }

    // now chance of garbabe collection
    _garbageCollectMallocHistory();
  }

  free(rset);
}


/*****************************************\
  access functions for MemoryChunk_t
\*****************************************/

MemoryChunk_t *_newMemoryChunk(void *desc, char *orgAddr, size_t nbytes)
{
  MemoryChunk_t *chunk =
    (MemoryChunk_t*)malloc(sizeof(MemoryChunk_t));

  chunk->prev = NULL;
  chunk->next = NULL;
  chunk->headCoarray = _newCoarrayInfo_empty();
  chunk->tailCoarray = _newCoarrayInfo_empty();
  chunk->headCoarray->next = chunk->tailCoarray;
  chunk->tailCoarray->prev = chunk->headCoarray;
  chunk->headCoarray->parent = chunk;
  chunk->tailCoarray->parent = chunk;
  chunk->isGarbage = FALSE;

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

  free(chunk);
}


char *_dispMemoryChunk(MemoryChunk_t *chunk)
{
  static char work[30+30*4];
  CoarrayInfo_t *cinfo;
  int count;

  (void)sprintf(work, "<%p %u bytes ", chunk, chunk->nbytes);

  count = 0;
  forallCoarrayInfo(cinfo, chunk) {
    if (++count == 4) {
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


/*   NEW VERSION for Ver.6-conversion
 */
MemoryChunk_t *_getMemoryChunkFromLocalAddress(char *addr)
{
  return _findMemoryChunkInSortedChunkTable(addr);
}

/***************************************
 *    OLD VERSION
 */
/**************************
MemoryChunk_t *_getMemoryChunkFromLocalAddress(char *addr)
{
  MemoryChunkOrder_t *chunkP;
  MemoryChunk_t *chunk;

  ** current implementation:
     look for my memory chunk into all MemoryChunkOrder
  /
  forallMemoryChunkOrder(chunkP) {
    chunk = chunkP->chunk;
    if (chunk->orgAddr <= addr && addr < chunk->orgAddr + chunk->nbytes) {
      // found the memory chunk
      return chunk;
    }
  }
  return NULL;
}
*************************/

/*****************************************\
  access functions for CoarrayInfo_t
\*****************************************/

static CoarrayInfo_t *_newCoarrayInfo_empty(void)
{
  CoarrayInfo_t *cinfo =
    (CoarrayInfo_t*)calloc(1, sizeof(CoarrayInfo_t));
  return cinfo;
}


static CoarrayInfo_t *_newCoarrayInfo(char *baseAddr, size_t nbytes)
{
  CoarrayInfo_t *cinfo =
    (CoarrayInfo_t*)calloc(1, sizeof(CoarrayInfo_t));
  cinfo->baseAddr = baseAddr;
  cinfo->nbytes = nbytes;

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
  free(cinfo->name);
  free(cinfo->lcobound);
  free(cinfo->ucobound);
  free(cinfo->cosize);
  free(cinfo);
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


/***********************************************\
   inquire function this_image(coarray)
   inquire function this_image(coarray, dim)
\***********************************************/

int xmpf_this_image_coarray_dim_(void **descPtr, int *corank, int *dim)
{
  int size, index, magic;
  int k = *dim - 1;

  if (k < 0 || *corank <= k)
    _XMP_fatal("Argument 'dim' of this_image is out of range");

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  magic = _XMPF_get_current_this_image() - 1;
  for (int i = 0; i < k; i++) {
    size = cinfo->cosize[i];
    magic /= size;
  }
  size = cinfo->cosize[k];
  index = magic % size;
  return index + cinfo->lcobound[k];
}

void xmpf_this_image_coarray_(void **descPtr, int *corank, int image[])
{
  int size, index, magic;

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  magic = _XMPF_get_current_this_image() - 1;
  for (int i = 0; i < *corank; i++) {
    size = cinfo->cosize[i];
    index = magic % size;
    image[i] = index + cinfo->lcobound[i];
    magic /= size;
  }
}


/***********************************************\
   inquire function lcobound/ucobound(coarray)
   inquire function lcobound/ucobound(coarray ,dim)
\***********************************************/

void lcobound_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of lcobound_");
}

void ucobound_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of ucobound_");
}


int xmpf_cobound_dim_(void **descPtr, int *dim, int *kind,
                      int *lu, int *corank)
{
  int index;
  int k = *dim - 1;

  if (*kind != 4)
    _XMP_fatal("Only kind=4 is allowed in lcobound/ucobound.");

  if (k < 0 || *corank <= k)
    _XMP_fatal("Argument 'dim' of lcobound/ucobound is out of range");

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  if (*lu <= 0)
    index = cinfo->lcobound[k];
  else
    index = cinfo->ucobound[k];

  return index;
}

void xmpf_cobound_nodim_subr_(void **descPtr, int *kind, 
                              int *lu, int *corank, int bounds[])
{
  if (*kind != 4)
    _XMP_fatal("Only kind=4 is allowed in lcobound/ucobound.");

  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  for (int i = 0; i < *corank; i++) {
    if (*lu <= 0)
      bounds[i] = cinfo->lcobound[i];
    else
      bounds[i] = cinfo->ucobound[i];
  }
}

/*  other interface for internal use
 */
int xmpf_lcobound_(void **descPtr, int *dim)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  return cinfo->lcobound[*dim - 1];
}

int xmpf_ucobound_(void **descPtr, int *dim)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  return cinfo->ucobound[*dim - 1];
}


/***********************************************\
   inquire function image_index(coarray, sub)
\***********************************************/

void image_index_(void)
{
  _XMPF_coarrayFatal("INTERNAL ERROR: illegal call of image_index_");
}


int xmpf_image_index_(void **descPtr, int coindexes[])
{
  int i, idx, lb, ub, factor, count, image;

  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  count = 0;
  factor = 1;
  for (i = 0; i < cp->corank; i++) {
    idx = coindexes[i];
    lb = cp->lcobound[i];
    ub = cp->ucobound[i];
    if (idx < lb) {
      _XMPF_coarrayDebugPrint
        ("The %s cosubscript of coarray \'%s\' is too small.\n"
         "  value=%d, range=[%d,%d]\n",
         _toOrdinalNumberString(i+1), cp->name, idx, lb, ub);
      return 0;
    }
    if (ub < idx && i < cp->corank - 1) {
      _XMPF_coarrayDebugPrint
        ("The %s cosubscript of coarray \'%s\' is too large.\n"
         "  value=%d, range=[%d,%d]\n",
         _toOrdinalNumberString(i+1), cp->name, idx, lb, ub);
      return 0;
    }
    count += (idx - lb) * factor;
    factor *= cp->cosize[i];
  }

  image = count + 1;
  if (image > num_images_())
    image = 0;

  return image;
}


/*  another interface for internal use
 */
int xmpf_coarray_get_image_index_(void **descPtr, int *corank, ...)
{
  int i, idx, lb, ub, factor, count;
  va_list(args);
  va_start(args, corank);

  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  if (cp->corank != *corank) {
    _XMPF_coarrayFatal("INTERNAL: found corank %d, which is "
                       "different from the declared corank %d",
                       *corank, cp->corank);
  }

  count = 0;
  factor = 1;
  for (i = 0; i < *corank; i++) {
    idx = *va_arg(args, int*);
    lb = cp->lcobound[i];
    ub = cp->ucobound[i];
    if (idx < lb || ub < idx) {
      _XMPF_coarrayFatal("%d-th cosubscript of \'%s\', %d, "
                         "is out of range %d to %d.\n",
                         i+1, cp->name, idx, lb, ub);
    }
    count += (idx - lb) * factor;
    factor *= cp->cosize[i];
  }

  va_end(args);

  return count + 1;
}



/***********************************************\
   inquire functions (internal)
\***********************************************/

char *_XMPF_get_coarrayName(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->name;
}

void *_XMPF_get_coarrayDesc(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->parent->desc;
}

size_t _XMPF_get_coarrayOffset(void *descPtr, char *baseAddr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  char* orgAddr = cinfo->parent->orgAddr;
  //int offset = ((size_t)baseAddr - (size_t)orgAddr);
  size_t offset = baseAddr - orgAddr;
  
  return offset;
}

void *_XMPF_get_localBufCoarrayDesc(char **baseAddr, size_t *offset,
                                    char **name)
{
  MemoryChunk_t *chunk = _cinfo_localBuf->parent;
  char *orgAddr = chunk->orgAddr;                    // origin address of the memory pool

  *baseAddr = _cinfo_localBuf->baseAddr;             // base address of the local buffer
  *offset = orgAddr - *baseAddr;                     // offset of the local buffer in the memory pool
  *name = _cinfo_localBuf->name;                     // name of the local buffer
  return chunk->desc;                                // descriptor of the memory pool
}


void *_XMPF_get_coarrayDescFromAddr(char *localAddr, char **orgAddr,
                                    size_t *offset, char **name)
{
  MemoryChunk_t* chunk = _getMemoryChunkFromLocalAddress(localAddr);
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


/***********************************************\
   local
\***********************************************/

char* _toOrdinalNumberString(int n)
{
  static char work[6];

  switch (n) {
  case 1:
    return "1st";
  case 2:
    return "2nd";
  case 3:
    return "3rd";
  default:
    break;
  }

  sprintf(work, "%dth", n);
  return work;
}
