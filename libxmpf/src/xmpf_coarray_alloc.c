#include "xmpf_internal.h"

#define BOOL   int
#define TRUE   1
#define FALSE  0

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


/*****************************************\
  typedef and static declaration
\*****************************************/

typedef struct _resourceSet_t  ResourceSet_t;
typedef struct _memoryChunk_t  MemoryChunk_t;
typedef struct _coarrayInfo_t  CoarrayInfo_t;

typedef struct _memoryChunkStack_t  MemoryChunkStack_t;
typedef struct _memoryChunkOrder_t  MemoryChunkOrder_t;

// access functions for resource set
static ResourceSet_t *_newResourceSet(void);
static void _freeResourceSet(ResourceSet_t *rset);

// access functions for memory chunk
static MemoryChunk_t *_newMemoryChunk(void);
static void _linkMemoryChunk(ResourceSet_t *rset, MemoryChunk_t *chunk);
static void _unlinkAndFreeMemoryChunk(MemoryChunk_t *chunk);
static void _unlinkMemoryChunk(MemoryChunk_t *chunk);
static void _freeMemoryChunk(MemoryChunk_t *chunk);

static MemoryChunk_t *pool_chunk = NULL;
static size_t pool_totalSize = 0;
static char *pool_currentAddr;

// access functions for coarray info
static CoarrayInfo_t *_newCoarrayInfo(void);
static void _addCoarrayInfo(MemoryChunk_t *chunk, CoarrayInfo_t *cinfo2);
static void _unlinkCoarrayInfo(CoarrayInfo_t *cinfo2);
static void _freeCoarrayInfo(CoarrayInfo_t *cinfo);

static CoarrayInfo_t *_getShareOfCoarray(int count, size_t element);


// allocation and deallocation
static MemoryChunk_t *_mallocMemoryChunk(int count, size_t element);
//static void _flushResourceSetInReverseOrder(void);
//static void _freeMemoryChunkReverseOrder(void);

// malloc/free history
static void _initMallocHistory(void);
static void _addMallocHistory(MemoryChunk_t *chunk);
static void _flushMallocHistory(void);

// historical order
static MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk);
static void _unlinkMemoryChunkOrder(MemoryChunkOrder_t *chunkP);
static void _freeMemoryChunkOrder(MemoryChunkOrder_t *chunkP);



/*****************************************\
  inernal structures
\*****************************************/

/** runtime resource corresponding to a procedure or to the entire program
 *  A tag, cast of the address of a resource-set, is an interface to Fortran.
 */
struct _resourceSet_t {
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
  BOOL             isSuspended;  // true if already encountered DEALLOCATE stmt
  char            *orgAddr;        // local address of the allocated memory
  size_t           nbytes;         // allocated size of memory [bytes]
  void            *desc;           // address of the lower layer's descriptor 
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



/** structure to manage the history of malloc
 */
struct _memoryChunkStack_t {
  MemoryChunkOrder_t  *head;
  MemoryChunkOrder_t  *tail;
};

static MemoryChunkStack_t _mallocStack;

struct _memoryChunkOrder_t {
  MemoryChunkOrder_t  *prev;
  MemoryChunkOrder_t  *next;
  MemoryChunk_t       *chunk;
};


/***********************************************\
  hidden utility functions
\***********************************************/

int xmpf_coarray_allocated_()
{
  MemoryChunkOrder_t *chunkp;
  MemoryChunk_t *chunk;
  size_t size;

  size = 0;
  forallMemoryChunkOrder(chunkp) {
    chunk = chunkp->chunk;
    size += chunk->nbytes;
  }

  // returns illegal value if it exceeds huge(int)
  return size;
}

int xmpf_coarray_suspended_()
{
  MemoryChunkOrder_t *chunkp;
  MemoryChunk_t *chunk;
  size_t size;

  size = 0;
  forallMemoryChunkOrder(chunkp) {
    chunk = chunkp->chunk;
    if (chunk->isSuspended)
      size += chunk->nbytes;
  }

  // returns illegal value if it exceeds huge(int)
  return size;
}


/***********************************************\
  ALLOCATE statement
  Type-1: alloc/free by the low-level library
\***********************************************/

/*  1. malloc by the low-level library
 *  2. make a memoryChunk with a coarrayInfo
 */
void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                          int *count, int *element, void **tag)
{
  _XMPF_checkIfInTask("allocatable coarray allocation");
  ResourceSet_t *rset;

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MALLOC\n"
                          "  *count=%d, *element=%d, *tag=%p\n",
                          *count, *element, *tag);

  // malloc
  MemoryChunk_t *chunk = _mallocMemoryChunk(*count, (size_t)(*element));

  if (*tag != NULL) {
    rset = (ResourceSet_t*)(*tag);
    _linkMemoryChunk(rset, chunk);
  }

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _newCoarrayInfo();
  cinfo->nbytes = (*count) * (size_t)(*element);
  _addCoarrayInfo(chunk, cinfo);

  // output #1, #2
  *descPtr = (void*)cinfo;
  *crayPtr = chunk->orgAddr;
}


size_t _roundUpElementSize(int count, size_t element)
{
  size_t elementRU;

  // boundary check and recovery
  if (element % BOUNDARY_BYTE == 0) {
    elementRU = element;
  } else if (count == 1) {              // scalar or one-element array
    /* round up */
    elementRU = ROUND_UP_BOUNDARY(element);
    _XMPF_coarrayDebugPrint("round-up element size\n"
                            "  count=%d, element=%d to %zd\n",
                            count, element, elementRU);
  } else {
    /* restriction */
    _XMPF_coarrayFatal("violation of boundary: xmpf_coarray_malloc_() in %s",
                       __FILE__);
  }

  return elementRU;
}


MemoryChunk_t *_mallocMemoryChunk(int count, size_t element)
{
  MemoryChunk_t *chunk = _newMemoryChunk();
  size_t elementRU = _roundUpElementSize(count, element);
  chunk->nbytes = (size_t)count * elementRU;

  if (chunk->nbytes == 0) {
    _XMPF_coarrayDebugPrint("*** a memory-chunk not allocated\n");
    return chunk;
  }

  // _XMP_coarray_malloc() and set mallocInfo
  _XMP_coarray_malloc_info_1(chunk->nbytes, 1);    // set shape
  _XMP_coarray_malloc_image_info_1();                          // set coshape
  _XMP_coarray_malloc_do(&(chunk->desc), &(chunk->orgAddr));   // malloc

  _XMPF_coarrayDebugPrint("*** MemoryChunk %p allocated\n"
                          "  requred: %zd bytes (count=%d, element=%d)\n"
                          "  result : %d bytes\n",
                          chunk, chunk->nbytes, count, element, chunk->nbytes);

  // stack to mallocHistory
  _addMallocHistory(chunk);

  return chunk;
}


/***********************************************\
  DEALLOCATE statement
  Type-1: alloc/free by the low-level library
  Type-1a: to keep the reverse order of allocation,
           actual deallocation may be delayed.
  RETHINK!
\***********************************************/

void xmpf_coarray_free_(void **descPtr, void **tag)
{
  ResourceSet_t *rset = (ResourceSet_t*)(*tag);
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  MemoryChunk_t *chunk = cinfo->parent;

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_FREE\n"
                          "  MemoryChunk %p in ResourceSet %p\n",
                          chunk, rset);

  // unlink and free CoarrayInfo keeping MemoryChunk
  _unlinkCoarrayInfo(cinfo);
  _freeCoarrayInfo(cinfo);

  if (IsEmptyMemoryChunk(chunk)) {
    // unlink this memory chunk and suspend it to free
    _unlinkMemoryChunk(chunk);
    // chance to free suspended memory chunks
    _flushMallocHistory();
  }
}


/*****************************************\
  handling memory pool
   for static coarrays
\*****************************************/

void xmpf_coarray_malloc_pool_(void)
{
  _XMPF_coarrayDebugPrint("XMPF_COARRAY_MALLOC_POOL\n"
                          "  required pool_totalSize = %zd\n",
                          pool_totalSize);

  // init malloc/free history
  _initMallocHistory();

  // malloc
  pool_chunk = _mallocMemoryChunk(1, pool_totalSize);
  pool_currentAddr = pool_chunk->orgAddr;
}


/*
 * have a share of memory in the pool
 *    out: descPtr: pointer to descriptor CoarrayInfo_t
 *         crayPtr: cray pointer to the coarray object
 *    in:  count  : count of elements
 *         element: element size
 *         name   : name of the coarray (for debugging)
 *         namelen: character length of name
 */
void xmpf_coarray_share_pool_(void **descPtr, char **crayPtr,
                              int *count, int *element,
                              char *name, int *namelen)
{
  CoarrayInfo_t *cinfo =
    _getShareOfCoarray(*count, (size_t)(*element));

  cinfo->name = (char*)malloc(sizeof(char)*(*namelen + 1));
  strncpy(cinfo->name, name, *namelen);

  *descPtr = (void*)cinfo;
  *crayPtr = cinfo->baseAddr;
}


CoarrayInfo_t *_getShareOfCoarray(int count, size_t element)
{
  _XMPF_checkIfInTask("static coarray allocation");

  size_t elementRU = _roundUpElementSize(count, element);

  // allocate and set _coarrayInfo
  size_t thisSize = (size_t)count * elementRU;
  CoarrayInfo_t *cinfo = _newCoarrayInfo();
  cinfo->nbytes = thisSize;
  _addCoarrayInfo(pool_chunk, cinfo);

  // check: too large allocation
  if (pool_currentAddr + thisSize > pool_chunk->orgAddr + pool_totalSize) {
    _XMPF_coarrayFatal("insufficient memory pool for static coarrays: "
                      "xmpf_coarray_share_pool_() in %s", __FILE__);
  }

  _XMPF_coarrayDebugPrint("*** get share of coarray \'%s\' in memory pool:\n"
                          "  address = %p\n"
                          "  size    = %zd\n"
                          "  (originally, count = %d, element = %zd)\n",
                          cinfo->name, pool_currentAddr,
                          thisSize, count, element);

  cinfo->baseAddr = pool_currentAddr;
  cinfo->nbytes = thisSize;

  pool_currentAddr += thisSize;

  return cinfo;
}








void xmpf_coarray_count_size_(int *count, int *element)
{
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);

  _XMPF_coarrayDebugPrint("*** count-up allocation size: %zd[byte].\n", mallocSize);

  pool_totalSize += mallocSize;
}


void xmpf_coarray_proc_init_(void **tag)
{
  ResourceSet_t *resource;

  resource = _newResourceSet();
  *tag = (void*)resource;
}


void xmpf_coarray_proc_finalize_(void **tag)
{
  if (*tag == NULL)
    return;

  ResourceSet_t *rset = (ResourceSet_t*)(*tag);
  _freeResourceSet(rset);

  *tag = NULL;
}


/*****************************************\
   entry
\*****************************************/

/** generate and return a descriptor of a coarray dummy argument
 *   1. find the memory chunk that contains the coarray data object,
 *   2. generate coarrayInfo for the coarray dummy argument and link it 
 *      to the memory chunk, and
 *   3. return coarrayInfo as descPtr
 */
void xmpf_coarray_descptr_(void **descPtr, char *baseAddr, void **tag)
{
  ResourceSet_t *rset = (ResourceSet_t*)(*tag);
  MemoryChunkOrder_t *chunkP;
  MemoryChunk_t *chunk;
  BOOL found;

  if (rset == NULL)
    rset = _newResourceSet();

  _XMPF_coarrayDebugPrint("XMPF_COARRAY_DESCPTR\n");
  _XMPF_coarrayDebugPrint("  coarray dummy argument: %p\n", baseAddr);

  found = FALSE;
  forallMemoryChunkOrder(chunkP) {
    chunk = chunkP->chunk;

    _XMPF_coarrayDebugPrint("  comparing with a chunk: %p  +%zd\n",
                            chunk->orgAddr, chunk->nbytes);

    if (chunk->orgAddr <= baseAddr && baseAddr < chunk->orgAddr + chunk->nbytes) {
      // found the memory chunk
      found = TRUE;
      break;
    }
  }

  if (!found) {
    _XMPF_coarrayDebugPrint("did not find the memory chunk that contains the coarray:\n"
                            "  start address of the coarray dummy arg.: %p\n", 
                            baseAddr);

  }

  _XMPF_coarrayDebugPrint("  found.\n");

  // generate a new descPtr for the dummy coarray
  CoarrayInfo_t *cinfo = _newCoarrayInfo();
  _addCoarrayInfo(chunk, cinfo);
  //cinfo->size = ???
  
  // return coarrayInfo as descPtr
  *descPtr = (void*)cinfo;
}


/*****************************************\
   management of the history of malloc/free
\*****************************************/

void _initMallocHistory(void)
{
  _mallocStack.head = _newMemoryChunkOrder(NULL);
  _mallocStack.tail = _newMemoryChunkOrder(NULL);
  _mallocStack.head->next = _mallocStack.tail;
  _mallocStack.tail->prev = _mallocStack.head;
}


void _addMallocHistory(MemoryChunk_t *chunk)
{
  MemoryChunkOrder_t *chunkP = _newMemoryChunkOrder(chunk);
  MemoryChunkOrder_t *tailP = _mallocStack.tail;
  if (tailP == NULL) {
    _initMallocHistory();
    tailP = _mallocStack.tail;
  }
  MemoryChunkOrder_t *lastP = tailP->prev;

  lastP->next = chunkP;
  chunkP->prev = lastP;
  chunkP->next = tailP;
  tailP->prev = chunkP;
}


MemoryChunkOrder_t *_newMemoryChunkOrder(MemoryChunk_t *chunk)
{
  MemoryChunkOrder_t *chunkP =
    (MemoryChunkOrder_t*)malloc(sizeof(MemoryChunkOrder_t));
  chunkP->prev = NULL;
  chunkP->next = NULL;
  chunkP->chunk = chunk;

  return chunkP;
}


/*  free deallocated coarry data objects as much as possible, keeping the
 *  reverse order of allocations.
 */
void _flushMallocHistory()
{
  MemoryChunkOrder_t *chunkP;

  _XMPF_coarrayDebugPrint("FLUSH suspended MemoryChunk\n");

  forallMemoryChunkOrderRev(chunkP) {
    if (!chunkP->chunk->isSuspended)
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
}


/* translate n.
 * set the name of coarray object
 */
void xmpf_coarray_set_varname_(void **descPtr, char *name, int *namelen)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);

  cinfo->name = (char*)malloc(sizeof(char)*(*namelen + 1));
  memcpy(cinfo->name, name, *namelen);
  cinfo->name[*namelen] = '\0';
}


/*****************************************\
  intrinsic functions
\*****************************************/

/*
 * get an image index corresponding to the current lower and upper cobounds
 */
int xmpf_coarray_get_image_index_(void **descPtr, int *corank, ...)
{
  int i, idx, lb, ub, factor, count;
  va_list(args);
  va_start(args, corank);

  CoarrayInfo_t *cp = (CoarrayInfo_t*)(*descPtr);

  if (cp->corank != *corank) {
    _XMPF_coarrayFatal("INTERNAL: corank %d here is different from the declared corank %d",
                       *corank, cp->corank);
  }

  count = 0;
  factor = 1;
  for (i = 0; i < *corank; i++) {
    idx = *va_arg(args, int*);
    lb = cp->lcobound[i];
    ub = cp->ucobound[i];
    if (idx < lb || ub < idx) {
      _XMPF_coarrayFatal("%d-th cosubscript of \'%s\', %d, is out of range %d to %d.",
                         i+1, cp->name, idx, lb, ub);
    }
    count += (idx - lb) * factor;
    factor *= cp->cosize[i];
  }

  va_end(args);

  return count + 1;
}




/*****************************************\
  access functions for ResourceSet_t
\*****************************************/

ResourceSet_t *_newResourceSet(void)
{
  ResourceSet_t *rset =
    (ResourceSet_t*)malloc(sizeof(ResourceSet_t));

  rset->headChunk = _newMemoryChunk();
  rset->tailChunk = _newMemoryChunk();
  rset->headChunk->next = rset->tailChunk;
  rset->tailChunk->prev = rset->headChunk;
  rset->headChunk->parent = rset;
  rset->tailChunk->parent = rset;
  //_linkResourceSet(rset);
  return rset;
}

void _freeResourceSet(ResourceSet_t *rset)
{
  MemoryChunk_t *chunk;

  forallMemoryChunk (chunk, rset) {
    // allocated -> suspended (waiting for freeing)
    _unlinkMemoryChunk(chunk);
  }

  free(rset);
}


/*****************************************\
  access functions for MemoryChunk_t
\*****************************************/

MemoryChunk_t *_newMemoryChunk(void)
{
  MemoryChunk_t *chunk =
    (MemoryChunk_t*)malloc(sizeof(MemoryChunk_t));

  chunk->prev = NULL;
  chunk->next = NULL;
  chunk->headCoarray = _newCoarrayInfo();
  chunk->tailCoarray = _newCoarrayInfo();
  chunk->headCoarray->next = chunk->tailCoarray;
  chunk->tailCoarray->prev = chunk->headCoarray;
  chunk->headCoarray->parent = chunk;
  chunk->tailCoarray->parent = chunk;
  chunk->isSuspended = FALSE;
  return chunk;
}


void _linkMemoryChunk(ResourceSet_t *rset, MemoryChunk_t *chunk2)
{
  MemoryChunk_t *chunk3 = rset->tailChunk;
  MemoryChunk_t *chunk1 = chunk3->prev;

  chunk1->next = chunk2;
  chunk3->prev = chunk2;

  chunk2->prev = chunk1;
  chunk2->next = chunk3;
  chunk2->parent = chunk1->parent;
}


void _unlinkMemoryChunk(MemoryChunk_t *chunk2)
{
  chunk2->isSuspended = TRUE;

  MemoryChunk_t *chunk1 = chunk2->prev;
  MemoryChunk_t *chunk3 = chunk2->next;

  chunk1->next = chunk3;
  chunk3->prev = chunk1;

  chunk2->next = chunk2->prev = chunk2->parent = NULL;
}


/*  current restriction of the lower-level library:
 *   only the last allocated data can be freed.
 */
void _freeMemoryChunk(MemoryChunk_t *chunk)
{
  CoarrayInfo_t *cinfo;

  forallCoarrayInfo (cinfo, chunk) {
    _unlinkCoarrayInfo(cinfo);
    _freeCoarrayInfo(cinfo);
  }

  // found a formal-deallocated memory chunk that should be free
  _XMPF_coarrayDebugPrint("*** freeing suspended MemoryChunk %p, %zd bytes\n",
                          chunk, chunk->nbytes);

  // free the last memory chunk object
  _XMP_coarray_lastly_deallocate();

  free(chunk);
}


/*****************************************\
  access functions for CoarrayInfo_t
\*****************************************/

static CoarrayInfo_t *_newCoarrayInfo(void)
{
  CoarrayInfo_t *cinfo =
    (CoarrayInfo_t*)malloc(sizeof(CoarrayInfo_t));
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
}

void _freeCoarrayInfo(CoarrayInfo_t *cinfo)
{
  free(cinfo->name);
  free(cinfo->lcobound);
  free(cinfo->ucobound);
  free(cinfo->cosize);
  free(cinfo);
}


/***********************************************\
   inquire functions
\***********************************************/

void *_XMPF_get_coarrayDesc(void *descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  return cinfo->parent->desc;
}

size_t _XMPF_get_coarrayOffset(void *descPtr, char *baseAddr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)descPtr;
  char* orgAddr = cinfo->parent->orgAddr;
  int offset = ((size_t)baseAddr - (size_t)orgAddr);
  return offset;
}


