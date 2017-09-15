/**************************************************\
    internal header for Coarray Memory Allocation
\**************************************************/

#ifndef XMPF_INTERNAL_CO_ALLOC_H
#define XMPF_INTERNAL_CO_ALLOC_H


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
//typedef struct _coarrayInfo_t  CoarrayInfo_t;   moved into xmpf_internal_coarray.h

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
  size_t           nbytes;       // allocated size of memory [bytes]
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
  size_t          size;      // size of the coarray [bytes]
  int             corank;    // number of codimensions
  int            *lcobound;  // array of lower cobounds [0..(corank-1)]
  int            *ucobound;  // array of upper cobounds [0..(corank-1)]
  int            *cosize;    // cosize[k] = max(ucobound[k]-lcobound[k]+1, 0)
  _XMP_nodes_t   *nodes;     // XMP descriptor for the mapping nodes (if any)
};



/** MEMORY MANAGEMENT STRUCTURE-II (for dynamic ALLOCATE/DEALLOCATE stmts. in Fortran)
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


/** MEMORY MANAGEMENT STRUCTURE-III (for binary search for memory chunk)
 *  table of memory chunks sorted in order of local address
 */
struct _sortedChunkTable_t {
  unsigned long   orgAddr;
  MemoryChunk_t  *chunk;
};


/*****************************************\
  hidden inquire functions
\*****************************************/

extern size_t xmp_coarray_malloc_bytes(void);
extern size_t xmp_coarray_allocated_bytes(void);
extern size_t xmp_coarray_garbage_bytes(void);


/*****************************************\
  Allocation & Registration
\*****************************************/
extern void _XMP_CO_malloc_coarray(void **descPtr, char **addr,
                                   int count, size_t element,
                                   ResourceSet_t *rset);
extern void _XMP_CO_regmem_coarray(void **descPtr, void *var,
                                   int count, size_t element,
                                   ResourceSet_t *rset);
extern void _XMP_CO_alloc_static_coarray(void **descPtr, char **addr,
                                         int count, size_t element,
                                         int namelen, char *name);
extern void _XMP_CO_regmem_static_coarray(void **descPtr, void *var,
                                          int count, size_t element,
                                          int namelen, char *name);

/*****************************************\
  Deallocation & Deregistration
\*****************************************/
extern void _XMP_CO_free_coarray(CoarrayInfo_t *cinfo);
extern void _XMP_CO_deregmem_coarray(CoarrayInfo_t *cinfo);

/*****************************************\
  Initialization/Finalization
  Handling memory pool
\*****************************************/
extern void _XMP_CO_malloc_pool(void);
extern void _XMP_CO_alloc_static(void **descPtr, char **crayPtr,
                                 int count, size_t element,
                                 int namelen, char *name);


extern void xmp_coarray_count_size(int count, size_t element);

extern MPI_Comm _XMP_CO_communicatorFromCoarrayInfo(CoarrayInfo_t *cinfo);

extern void _XMP_CO_prolog(void **tag, int namelen, char *name);
extern void _XMP_CO_epilog(void **tag);

/*****************************************\
  Find descriptor from the local address
\*****************************************/
extern void* xmp_coarray_find_descptr(char *addr, int namelen, char *name);

/*****************************************\
  set attributes of CoarrayInfo
\*****************************************/
extern void _XMP_CO_set_corank(CoarrayInfo_t *cp, int corank);
extern void _XMP_CO_set_codim_withBOUNDS(CoarrayInfo_t *cp, int dim,
                                         int lb, int ub);
extern void _XMP_CO_set_varname(CoarrayInfo_t *cp, int namelen,
                                char *name);
extern CoarrayInfo_t* _XMP_CO_set_nodes(CoarrayInfo_t *cinfo,
                                        _XMP_nodes_t *nodes);

/*****************************************\
   STRUCTURE-II
   management of the history of malloc/free
\*****************************************/
extern MemoryChunkOrder_t *_XMP_CO_newMemoryChunkOrder(MemoryChunk_t *chunk);
extern void _XMP_CO_garbageCollectMallocHistory(void);
extern void _XMP_CO_unlinkMemoryChunkOrder(MemoryChunkOrder_t *chunkP2);
extern void _XMP_CO_freeMemoryChunkOrder(MemoryChunkOrder_t *chunkP);


/*****************************************\
  lower library functions and variables
\*****************************************/
extern void *MALLOC(size_t size);
extern void *CALLOC(size_t nmemb, size_t size);
extern void _FREE(void *ptr);
extern void FREE_ResourceSet_t(ResourceSet_t *rset);
extern void FREE_MemoryChunkOrder_t(MemoryChunkOrder_t *chunkp);
extern void FREE_MemoryChunk_t(MemoryChunk_t *chunk);
extern void FREE_CoarrayInfo_t(CoarrayInfo_t *cinfo);
extern void FREE_string(char *name);
extern void FREE_int_n(int *intp, int n);


#endif /*XMPF_INTERNAL_CO_ALLOC_H*/
