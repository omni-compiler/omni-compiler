#ifndef _XMP_CO_INTERNAL_H
#define _XMP_CO_INTERNAL_H

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
  user functions in xmp_co_alloc.c
\*****************************************/
// inquire functions
extern size_t xmp_coarray_malloc_bytes(void);
extern size_t xmp_coarray_allocated_bytes(void);
extern size_t xmp_coarray_garbage_bytes(void);


/*****************************************\
  utilities in xmp_co_alloc.c
\*****************************************/
// inquire functions
extern char *_XMP_CO_get_coarrayName(void *descPtr);
extern char *_XMP_CO_get_coarrayBaseAddr(void *descPtr);
extern size_t _XMP_CO_get_coarraySize(void *descPtr);
extern size_t _XMP_CO_get_coarrayOffset(void *descPtr, char *addr);

extern void *_XMP_CO_get_coarrayChunkDesc(void *descPtr);
extern char *_XMP_CO_get_coarrayChunkOrgAddr(void *descPtr);
extern size_t _XMP_CO_get_coarrayChunkSize(void *descPtr);
extern size_t _XMP_CO_get_coarrayChunkOffset(void *descPtr, char *addr);

extern void *_XMP_CO_get_cntlDataCoarrayDesc(char **baseAddr, size_t *offset,
                                           char **name);
extern void *_XMP_CO_get_localBufCoarrayDesc(char **baseAddr, size_t *offset,
                                           char **name);
//extern BOOL _XMP_CO_isAddrInRangeOfCoarray(char *localAddr, void *descPtr);    // obsolated ?
extern BOOL _XMP_CO_isAddrInCoarrayChunk(char *localAddr, void *descPtr);
extern void *_XMP_CO_get_coarrayDescFromAddr(char *localAddr, char **orgAddr,
                                           size_t *offset, char **nameAddr);
extern MPI_Comm _XMP_CO_get_communicatorFromDescPtr(void *descPtr);


/*****************************************\
  object interface in xmp_co_alloc.c
\*****************************************/
// allocation & registration
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

// deallocation & deregistration
extern void _XMP_CO_free_coarray(CoarrayInfo_t *cinfo);
extern void _XMP_CO_deregmem_coarray(CoarrayInfo_t *cinfo);

// initialization & finalization
extern void _XMP_CO_malloc_pool(void);
extern void _XMP_CO_alloc_static(void **descPtr, char **crayPtr,
                                 int count, size_t element,
                                 int namelen, char *name);
extern void _XMP_CO_prolog(void **tag, int namelen, char *name);
extern void _XMP_CO_epilog(void **tag);

// tools
extern void _XMP_CO_coarray_count_size(int count, size_t element);
extern MPI_Comm _XMP_CO_communicatorFromCoarrayInfo(CoarrayInfo_t *cinfo);
extern void* _XMP_CO_find_descptr(char *addr, int namelen, char *name);


/*****************************************\
  set functions in xmp_co_alloc.c
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

#endif
