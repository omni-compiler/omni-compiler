#ifndef _XMP_CO_INTERNAL_H
#define _XMP_CO_INTERNAL_H

/*****************************************\
  xmp_co_alloc.c
    typedef
\*****************************************/
// MEMORY MANAGEMENT STRUCTURE-I (linkage with procedures)
typedef struct _resourceSet_t  ResourceSet_t;   // corresponding to a procedure
typedef struct _memoryChunk_t  MemoryChunk_t;   // contains one or more coarrays
typedef struct _coarrayInfo_t  CoarrayInfo_t;   // corresponding to a coarray

// MEMORY MANAGEMENT STRUCTURE-II (management of the order of alloc/free)
typedef struct _memoryChunkOrder_t  MemoryChunkOrder_t;  // contains a pointer to Chunk

// MEMORY MANAGEMENT STRUCTURE-III (for binary search of memory chunk)
typedef struct _sortedChunkTable_t  SortedChunkTable_t;  // contains a pointer to Chunk


/*****************************************\
  user functions
\*****************************************/
// built-in functions (xmp_co_alloc.c)
extern size_t xmp_coarray_malloc_bytes(void);
extern size_t xmp_coarray_allocated_bytes(void);
extern size_t xmp_coarray_garbage_bytes(void);


/*****************************************\
  object interface
\*****************************************/
// allocation & registration (xmp_co_alloc.c)
extern CoarrayInfo_t *
_XMP_CO_malloc_coarray(char **addr, int count, size_t element,
                       ResourceSet_t *rset);
extern CoarrayInfo_t *
_XMP_CO_regmem_coarray(void *var, int count, size_t element,
                       ResourceSet_t *rset);
extern CoarrayInfo_t *
_XMP_CO_malloc_staticCoarray(char **addr, int count, size_t element,
                             int namelen, char *name);
extern CoarrayInfo_t *
_XMP_CO_regmem_staticCoarray(void *var, int count, size_t element,
                             int namelen, char *name);

// deallocation & deregistration (xmp_co_alloc.c)
extern void _XMP_CO_free_coarray(CoarrayInfo_t *cinfo);
extern void _XMP_CO_deregmem_coarray(CoarrayInfo_t *cinfo);

// initialization & finalization (xmp_co_alloc.c)
extern void _XMP_CO_malloc_pool(void);
extern void _XMP_CO_count_size(int count, size_t element);

// procedure entry & exit (xmp_co_alloc.c)
extern void _XMP_CO_prolog(ResourceSet_t **rset, int namelen, char *name);
extern void _XMP_CO_epilog(ResourceSet_t **rset);

// find descriptor of a dummy argument (xmp_co_alloc.c)
extern CoarrayInfo_t *_XMP_CO_find_descptr(char *addr, int namelen, char *name);

/*****************************************\
  inquire functions (xmp_co_alloc.c)
\*****************************************/
// CoarrayInfo
extern char *_XMP_CO_get_nameOfCoarray(CoarrayInfo_t *cinfo);
extern char *_XMP_CO_get_baseAddrOfCoarray(CoarrayInfo_t *cinfo);
extern size_t _XMP_CO_get_sizeOfCoarray(CoarrayInfo_t *cinfo);
extern size_t _XMP_CO_get_offsetInCoarray(CoarrayInfo_t *cinfo, char *addr);

// MemoryChunk
extern void *_XMP_CO_get_descForMemoryChunk(CoarrayInfo_t *cinfo);
extern char *_XMP_CO_get_orgAddrOfMemoryChunk(CoarrayInfo_t *cinfo);
extern size_t _XMP_CO_get_sizeOfMemoryChunk(CoarrayInfo_t *cinfo);
extern size_t _XMP_CO_get_offsetInMemoryChunk(CoarrayInfo_t *cinfo, char *addr);
extern BOOL _XMP_CO_isAddrInMemoryChunk(char *localAddr, CoarrayInfo_t *cinfo);

// built-in coarray variables
extern void *_XMP_CO_get_infoOfCtrlData(char **baseAddr, size_t *offset,
                                        char **name);
extern void *_XMP_CO_get_infoOfLocalBuf(char **baseAddr, size_t *offset,
                                        char **name);

/*****************************************\
  set functions (xmp_co_alloc.c)
\*****************************************/
extern void _XMP_CO_set_corank(CoarrayInfo_t *cp, int corank);
extern void _XMP_CO_set_codim_withBounds(CoarrayInfo_t *cp, int dim,
                                         int lb, int ub);
extern void _XMP_CO_set_codim_withSize(CoarrayInfo_t *cp, int dim,
                                       int lb, int size);
extern void _XMP_CO_set_varname(CoarrayInfo_t *cp, int namelen,
                                char *name);
extern CoarrayInfo_t* _XMP_CO_set_nodes(CoarrayInfo_t *cinfo,
                                        _XMP_nodes_t *nodes);


/*****************************************\
  utilities (xmp_co_alloc.c)
\*****************************************/
// search for the memory chunk from a local address
extern void *_XMP_CO_get_descFromLocalAddr(char *localAddr, char **orgAddr,
                                           size_t *offset, char **name);

// tools
extern MPI_Comm _XMP_CO_communicatorFromCoarrayInfo(CoarrayInfo_t *cinfo);


#endif
