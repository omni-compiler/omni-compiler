#ifndef _XMPCO_INTERNAL_H
#define _XMPCO_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "xmp_internal.h"
#include "xmp_data_struct.h"
#include "xmpco_params.h"


#define BOOL   int
#define TRUE   1
#define FALSE  0

#define _ROUND_UP(n,p)        (((((size_t)(n))-1)/(p)+1)*(p))
#define _ROUND_UP_PLUS(n,p)   (((n)>0) ? _ROUND_UP(n,p) : (p))
#define ROUND_UP_COMM(n)      _ROUND_UP((n),COMM_UNIT)
#define ROUND_UP_MALLOC(n)    _ROUND_UP_PLUS((n),MALLOC_UNIT)


/*****************************************      \
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
  Built-in Functions for Users
\*****************************************/
// inquiry functions (xmpco_alloc.c)
extern size_t xmp_coarray_malloc_bytes(void);
extern size_t xmp_coarray_allocated_bytes(void);
extern size_t xmp_coarray_garbage_bytes(void);


/*****************************************\
  Object-library Interface
\*****************************************/
// allocation & registration (xmpco_alloc.c)
extern CoarrayInfo_t *
XMPCO_malloc_coarray(char **addr, int count, size_t element,
                     ResourceSet_t *rset);
extern CoarrayInfo_t *
XMPCO_regmem_coarray(void *var, int count, size_t element,
                     ResourceSet_t *rset);
extern CoarrayInfo_t *
XMPCO_malloc_staticCoarray(char **addr, int count, size_t element,
                           int namelen, char *name);
extern CoarrayInfo_t *
XMPCO_regmem_staticCoarray(void *var, int count, size_t element,
                           int namelen, char *name);

// deallocation & deregistration (xmpco_alloc.c)
extern void XMPCO_free_coarray(CoarrayInfo_t *cinfo);
extern void XMPCO_deregmem_coarray(CoarrayInfo_t *cinfo);

// initialization & finalization (xmpco_alloc.c)
extern void XMPCO_malloc_pool(void);
extern void XMPCO_count_size(int count, size_t element);

// procedure entry & exit (xmpco_alloc.c)
extern void XMPCO_prolog(ResourceSet_t **rset, int namelen, char *name);
extern void XMPCO_epilog(ResourceSet_t **rset);

// find descriptor of a dummy argument (xmpco_alloc.c)
extern CoarrayInfo_t *XMPCO_find_descptr(char *addr, int namelen, char *name);

// synchronizations (xmpco_sync.c)
extern void XMPCO_sync_all(void);
extern void XMPCO_sync_all_auto(void);
extern void XMPCO_sync_all_withComm(MPI_Comm comm);

// PUT and GET
extern void XMPCO_GET_arrayStmt(CoarrayInfo_t *descPtr, char *baseAddr,
                          int element, int coindex, char *localAddr,
                          int rank, int skip[], int skip_local[], int count[]);
extern void XMPCO_GET_scalarExpr(CoarrayInfo_t *descPtr, char *baseAddr,
                                 int element, int coindex, char *result);
extern void XMPCO_GET_arrayExpr(CoarrayInfo_t *descPtr, char *baseAddr,
                                int element, int coindex, char *result,
                                int rank, int skip[], int count[]);

extern void XMPCO_PUT_scalarStmt(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                                 int coindex, char *rhs, BOOL synchronous);
extern void XMPCO_PUT_arrayStmt(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                                int coindex, char *rhsAddr, int rank,
                                int skip[], int skip_rhs[], int count[],
                                BOOL synchronous);
extern void XMPCO_PUT_spread(CoarrayInfo_t *descPtr, char *baseAddr, int element,
                             int coindex, char *rhs, int rank,
                             int skip[], int count[], BOOL synchronous);


// inquire functions (xmpco_lib.c)
extern int XMPCO_this_image_coarray_dim(CoarrayInfo_t *cinfo,
                                        int corank, int dim);
extern void XMPCO_this_image_coarray(CoarrayInfo_t *cinfo,
                                     int corank, int image[]);


/*****************************************\
  set/get options and environment vars
\*****************************************/
// set functions (xmpco_params.c)
extern void _XMPCO_set_poolThreshold(unsigned size);
extern void _XMPCO_set_localBufSize(unsigned size);
extern void _XMPCO_set_isMsgMode(BOOL sw);
extern void _XMPCO_set_isMsgMode_quietly(BOOL sw);
extern void _XMPCO_set_isSafeBufferMode(BOOL sw);
extern void _XMPCO_set_isSyncPutMode(BOOL sw);
extern void _XMPCO_set_isEagerCommMode(BOOL sw);

extern void _XMPCO_reset_isMsgMode(void);

// get functions (xmpco_params.c)
extern unsigned _XMPCO_get_poolThreshold(void);
extern size_t   _XMPCO_get_localBufSize(void);
extern BOOL     _XMPCO_get_isMsgMode(void);
extern BOOL     _XMPCO_get_isSafeBufferMode(void);
extern BOOL     _XMPCO_get_isSyncPutMode(void);
extern BOOL     _XMPCO_get_isEagerCommMode(void);


/*****************************************\
  Internal library Interface
\*****************************************/
// CoarrayInfo (xmpco_alloc.c)
extern char *_XMPCO_get_nameOfCoarray(CoarrayInfo_t *cinfo);
extern char *_XMPCO_get_baseAddrOfCoarray(CoarrayInfo_t *cinfo);
extern size_t _XMPCO_get_sizeOfCoarray(CoarrayInfo_t *cinfo);
extern size_t _XMPCO_get_offsetInCoarray(CoarrayInfo_t *cinfo, char *addr);
extern void _XMPCO_set_corank(CoarrayInfo_t *cp, int corank);
extern void _XMPCO_set_codim_withBounds(CoarrayInfo_t *cp, int dim,
                                         int lb, int ub);
extern void _XMPCO_set_codim_withSize(CoarrayInfo_t *cp, int dim,
                                       int lb, int size);
extern void _XMPCO_set_varname(CoarrayInfo_t *cp, int namelen,
                                char *name);
extern CoarrayInfo_t* _XMPCO_set_nodes(CoarrayInfo_t *cinfo,
                                        _XMP_nodes_t *nodes);

// MemoryChunk (xmpco_alloc.c)
extern void *_XMPCO_get_descForMemoryChunk(CoarrayInfo_t *cinfo);
extern char *_XMPCO_get_orgAddrOfMemoryChunk(CoarrayInfo_t *cinfo);
extern size_t _XMPCO_get_sizeOfMemoryChunk(CoarrayInfo_t *cinfo);
extern size_t _XMPCO_get_offsetInMemoryChunk(CoarrayInfo_t *cinfo, char *addr);
extern BOOL _XMPCO_isAddrInMemoryChunk(char *localAddr, CoarrayInfo_t *cinfo);

// system-defined coarray variables (xmpco_alloc.c)
extern void *_XMPCO_get_infoOfCtrlData(char **baseAddr, size_t *offset,
                                        char **name);
extern void *_XMPCO_get_infoOfLocalBuf(char **baseAddr, size_t *offset,
                                        char **name);

// other inquire functions (xmpco_alloc.c)
extern void *_XMPCO_get_desc_fromLocalAddr(char *localAddr, char **orgAddr,
                                           size_t *offset, char **name);
extern MPI_Comm _XMPCO_get_comm_fromCoarrayInfo(CoarrayInfo_t *cinfo);

// error handling and messages (xmpco_msg.c)
extern void _XMPCO_fatal(char *format, ...);
extern void _XMPCO_debugPrint(char *format, ...);

// initialization for PUT/GET (xmpco_put.c, xmpco_get_*.c)
extern void _XMPCO_coarrayInit_get(void);
extern void _XMPCO_coarrayInit_getsub(void);
extern void _XMPCO_coarrayInit_put(void);


// TEMPORARY restriction check
extern int _XMPCO_nowInTask(void);   // for restriction check
extern void _XMPCO_checkIfInTask(char *msgopt);   // restriction check

// images (xmpco_lib.c)
extern void _XMPCO_set_initialThisImage(void);
extern void _XMPCO_set_initialNumImages(void);
extern int _XMPCO_get_initialThisImage(void);
extern int _XMPCO_get_initialNumImages(void);

extern int _XMPCO_get_currentThisImage(void);
extern int _XMPCO_get_currentNumImages(void);

extern MPI_Comm _XMPCO_get_currentComm(void);
extern BOOL _XMPCO_is_subset_exec(void);
extern int _XMPCO_transImage_withComm(MPI_Comm comm1, int image1,
                                      MPI_Comm comm2);
extern int _XMPCO_transImage_current2initial(int image);
extern int _XMPCO_get_initial_image_withDescPtr(int image,
                                                CoarrayInfo_t *descPtr);

// image-directive nodes (xmpco_lib.c)
extern void _XMPCO_clean_imageDirNodes(void);
extern void _XMPCO_set_imageDirNodes(_XMP_nodes_t *nodes);
extern _XMP_nodes_t *_XMPCO_get_imageDirNodes(void);
extern _XMP_nodes_t *_XMPCO_consume_imageDirNodes(void);

// values obtained from nodes (xmpco_lib.c)
extern MPI_Comm _XMPCO_get_comm_of_nodes(_XMP_nodes_t *nodes);
extern int _XMPCO_num_images_onNodes(_XMP_nodes_t *nodes);
extern int _XMPCO_this_image_onNodes(_XMP_nodes_t *nodes);

// current communicator (xmpco_lib.c)
extern MPI_Comm _XMPCO_get_comm_current(void);
extern MPI_Comm _XMPCO_consume_comm_current(void);


/*****************************************\
  lower-level interface
  (TEMPORARY)
\*****************************************/

//#include "xmp_func_decl.h"   // conflicts with xmp_internal.h

extern void xmp_sync_memory(const int* status);
extern void xmp_sync_all(const int* status);
extern void xmp_sync_image(int image, int* status);
extern void xmp_sync_images(const int num, int* image_set, int* status);
extern void xmp_sync_images_all(int* status);

extern void _XMP_coarray_malloc_info_1(const long, const size_t);
extern void _XMP_coarray_malloc_image_info_1();
extern void _XMP_coarray_malloc(void **, void *);
extern void _XMP_coarray_regmem(void **, void *);
extern void _XMP_coarray_contiguous_put(const int, void*, const void*, const long, const long, const long, const long);
extern void _XMP_coarray_contiguous_get(const int, void*, const void*, const long, const long, const long, const long);

extern void _XMP_coarray_rdma_coarray_set_1(const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_1(const long, const long, const long, const long, const size_t);
extern void _XMP_coarray_rdma_image_set_1(const int);
extern void _XMP_coarray_put(void*, void*, void *);
extern void _XMP_coarray_get(void*, void*, void *);


#endif
