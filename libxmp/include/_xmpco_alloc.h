// only for xmpco_alloc.c

#ifndef _XMPCO_ALLOC_H
#define _XMPCO_ALLOC_H

#include "xmpco_internal.h"
#include "_xmpco_alloc.h"

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
  void            *desc;         // descriptor in the lower-layer library
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


#endif
