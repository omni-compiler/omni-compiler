#include "xmpf_internal.h"

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

#define forallSegmentInfo(si,ri) for(SegmentInfo_t *_si1_ = ((si)=(ri)->headSegment->next)->next; \
                                     _si1_ != NULL;                     \
                                     (si) = _si1_)

#define forallCoarrayInfo(ci,si) for(CoarrayInfo_t *_ci1_ = ((ci)=(si)->headCoarray->next)->next; \
                                     _ci1_ != NULL;                     \
                                     (ci) = _ci1_)

#define IsFirstCoarrayInfo(ci)  ((ci)->prev->prev == NULL)
#define IsLastCoarrayInfo(ci)   ((ci)->next->next == NULL)
#define IsOnlyCoarrayInfo(ci)   (IsFirstCoarrayInfo(ci) && IsLastCoarrayInfo(ci))

#define IsFirstSegmentInfo(si)  ((si)->prev->prev == NULL)
#define IsLastSegmentInfo(si)   ((si)->next->next == NULL)
#define IsOnlySegmentInfo(si)   (IsFirstSegmentInfo(si) && IsLastSegmentInfo(si))


typedef struct _resourceInfo_t ResourceInfo_t;
typedef struct _segmentInfo_t  SegmentInfo_t;
typedef struct _coarrayInfo_t  CoarrayInfo_t;


/*****************************************\
  inernal structures
\*****************************************/

/* structure for each procedure and for the entire program
 */
struct _resourceInfo_t {
  SegmentInfo_t *headSegment;
  SegmentInfo_t *tailSegment;
};

/* structure for each malloc/free call
 */
struct _segmentInfo_t {
  SegmentInfo_t   *prev;
  SegmentInfo_t   *next;
  ResourceInfo_t  *parent;
  char            *orgAddr;      // local address of the allocated memory
  char            *endAddr;      // orgAddr + (count * element) 
  int              count;        // size of allocated data [elements]
  size_t           element;      // size of the elmement [bytes]
  void            *desc;         // address of the lower layer's descriptor 
  CoarrayInfo_t   *headCoarray;
  CoarrayInfo_t   *tailCoarray;
};

static ResourceInfo_t *_newResourceInfo(void);
static void _freeResourceInfo(ResourceInfo_t *rinfo);

static SegmentInfo_t *_newSegmentInfo(void);
static void _linkSegmentInfo(ResourceInfo_t *rinfo, SegmentInfo_t *sinfo2);
static void _unlinkSegmentInfo(SegmentInfo_t *sinfo2);
static void _freeSegmentInfo(SegmentInfo_t *sinfo);

static SegmentInfo_t *pool_sinfo = NULL;
static size_t pool_totalSize = 0;
static char *pool_currentAddr;

static SegmentInfo_t *_mallocSegment(int count, size_t element);


/* structure for each coarray variable
 * (Some coarrays can be allocated together and be connected with
 *  the same struct _segmentInfo_t.)
 */
struct _coarrayInfo_t {
  CoarrayInfo_t *prev;
  CoarrayInfo_t *next;
  SegmentInfo_t  *parent;
  char           *name;      // name of the variable (for debug message)
  char           *baseAddr;  // local address of the coarray (cray pointer)
  char           *endAddr;   // baseAddr + (count * element) 
  int             count;     // size of the coarray [elements]
  size_t          element;   // size of the elmement [bytes]
  int             corank;    // number of codimensions
  int            *lcobound;  // array of lower cobounds [0..(corank-1)]
  int            *ucobound;  // array of upper cobounds [0..(corank-1)]
  int            *cosize;    // cosize[k] = max(ucobound[k]-lcobound[k]+1, 0)
};

static CoarrayInfo_t *_mallocCoarrayInfo(int count, size_t element);
static void _addCoarrayInfo(SegmentInfo_t *sinfo, CoarrayInfo_t *cinfo2);
static void _unlinkCoarrayInfo(CoarrayInfo_t *cinfo2);
static void _freeCoarrayInfo(CoarrayInfo_t *cinfo);

static CoarrayInfo_t *_getCoarray(int count, size_t element);


/*****************************************\
  ResourceInfo_t
\*****************************************/

ResourceInfo_t *_newResourceInfo(void)
{
  ResourceInfo_t *rinfo =
    (ResourceInfo_t*)malloc(sizeof(ResourceInfo_t));

  rinfo->headSegment = _newSegmentInfo();
  rinfo->tailSegment = _newSegmentInfo();
  rinfo->headSegment->next = rinfo->tailSegment;
  rinfo->tailSegment->prev = rinfo->headSegment;
  rinfo->headSegment->parent = rinfo;
  rinfo->tailSegment->parent = rinfo;
  return rinfo;
}

void _freeResourceInfo(ResourceInfo_t *rinfo)
{
  SegmentInfo_t *sinfo;

  forallSegmentInfo (sinfo, rinfo) {
    _unlinkSegmentInfo(sinfo);
    _freeSegmentInfo(sinfo);
  }
}



/*****************************************\
  SegmentInfo_t
\*****************************************/

SegmentInfo_t *_newSegmentInfo(void)
{
  SegmentInfo_t *sinfo =
    (SegmentInfo_t*)malloc(sizeof(SegmentInfo_t));

  sinfo->prev = NULL;
  sinfo->next = NULL;
  sinfo->headCoarray = _mallocCoarrayInfo(0,0);
  sinfo->tailCoarray = _mallocCoarrayInfo(0,0);
  sinfo->headCoarray->next = sinfo->tailCoarray;
  sinfo->tailCoarray->prev = sinfo->headCoarray;
  sinfo->headCoarray->parent = sinfo;
  sinfo->tailCoarray->parent = sinfo;
  return sinfo;
}


void _linkSegmentInfo(ResourceInfo_t *rinfo, SegmentInfo_t *sinfo2)
{
  SegmentInfo_t *sinfo3 = rinfo->tailSegment;
  SegmentInfo_t *sinfo1 = sinfo3->prev;

  sinfo1->next = sinfo2;
  sinfo3->prev = sinfo2;

  sinfo2->prev = sinfo1;
  sinfo2->next = sinfo3;
  sinfo2->parent = sinfo1->parent;
}

void _unlinkSegmentInfo(SegmentInfo_t *sinfo2)
{
  SegmentInfo_t *sinfo1 = sinfo2->prev;
  SegmentInfo_t *sinfo3 = sinfo2->next;

  sinfo1->next = sinfo3;
  sinfo3->prev = sinfo1;

  sinfo2->prev = NULL;
  sinfo2->next = NULL;
  sinfo2->parent = NULL;
}

void _freeSegmentInfo(SegmentInfo_t *sinfo)
{
  CoarrayInfo_t *cinfo;

  forallCoarrayInfo (cinfo, sinfo) {
    _unlinkCoarrayInfo(cinfo);
    _freeCoarrayInfo(cinfo);
  }
  free(sinfo->prev);
  free(sinfo->next);
  //freeDespptr(sinfo->desc);
}


/*****************************************\
  CoarrayInfo_t
\*****************************************/

static CoarrayInfo_t *_mallocCoarrayInfo(int count, size_t element)
{
  CoarrayInfo_t *cinfo =
    (CoarrayInfo_t*)malloc(sizeof(CoarrayInfo_t));

  cinfo->count = count;
  cinfo->element = element;
  cinfo->prev = NULL;
  cinfo->next = NULL;
  return cinfo;
}

void _addCoarrayInfo(SegmentInfo_t *parent, CoarrayInfo_t *cinfo2)
{
  CoarrayInfo_t *cinfo3 = parent->tailCoarray;
  CoarrayInfo_t *cinfo1 = cinfo3->prev;

  cinfo1->next = cinfo2;
  cinfo3->prev = cinfo2;

  cinfo2->prev = cinfo1;
  cinfo2->next = cinfo3;
  cinfo2->parent = cinfo1->parent;
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

static void _freeCoarrayInfo(CoarrayInfo_t *cinfo)
{
  free(cinfo->name);
  free(cinfo->lcobound);
  free(cinfo->ucobound);
  free(cinfo->cosize);
}


/***********************************************\
  free coarray data object
\***********************************************/

static void _freeCoarray(CoarrayInfo_t *cinfo);
static void _freeSegmentObject(SegmentInfo_t *sinfo);
static void _freeByDescriptor(char *desc);


/*  If it is the only coarray of segmentInfo, do
 *  GASNet/FJ-RDMA-free() for the allocated memory.
 *  Else, only unlink the coarrayInfo structure.
 */
void _freeCoarray(CoarrayInfo_t *cinfo)
{
  if (IsOnlyCoarrayInfo(cinfo)) {
    _freeSegmentObject(cinfo->parent);
    return;
  }

  _unlinkCoarrayInfo(cinfo);
  _freeCoarrayInfo(cinfo);
}


void _freeSegmentObject(SegmentInfo_t *sinfo)
{
  CoarrayInfo_t *cinfo;

  forallCoarrayInfo(cinfo, sinfo) {
    _unlinkCoarrayInfo(cinfo);
    _freeCoarrayInfo(cinfo);
  }

#if defined(_XMP_COARRAY_FJRDMA)
  _freeByDescriptor(sinfo->desc);
#elif defined(_XMP_COARRAY_GASNET)
  _XMPF_coarrayDebugPrint("DEALLOCATE in any order is not supported on GASNet.\n");
  if (IsLastSegmentInfo(sinfo)) 
    _freeByDescriptor(sinfo->desc);
#endif
}


void _freeByDescriptor(char *desc)
{
  _XMPF_coarrayDebugPrint("current restriction: "
                          "allocatable coarray cannot be deallocated.\n");
}



/***********************************************\
  ALLOCATE statement
  DEALLOCATE statement
\***********************************************/


/*  unlink & delete coarrayInfo (always),
 *  unlink & delete delete segmentInfo (conditional), and
 *  free memory (conditional)
 */
void xmpf_coarray_dealloc_(void **descPtr)
{
  CoarrayInfo_t *cinfo = (CoarrayInfo_t*)(*descPtr);
  _freeCoarray(cinfo);
}


/*  malloc and make segmentInfo and coarrayInfo
 *  for allocatable coarrays
 */
void xmpf_coarray_malloc_(void **descPtr, char **crayPtr,
                          int *count, int *element, void **tag)
{
  _XMPF_checkIfInTask("allocatable coarray allocation");
  ResourceInfo_t *rinfo;

  // malloc
  SegmentInfo_t *sinfo = _mallocSegment(*count, (size_t)(*element));

  if (*tag != NULL) {
    rinfo = (ResourceInfo_t*)(*tag);
    _linkSegmentInfo(rinfo, sinfo);
  }

  // make coarrayInfo and linkage
  CoarrayInfo_t *cinfo = _mallocCoarrayInfo(*count, (size_t)(*element));
  _addCoarrayInfo(sinfo, cinfo);

  // output #1, #2
  *descPtr = (void*)cinfo;
  *crayPtr = sinfo->orgAddr;
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

SegmentInfo_t *_mallocSegment(int count, size_t element)
{
  SegmentInfo_t *sinfo = _newSegmentInfo();
  size_t elementRU = _roundUpElementSize(count, element);

  // _XMP_coarray_malloc() and set mallocInfo
  sinfo->count = count;
  sinfo->element = elementRU;
  _XMP_coarray_malloc_info_1(sinfo->count, sinfo->element);    // set shape
  _XMP_coarray_malloc_image_info_1();                          // set coshape
  _XMP_coarray_malloc_do(&(sinfo->desc), &(sinfo->orgAddr));   // malloc

  return sinfo;
}



/*****************************************\
  internal structure access functions
  OLD VERSION
\*****************************************/

/***************************************************************
static void _remove_coarrayInfo(ResourceInfo_t *resourceInfo,
                                CoarrayInfo_t *coarrayInfo)
{
  if (resourceInfo->coarrayInfoFirst == coarrayInfo) {
    if (resourceInfo->coarrayInfoLast == coarrayInfo) {
      resourceInfo->coarrayInfoFirst = NULL;
      resourceInfo->coarrayInfoLast = NULL;
    } else {
      resourceInfo->coarrayInfoFirst = coarrayInfo->next;
      coarrayInfo->next->prev = NULL;
    }
  } else {
    if (resourceInfo->coarrayInfoLast == coarrayInfo) {
      resourceInfo->coarrayInfoLast = coarrayInfo->prev;
      coarrayInfo->prev->next = NULL;
    } else {
      coarrayInfo->prev->next = coarrayInfo->next;
      coarrayInfo->next->prev = coarrayInfo->prev;
    }
  }

  coarrayInfo->prev = coarrayInfo->next = NULL;
}


static void _del_coarrayInfo(ResourceInfo_t *resourceInfo,
                             CoarrayInfo_t *coarrayInfo)
{
  _remove_coarrayInfo(resourceInfo, coarrayInfo);
  free(coarrayInfo);
}
*************************************************************/


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



/*****************************************\
  handling memory pool
   for static coarrays
\*****************************************/

void xmpf_coarray_malloc_pool_(void)
{
  _XMPF_coarrayDebugPrint("estimated pool_totalSize = %zd\n",
                          pool_totalSize);

  // malloc
  pool_sinfo = _mallocSegment(1, pool_totalSize);
  pool_currentAddr = pool_sinfo->orgAddr;
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
    _getCoarray(*count, (size_t)(*element));

  cinfo->name = (char*)malloc(sizeof(char)*(*namelen + 1));
  strncpy(cinfo->name, name, *namelen);

  *descPtr = (void*)cinfo;
  *crayPtr = cinfo->baseAddr;
}


CoarrayInfo_t *_getCoarray(int count, size_t element)
{
  _XMPF_checkIfInTask("static coarray allocation");

  size_t elementRU = _roundUpElementSize(count, element);

  // allocate and set _coarrayInfo
  CoarrayInfo_t *cinfo = _mallocCoarrayInfo(count, elementRU);
  _addCoarrayInfo(pool_sinfo, cinfo);

  // check: too large allocation
  size_t thisSize = (size_t)count * elementRU;

  if (pool_currentAddr + thisSize > pool_sinfo->orgAddr + pool_totalSize) {
    _XMPF_coarrayFatal("lack of memory pool for static coarrays: "
                      "xmpf_coarray_share_pool_() in %s", __FILE__);
  }

  _XMPF_coarrayDebugPrint("Coarray %s gets share of memory:\n"
                          "  address = %p to %p\n"
                          "  size    = %zd\n",
                          cinfo->name, pool_currentAddr, pool_currentAddr+thisSize,
                          thisSize);

  cinfo->baseAddr = pool_currentAddr;
  cinfo->endAddr = pool_currentAddr += thisSize;

  return cinfo;
}



/************\
   entry
\************/

void xmpf_coarray_count_size_(int *count, int *element)
{
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);

  _XMPF_coarrayDebugPrint("count-up allocation size: %zd[byte].\n", mallocSize);

  pool_totalSize += mallocSize;
}


void xmpf_coarray_proc_init_(void **tag)
{
  ResourceInfo_t *resource;

  resource = _newResourceInfo();
  *tag = (void*)resource;
}


void xmpf_coarray_proc_finalize_(void **tag)
{
  if (*tag == NULL)
    return;

  ResourceInfo_t *rinfo = (ResourceInfo_t*)(*tag);
  _freeResourceInfo(rinfo);

  *tag = NULL;
}


/*
 * find descriptor corresponding to baseAddr
 *   This function is used at the entrance of a user procedure to find
 *   the descriptors of the dummy arguments.
 */
/************************
void xmpf_coarray_descptr_(void **descPtr, char *baseAddr, void **tag)
{
  ResourceInfo_t *rinfo = (ResourceInfo_t*)(*tag);
  SegmentInfo_t *sinfo, *sinfo_found;
  CoarrayInfo_t *cinfo, *cinfo_found;

  sinfo_found = NULL;
  forallSegmentInfo (sinfo, rinfo) {
    if (sinfo->orgAddr <= baseAddr && baseAddr < sinfo->endAddr) {
      // found segment of the coarray
      _XMPF_coarrayDebugPrint("found a segment (%p) for the coarray (%p).\n", 
                              sinfo->orgAddr, baseAddr);
      sinfo_found = sinfo;
      break;
    }
  }

  if (sinfo_found != NULL) {
    _XMPF_coarrayFatal("INTERNAL: could not find serno for a dummy coarray, "
                       "baseAddr=%p\n", baseAddr);
  }
  
  cinfo_found = NULL;
  forallCoarrayInfo (cinfo, sinfo) {
    if (cinfo->baseAddr <= baseAddr && baseAddr < cinfo->endAddr) {
      // found the coarray is allocated
      cinfo_found = cinfo;
      break;
    }
  }

  if (cinfo_found) {
    _XMPF_coarrayDebugPrint("found the super-coarray (%p) of the coarray (%p).\n", 
                            cinfo->baseAddr, baseAddr);
      
  CoarrayInfo_t *cp = (CoarrayInfo_t*)malloc(sizeof(CoarrayInfo_t));
  cp->serno = serno;
  cp->baseAddr = baseAddr;

  _add_coarrayInfo(rinfo, cp);

  *descPtr = (char*)cp;
}
*************************************************/

/*
 * find descriptor-ID corresponding to baseAddr
 *   This routine is used for dummy arguments currently.
 */
/*********************************************************************
int xmpf_get_descr_id_(char *baseAddr)
{
  int i;
  SegmentInfo_t *cp;

  for (i = 0, cp = SegmentInfoTab; i < DESCR_ID_MAX; i++, cp++) {
    if (cp->is_used) {
      if (cp->orgAddr <= baseAddr && baseAddr < cp->orgAddr + cp->size)
        // found its descriptor-ID
          _XMPF_coarrayDebugPrint("found descriptor-ID #%d for baseAddr=%p\n",
                                  i, baseAddr);
        return i;
    }
  }

  _XMPF_coarrayFatal("cannot access unallocated coarray");
  return -1;
}
**********************************************************************/



/*****************************************\
   management of dynamic attribute:
     current coshapes
\*****************************************/

/*
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


/*****************************************
  sub OLD

int _setSegmentInfo(char *desc, char *orgAddr, size_t size)
{
  int serno;

  serno = _getNewSerno();
  if (serno < 0) {         // fail 
    _XMPF_coarrayFatal("xmpf_coarray.c: no more desc table.");
    return serno;
  }

  SegmentInfoTab[serno].is_used = TRUE;
  SegmentInfoTab[serno].desc = desc;
  SegmentInfoTab[serno].orgAddr = orgAddr;
  //SegmentInfoTab[serno].count = count;
  //SegmentInfoTab[serno].element = element;
  SegmentInfoTab[serno].size = size;

  return serno;
}


int _getNewSerno() {
  int i;

  // try white area
  for (i = _nextId; i < DESCR_ID_MAX; i++) {
    if (! SegmentInfoTab[i].is_used) {
      ++_nextId;
      return i;
    }
  }

  // try reuse
  for (i = 0; i < _nextId; i++) {
    if (! SegmentInfoTab[i].is_used) {
      _nextId = i + 1;
      return i;
    }
  }

  // error: no room 
  return -1;
}
*****************************************/


/*****************************************      \
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


