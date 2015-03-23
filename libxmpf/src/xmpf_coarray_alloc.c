#include "xmpf_internal.h"

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

static int _set_coarrayMallocInfo(char *desc, char *orgAddr, size_t size);
static int _getNewSerno(void);


/*****************************************\
  internal representation
\*****************************************/

/*
 * structure for each malloc unit
 */
typedef struct _coarrayMallocInfo_t _coarrayMallocInfo_t;
struct _coarrayMallocInfo_t {
  BOOL    is_used;       // if this is used in the table
  char   *desc;          // address of the lower layer's descriptor 
  char   *orgAddr;       // local address of the allocated memory
  size_t  size;          // size of the allocated memory
  //  int     corank;        // number of codimensions
  //  int    *lcobound;      // array of lower cobounds [0..(corank-1)]
  //  int    *ucobound;      // array of upper cobounds [0..(corank-1)]
  //  int    *cosize;        // cosize[k] = max(ucobound[k]-lcobound[k]+1, 0)
};

static _coarrayMallocInfo_t _coarrayMallocInfoTab[DESCR_ID_MAX] = {};
static int _nextId = 0;

static size_t pool_totalSize = 0;
static const size_t pool_maxSize = SIZE_MAX;
static void *pool_rootDesc;
static char *pool_rootAddr, *pool_currentAddr;
static int pool_serno;


/*
 * structure for each coarray variable, incl. dummy variable
 */
typedef struct _coarrayInfo_t _coarrayInfo_t;
struct _coarrayInfo_t {
  _coarrayInfo_t *prev;
  _coarrayInfo_t *next;
  char   *name;          // name of the variable (for debug message)
  int     serno;         // index of table _coarrayMallocInfoTab[]
  char   *baseAddr;      // local address of the coarray
  int     corank;        // number of codimensions
  int    *lcobound;      // array of lower cobounds [0..(corank-1)]
  int    *ucobound;      // array of upper cobounds [0..(corank-1)]
  int    *cosize;        // cosize[k] = max(ucobound[k]-lcobound[k]+1, 0)
};

static void _freeCoarray(_coarrayInfo_t *coarrayInfo);

char *_XMPF_get_coarrayDesc(char *descPtr)
{
  _coarrayInfo_t *cp = (_coarrayInfo_t*)descPtr;
  return _coarrayMallocInfoTab[cp->serno].desc;
}

size_t _XMPF_get_coarrayOffset(char *descPtr, char *baseAddr)
{
  _coarrayInfo_t *cp = (_coarrayInfo_t*)descPtr;
  char* orgAddr = _coarrayMallocInfoTab[cp->serno].orgAddr;
  int offset = ((size_t)baseAddr - (size_t)orgAddr);
  return offset;
}


typedef struct {
  _coarrayInfo_t *coarrayInfoFirst;
  _coarrayInfo_t *coarrayInfoLast;
} _resourceInfo_t;


static void _add_coarrayInfo(_resourceInfo_t *resourceInfo,
                             _coarrayInfo_t *coarrayInfo)
{
  if (resourceInfo->coarrayInfoFirst == NULL) {
    coarrayInfo->prev = NULL;
    coarrayInfo->next = NULL;
    resourceInfo->coarrayInfoFirst = coarrayInfo;
    resourceInfo->coarrayInfoLast = coarrayInfo;
  } else {
    coarrayInfo->prev = resourceInfo->coarrayInfoLast;
    coarrayInfo->next = NULL;
    resourceInfo->coarrayInfoLast->next = coarrayInfo;
    resourceInfo->coarrayInfoLast = coarrayInfo;
  }
}


static void _remove_coarrayInfo(_resourceInfo_t *resourceInfo,
                                _coarrayInfo_t *coarrayInfo)
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


static void _del_coarrayInfo(_resourceInfo_t *resourceInfo,
                             _coarrayInfo_t *coarrayInfo)
{
  _remove_coarrayInfo(resourceInfo, coarrayInfo);
  free(coarrayInfo);
}



/*****************************************\
  handling memory pool
   for static coarrays
\*****************************************/

/*
 * have a share of memory in the pool
 *    out: descPtr: pointer to descriptor _coarrayInfo_t
 *         crayPtr: cray pointer to the coarray object
 *    in:  count  : count of elements
 *         element: element size
 *         name   : name of the coarray (for debugging)
 *         namelen: character length of name
 */
void xmpf_coarray_share_pool_(char **descPtr, char **crayPtr,
                              int *count, int *element,
                              char *name, int *namelen)
{
  _XMPF_checkIfInTask("static coarray allocation");

  // error check: boundary check
  if ((*count) != 1 && (*element) % BOUNDARY_BYTE != 0) {
    /* restriction: the size must be a multiple of BOUNDARY_BYTE
       unless it is a scalar.
    */
    _XMPF_coarrayFatal("violation of static coarray allocation boundary");
  }

  // allocate and set _coarrayInfo
  _coarrayInfo_t *cp =
    (_coarrayInfo_t*)malloc(sizeof(_coarrayInfo_t));
  cp->serno = pool_serno;
  cp->name = (char*)malloc(sizeof(char)*(*namelen + 1));
  strncpy(cp->name, name, *namelen);

  // get memory share
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);

  if (pool_currentAddr + mallocSize > pool_rootAddr + pool_totalSize) {
    _XMPF_coarrayFatal("lack of memory pool for static coarrays: "
                      "xmpf_coarray_share_pool_() in %s", __FILE__);
  }

  _XMPF_coarrayDebugPrint("Coarray %s gets memory in the pool:\n"
                          "  address = %p to %p\n"
                          "  size    = %zd\n",
                          cp->name, pool_currentAddr, pool_currentAddr+mallocSize,
                          mallocSize);

  // output #1
  *descPtr = (char*)cp;

  // output #2
  *crayPtr = pool_currentAddr;
  pool_currentAddr += mallocSize;
}


void xmpf_coarray_count_size_(int *count, int *element)
{
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);
  size_t lastTotalSize = pool_totalSize;

  _XMPF_coarrayDebugPrint("count-up allocation size: "
                          "%zd[byte].\n", mallocSize);

  pool_totalSize += mallocSize;

  // error check
  if (pool_totalSize > pool_maxSize ||
      pool_totalSize < lastTotalSize) {
    _XMPF_coarrayFatal("Static coarrays require too much memory in total.");
  }
}


void xmpf_coarray_malloc_pool_(void)
{
  _XMPF_coarrayDebugPrint("estimated pool_totalSize = %zd\n",
                          pool_totalSize);

  _XMP_coarray_malloc_info_1(pool_totalSize, 1);
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&pool_rootDesc, &pool_rootAddr);

  _XMPF_coarrayDebugPrint("allocate a memory pool:\n"
                          "  pool_rootDesc  = %p\n"
                          "  pool_rootAddr  = %p\n"
                          "  pool_totalSize = %zd [byte]\n",
                          pool_rootDesc, pool_rootAddr, pool_totalSize);

  pool_serno = _set_coarrayMallocInfo(pool_rootDesc, pool_rootAddr,
                                      pool_totalSize);

  pool_currentAddr = pool_rootAddr;
}


/***********************************************\
  allocate memory for an allocatable coarray
\***********************************************/

void xmpf_coarray_dealloc_(char **descPtr, void **tag)
{
  _coarrayInfo_t *cp = (_coarrayInfo_t*)(*descPtr);
  _resourceInfo_t *resource;

  _freeCoarray(cp);
  _del_coarrayInfo(resource, cp);

  if (*tag != NULL) {
    resource = (_resourceInfo_t*)(*tag);
    _add_coarrayInfo(resource, cp);
  }
}


void xmpf_coarray_malloc_(char **descPtr, char **crayPtr,
                          int *count, int *element, void **tag)
{
  void *desc;
  void *orgAddr;
  size_t elementRU;
  _resourceInfo_t *resource;

  _XMPF_checkIfInTask("allocatable coarray allocation");

  _coarrayInfo_t *cp = 
    (_coarrayInfo_t*)malloc(sizeof(_coarrayInfo_t));

  if (*tag != NULL) {
    resource = (_resourceInfo_t*)(*tag);
    _add_coarrayInfo(resource, cp);
  }

  // boundary check and recovery
  if ((*element) % BOUNDARY_BYTE == 0) {
    elementRU = (size_t)(*element);
  } else if (*count == 1) {              // scalar or one-element array
    /* round up */
    elementRU = (size_t)ROUND_UP_BOUNDARY(*element);
  } else {
    /* restriction */
    _XMPF_coarrayFatal("violation of boundary: xmpf_coarray_malloc_() in %s",
                       __FILE__);
    return;
  }

  // set (see libxmp/src/xmp_coarray_set.c)
  _XMPF_coarrayDebugPrint("COARRAY ALLOCATION\n"
                          "  *count=%d, elementRU=%zd, *element=%d\n",
                          *count, elementRU, *element);

  //_XMP_coarray_malloc_info_1(*count, elementRU);
  _XMP_coarray_malloc_info_1((*count)*elementRU, 1);
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&desc, &orgAddr);

  // output #2
  *crayPtr = (char*)orgAddr;

  // output #1
  cp->serno = _set_coarrayMallocInfo((char*)desc, (char*)orgAddr,
                                     (*count)*(*element));
  *descPtr = (char*)cp;
}



void _freeCoarray(_coarrayInfo_t *coarrayInfo)
{
  int serno = coarrayInfo->serno;
  char *desc = _coarrayMallocInfoTab[serno].desc;

  ////////////////////////////
  _XMPF_coarrayDebugPrint("freeCoarray() is not implemented\n");
  //_XMP_coarray_free_do((void*)desc);
  ////////////////////////////

}


/************\
   entry
\************/

void xmpf_coarray_proc_init_(void **tag)
{
  _resourceInfo_t *resource;

  resource = (_resourceInfo_t*)malloc(sizeof(_resourceInfo_t));
  resource->coarrayInfoFirst = NULL;
  resource->coarrayInfoLast = NULL;
  _XMPF_coarrayDebugPrint("malloc and initialize resourceInfo: %p\n", resource);
  *tag = (void*)resource;
}


void xmpf_coarray_proc_finalize_(void **tag)
{
  _coarrayInfo_t *cp, *cp_next;

  if (*tag == NULL)
    return;

  _resourceInfo_t *resource = (_resourceInfo_t*)(*tag);
  _XMPF_coarrayDebugPrint("decode resourceInfo for finalize: %p\n", resource); 
  cp = resource->coarrayInfoFirst;
  while (cp != NULL) {
    cp_next = cp->next;
    _freeCoarray(cp);
    _del_coarrayInfo(resource, cp);
    cp = cp_next;
  }
  free(resource);
  *tag = NULL;
}


/*
 * find descriptor corresponding to baseAddr
 *   This function is used at the entrance of a user procedure to find
 *   the descriptors of the dummy arguments.
 */
void xmpf_coarray_descptr_(char **descPtr, char *baseAddr, void **tag)
{
  _resourceInfo_t *resource = (_resourceInfo_t*)(*tag);
  _XMPF_coarrayDebugPrint("decode resourceInfo to set descptr: %p\n", resource); 

  int serno = -1;

  for (int i = 0; i < _nextId; i++) {
    char *orgAddr = _coarrayMallocInfoTab[i].orgAddr;
    size_t size =  _coarrayMallocInfoTab[i].size;
    if (orgAddr <= baseAddr && baseAddr < orgAddr + size) {
      // found serno
      _XMPF_coarrayDebugPrint("found memory #%d for a dummy coarray, "
                              "baseAddr=%p\n", i, baseAddr);
      serno = i;
      break;
    }
  }

  // if serno is (-1), the coarray is not allocated yet.
  if (serno < 0) {
    _XMPF_coarrayFatal("INTERNAL: could not find serno for a dummy coarray, "
                       "baseAddr=%p\n", baseAddr);
  }
  
  _coarrayInfo_t *cp = (_coarrayInfo_t*)malloc(sizeof(_coarrayInfo_t));
  cp->serno = serno;
  cp->baseAddr = baseAddr;

  _add_coarrayInfo(resource, cp);

  *descPtr = (char*)cp;
}


/*
 * find descriptor-ID corresponding to baseAddr
 *   This routine is used for dummy arguments currently.
 */
/*********************************************************************
int xmpf_get_descr_id_(char *baseAddr)
{
  int i;
  _coarrayMallocInfo_t *cp;

  for (i = 0, cp = _coarrayMallocInfoTab; i < DESCR_ID_MAX; i++, cp++) {
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
void xmpf_coarray_set_coshape_(char **descPtr, int *corank, ...)
{
  int i, n, count, n_images;
  _coarrayInfo_t *cp = (_coarrayInfo_t*)(*descPtr);

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


/*****************************************\
  sub
\*****************************************/

int _set_coarrayMallocInfo(char *desc, char *orgAddr, size_t size)
{
  int serno;

  serno = _getNewSerno();
  if (serno < 0) {         /* fail */
    _XMPF_coarrayFatal("xmpf_coarray.c: no more desc table.");
    return serno;
  }

  _coarrayMallocInfoTab[serno].is_used = TRUE;
  _coarrayMallocInfoTab[serno].desc = desc;
  _coarrayMallocInfoTab[serno].orgAddr = orgAddr;
  //_coarrayMallocInfoTab[serno].count = count;
  //_coarrayMallocInfoTab[serno].element = element;
  _coarrayMallocInfoTab[serno].size = size;

  return serno;
}


int _getNewSerno() {
  int i;

  /* try white area */
  for (i = _nextId; i < DESCR_ID_MAX; i++) {
    if (! _coarrayMallocInfoTab[i].is_used) {
      ++_nextId;
      return i;
    }
  }

  /* try reuse */
  for (i = 0; i < _nextId; i++) {
    if (! _coarrayMallocInfoTab[i].is_used) {
      _nextId = i + 1;
      return i;
    }
  }

  /* error: no room */
  return -1;
}


/*****************************************      \
  intrinsic functions
\*****************************************/

/*
 * get an image index corresponding to the current lower and upper cobounds
 */
int xmpf_coarray_get_image_index_(char **descPtr, int *corank, ...)
{
  int i, idx, lb, ub, factor, count;
  va_list(args);
  va_start(args, corank);

  _coarrayInfo_t *cp = (_coarrayInfo_t*)(*descPtr);

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


