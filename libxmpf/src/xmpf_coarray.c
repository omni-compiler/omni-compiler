#include "xmpf_internal.h"
extern int num_images_(void);
static int _getNewSerno();
static int _set_coarrayInfo(char *desc, char *orgAddr, size_t size);
static void _coarray_msg(int sw);

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)

/*****************************************\
  runtime environment
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off
int _XMPF_coarrayErr = 0;          // default: aggressive error check off

void _XMPF_coarray_init(void)
{
  char *str;

  if (xmp_node_num() == 1) {
    str = getenv("XMPF_COARRAY_MSG1");
    if (str != NULL) {
      _coarray_msg(atoi(str));
      return;
    }
  }

  str = getenv("XMPF_COARRAY_MSG");
  if (str != NULL) {
    _coarray_msg(atoi(str));
  }
}

/*
 *  hidden subroutine interface,
 *   which can be used in the user program
 */
void xmpf_coarray_msg_(int *sw)
{
  _coarray_msg(*sw);
}

void _coarray_msg(int sw)
{
  switch (sw) {
  case 0:
  default:
    if (_XMPF_coarrayMsg)
      _XMPF_coarrayDebugPrint("xmpf_coarray_msg OFF\n");
    _XMPF_coarrayMsg = 0;
    return;

  case 1:
    _XMPF_coarrayMsg = 1;
    break;
  }

  _XMPF_coarrayDebugPrint("xmpf_coarray_msg ON\n"
                          "  %zd-byte boundary, using %s\n",
                          BOUNDARY_BYTE,
#if defined(_XMP_COARRAY_FJRDMA)
                          "FJRDMA"
#elif defined(_XMP_COARRAY_GASNET)
                          "GASNET"
#else
                          "something unknown"
#endif
                          );
}


/*****************************************\
  internal information management
\*****************************************/

typedef struct {
  BOOL    is_used;
  char   *desc;
  char   *orgAddr;
  size_t  size;
  int     corank;
  int    *lcobound;
  int    *ucobound;
  int    *cosize;
} _coarrayInfo_t;

typedef struct {
  BOOL    is_used;
  char   *desc;
  char   *orgAddr;
  int     count;
  size_t  element;
} _coarrayInfo_t_V1;

static _coarrayInfo_t _coarrayInfoTab[DESCR_ID_MAX] = {};
static int _nextId = 0;


/*
 * find descriptor-ID corresponding to baseAddr
 *   This routine is used for dummy arguments currently.
 */
int xmpf_get_descr_id_(char *baseAddr)
{
  int i;
  _coarrayInfo_t *cp;

  for (i = 0, cp = _coarrayInfoTab;
       i < DESCR_ID_MAX;
       i++, cp++) {
    if (cp->is_used) {
      if (cp->orgAddr <= baseAddr &&
          //baseAddr < cp->orgAddr + cp->count * cp->element)   V1
          baseAddr < cp->orgAddr + cp->size)
        return i;
    }
  }

  _XMP_fatal("cannot access unallocated coarray");
  return -1;
}


/*
 * set the current lower and upper cobounds
 *
 */
void xmpf_coarray_set_coshape_(int *serno, int *corank, ...)
{
  int i, n, count, n_images;
  va_list args;
  va_start(args, corank);

  _coarrayInfo_t *cp = &_coarrayInfoTab[*serno];
  n = cp->corank = *corank;
  cp->lcobound = (int*)malloc(sizeof(int) * n);
  cp->ucobound = (int*)malloc(sizeof(int) * n);
  cp->cosize = (int*)malloc(sizeof(int) * n);

  for (count = 1, i = 0; i < n-1; i++) {
    cp->lcobound[i] = *va_arg(args, int*);
    cp->ucobound[i] = *va_arg(args, int*);
    cp->cosize[i] = cp->ucobound[i] - cp->lcobound[i] + 1;
    if (cp->cosize[i] <= 0)
      _XMP_fatal("found illegal lower and upper cobounds of a coarray");
    count *= cp->cosize[i];
  }

  n_images = num_images_();
  cp->lcobound[n-1] = *va_arg(args, int*);
  cp->cosize[n-1] = DIV_CEILING(n_images, count);
  cp->ucobound[n-1] = cp->lcobound[n-1] + cp->cosize[n-1] - 1;


  //////////////
  for (i = 0; i < n; i++) {
    fprintf(stdout, "  cp->lcobound[%d]  = %d\n", i, cp->lcobound[i]);
    fprintf(stdout, "  cp->ucobound[%d]  = %d\n", i, cp->ucobound[i]);
    fprintf(stdout, "  cp->cosize[%d]    = %d\n", i, cp->cosize[i]);
  }
  //////////////

}



//int _XMPF_get_coarrayElement(int serno)
//{
//  return _coarrayInfoTab[serno].element;
//}

char *_XMPF_get_coarrayDesc(int serno)
{
  return _coarrayInfoTab[serno].desc;
}

size_t _XMPF_get_coarrayOffset(int serno, char *baseAddr)
{
  char* orgAddr = _coarrayInfoTab[serno].orgAddr;
  int offset = ((size_t)baseAddr - (size_t)orgAddr);
  return offset;
}

/* disuse
 */
//int _XMPF_get_coarrayStart(int serno, char *baseAddr)
//{
//  int element = _coarrayInfoTab[serno].element;
//  char* orgAddr = _coarrayInfoTab[serno].orgAddr;
//  int start = ((size_t)baseAddr - (size_t)orgAddr) / element;
//  return start;
//}


//int _set_coarrayInfo(char *desc, char *orgAddr, int count, size_t element)
int _set_coarrayInfo(char *desc, char *orgAddr, size_t size)
{
  int serno;

  serno = _getNewSerno();
  if (serno < 0) {         /* fail */
    _XMP_fatal("xmpf_coarray.c: no more desc table.");
    return serno;
  }

  _coarrayInfoTab[serno].is_used = TRUE;
  _coarrayInfoTab[serno].desc = desc;
  _coarrayInfoTab[serno].orgAddr = orgAddr;
  //_coarrayInfoTab[serno].count = count;
  //_coarrayInfoTab[serno].element = element;
  _coarrayInfoTab[serno].size = size;

  return serno;
}

static int _getNewSerno() {
  int i;

  /* try white area */
  for (i = _nextId; i < DESCR_ID_MAX; i++) {
    if (! _coarrayInfoTab[i].is_used) {
      ++_nextId;
      return i;
    }
  }

  /* try reuse */
  for (i = 0; i < _nextId; i++) {
    if (! _coarrayInfoTab[i].is_used) {
      _nextId = i + 1;
      return i;
    }
  }

  /* error: no room */
  return -1;
}


/*****************************************\
  memory allocation for static coarrays
\*****************************************/

static size_t pool_totalSize = 0;
static const size_t pool_maxSize = SIZE_MAX;
static void *pool_rootDesc;
static char *pool_rootAddr, *pool_ptr;
static int pool_serno;

void xmpf_coarray_count_size_(int *count, int *element)
{
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);
  size_t lastTotalSize = pool_totalSize;

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("count allocation size of a static coarray: "
                            "%zd[byte].\n", mallocSize);
  }

  pool_totalSize += mallocSize;

  // error check
  if (pool_totalSize > pool_maxSize ||
      pool_totalSize < lastTotalSize) {
    _XMP_fatal("Static coarrays require too much memory in total.");
  }
}


void xmpf_coarray_memorypool_(void)
{
  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("estimated pool_totalSize = %zd\n",
                            pool_totalSize);
  }

  _XMP_coarray_malloc_info_1(pool_totalSize, 1);
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&pool_rootDesc, &pool_rootAddr);

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("allocate a pool for static coarrays\n"
                            "  pool_rootDesc  = %p\n"
                            "  pool_rootAddr  = %p\n"
                            "  pool_totalSize = %zd [byte]\n",
                            pool_rootDesc, pool_rootAddr, pool_totalSize);
  }


  pool_serno = _set_coarrayInfo(pool_rootDesc, pool_rootAddr, pool_totalSize);

  pool_ptr = pool_rootAddr;
}


void xmpf_coarray_share_(int *serno, char **pointer,
                         int *count, int *element)
{
  _XMPF_checkIfInTask("allocatable coarray allocation");

  // error check: boundary check
  if ((*count) != 1 && (*element) % BOUNDARY_BYTE != 0) {
    /* restriction: the size must be a multiple of BOUNDARY_BYTE
       unless it is a scalar.
    */
    _XMP_fatal("violation of static coarray allocation boundary");
  }

  // get memory
  size_t thisSize = (size_t)(*count) * (size_t)(*element);
  size_t mallocSize = ROUND_UP_UNIT(thisSize);

  if (pool_ptr + mallocSize > pool_rootAddr + pool_totalSize) {
    _XMP_fatal("lack of memory pool for static coarrays: "
               "xmpf_coarray_share_() in " __FILE__);
  }

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("get memory in the pool\n"
                            "  address = %p\n"
                            "  size    = %zd\n",
                            pool_ptr, mallocSize);
  }

  *serno = pool_serno;
  *pointer = pool_ptr;
  pool_ptr += mallocSize;
}


/*****************************************\
  memory allocation for allocatable coarrays
\*****************************************/

void xmpf_coarray_malloc_(int *serno, char **pointer, int *count, int *element)
{
  void *desc;
  void *orgAddr;
  size_t elementRU;

  _XMPF_checkIfInTask("allocatable coarray allocation");

  // boundary check and recovery
  if ((*element) % BOUNDARY_BYTE == 0) {
    elementRU = (size_t)(*element);
  } else if (*count == 1) {              // scalar or one-element array
    /* round up */
    elementRU = (size_t)ROUND_UP_BOUNDARY(*element);
  } else {
    /* restriction */
    _XMP_fatal("violation of boundary: "
               "xmpf_coarray_malloc_(), " __FILE__);
    return;
  }

  // set (see libxmp/src/xmp_coarray_set.c)
  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("COARRAY ALLOCATION\n"
                            "  *count=%d, elementRU=%zd, *element=%d\n",
                            *count, elementRU, *element);
  }
  //_XMP_coarray_malloc_info_1(*count, elementRU);
  _XMP_coarray_malloc_info_1((*count)*elementRU, 1);
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&desc, &orgAddr);

  *pointer = orgAddr;
  //*serno = _set_coarrayInfo(desc, orgAddr, *count, *element);
  *serno = _set_coarrayInfo(desc, orgAddr, (*count)*(*element));
}


/*****************************************\
  restriction checker
\*****************************************/

int _XMPF_nowInTask()
{
  return xmp_num_nodes() < xmp_all_num_nodes();
}

void _XMPF_checkIfInTask(char *msgopt)
{
  if (_XMPF_nowInTask()) {
    char work[200];
    sprintf(work, "current rextriction: cannot use %s in any task construct",
            msgopt);
    _XMP_fatal(work);
  }
}


void _XMPF_coarrayDebugPrint(char *format, ...)
{
  char work[1000];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  fprintf(stderr, "CAF[%d] %s", xmp_node_num(), work);
  va_end(list);
}

