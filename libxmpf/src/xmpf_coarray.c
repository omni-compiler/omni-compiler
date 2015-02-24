#include "xmpf_internal.h"

static int _getNewSerno();
static int _set_coarrayInfo(char *desc, char *orgAddr, size_t size);

static void _coarray_msg(int sw);


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
  intrinsic procedures
  through the wrappers in xmpf_coarray_wrap.f90
\*****************************************/

/*  MPI_Comm_size() of the current communicator
 */
int num_images_(void)
{
  _XMPF_checkIfInTask("num_images()");
  return xmp_num_nodes();
}

/*  (MPI_Comm_rank() + 1) in the current communicator
 */
int this_image_(void)
{
  _XMPF_checkIfInTask("this_image()");
  return xmp_node_num();
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


void xmpf_coarray_malloc_share_(void)
{
  ///////////////////////
  // TEMPORARY
  pool_totalSize = 100000;
  ///////////////////////

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


void xmpf_coarray_get_share_(int *serno, char **pointer,
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
  sync all
\*****************************************/

void xmpf_sync_all_nostat_(void)
{
  static unsigned int id = 0;

  _XMPF_checkIfInTask("sync all");

  id += 1;

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("SYNC ALL in (id=%d)\n", id);
  }

  int status;
  xmp_sync_all(&status);

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayDebugPrint("SYNC ALL out (id=%d)\n", id);
  }
}

void xmpf_sync_all_stat_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("sync all");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC ALL statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  int status;
  xmp_sync_all(&status);
}


/*****************************************\
  sync memory
\*****************************************/

void xmpf_sync_memory_nostat_(void)
{
  _XMPF_checkIfInTask("sync memory");

  int status;
  xmp_sync_memory(&status);
}

void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen)
{
  _XMPF_checkIfInTask("sync memory");

  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC MEMORY statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  int status;
  xmp_sync_memory(&status);
}


/*****************************************\
  sync images
\*****************************************/

void xmpf_sync_image_nostat_(int *image)
{
  int status;
  xmp_sync_image(*image, &status);
}

void xmpf_sync_image_stat_(int *image, int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (<image>) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_image(*image, &status);
}


void xmpf_sync_images_nostat_(int *images, int *size)
{
  int status;
  xmp_sync_images(*size, images, &status);
}

void xmpf_sync_images_stat_(int *images, int *size, int *stat,
                            char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (<images>) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_images(*size, images, &status);
}


void xmpf_sync_allimages_nostat_(void)
{
  int status;
  xmp_sync_images_all(&status);
}

void xmpf_sync_allimages_stat_(int *stat, char *msg, int *msglen)
{
  static BOOL firstCall = TRUE;
  if (firstCall) {
    firstCall = FALSE;
    fprintf(stderr, "not supported yet: "
            "stat= specifier in SYNC IMAGES (*) statement\n");
    fprintf(stderr, "  -- ignored.\n");
  }

  _XMPF_checkIfInTask("sync image");

  int status;
  xmp_sync_images_all(&status);
}



/*****************************************\
  error message to reply to Fortran (temporary)
  (not used yet)
\*****************************************/

char *_XMPF_errmsg = NULL;

void xmpf_get_errmsg_(unsigned char *errmsg, int *msglen)
{
  int i, len;

  if (_XMPF_errmsg == NULL) {
    len = 0;
  } else {
    len = strlen(_XMPF_errmsg);
    if (len > *msglen)
      len = *msglen;
    memcpy(errmsg, _XMPF_errmsg, len);      // '\n' is not needed
  }

  for (i = len; i < *msglen; )
    errmsg[i++] = ' ';

  return;
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

