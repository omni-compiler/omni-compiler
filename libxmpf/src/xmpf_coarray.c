#include "xmpf_internal.h"

static int _getNewSerno();
static int _set_coarrayInfo(char *desc, char *orgAddr, int count, size_t element);


/*****************************************\
  internal switches
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off

void xmpf_coarray_msg_(int *sw)
{
  _XMPF_coarrayMsg = *sw;

  if (_XMPF_coarrayMsg) {
    _XMPF_coarrayMsgPrefix();
    fprintf(stderr, "xmpf_coarray_msg ON\n");
#ifdef _XMP_COARRAY_FJRDMA
    fprintf(stderr, "  _XMP_COARRAY_FJRDMA defined\n");
#else
    fprintf(stderr, "  _XMP_COARRAY_FJRDMA not defined\n");
#endif
    fprintf(stderr, "  BOUNDARY_BYTE = %d\n", BOUNDARY_BYTE);
  }
}


/*****************************************\
  internal information management
\*****************************************/

typedef struct {
  BOOL    is_used;
  char   *desc;
  char   *orgAddr;
  int     count;
  size_t  element;
} _coarrayInfo_t;

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
          baseAddr < cp->orgAddr + cp->count * cp->element)
        return i;
    }
  }

  _XMP_fatal("cannot access unallocated coarray");
  return -1;
}


int _XMPF_get_coarrayElement(int serno)
{
  return _coarrayInfoTab[serno].element;
}

char *_XMPF_get_coarrayDesc(int serno)
{
  return _coarrayInfoTab[serno].desc;
}

int _XMPF_get_coarrayStart(int serno, char *baseAddr)
{
  int element = _coarrayInfoTab[serno].element;
  char* orgAddr = _coarrayInfoTab[serno].orgAddr;
  int start = ((size_t)baseAddr - (size_t)orgAddr) / element;
  return start;
}


int _set_coarrayInfo(char *desc, char *orgAddr, int count, size_t element)
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
  _coarrayInfoTab[serno].count = count;
  _coarrayInfoTab[serno].element = element;

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
  coarray allocation
\*****************************************/

void xmpf_coarray_malloc_(int *serno, char **pointer, int *count, int *element)
{
  void *desc;
  void *orgAddr;
  size_t elementRU;

  _XMPF_checkIfInTask("coarray allocation");

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
    _XMPF_coarrayMsgPrefix();
    fprintf(stderr, "COARRAY ALLOCATION\n");
    fprintf(stderr, "  *count=%d, elementRU=%zd, *element=%d\n",
            *count, elementRU, *element);
  }
  _XMP_coarray_malloc_info_1(*count, elementRU);
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&desc, &orgAddr);

  *pointer = orgAddr;
  *serno = _set_coarrayInfo(desc, orgAddr, *count, *element);
}


/*****************************************\
  sync all
\*****************************************/

void xmpf_sync_all_nostat_(void)
{
  _XMPF_checkIfInTask("sync all");

  int status;
  xmp_sync_all(&status);
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


void _XMPF_coarrayMsgPrefix(void)
{
  fprintf(stderr, "coarray[%d] ", xmp_num_nodes());
}

