#include "xmpf_internal.h"

#define NotSupportedError 1

static int _getNewSerno();
static int _set_coarrayInfo(char *desc, char *orgAddr, int count, size_t element);


/*****************************************\
  internal switches
\*****************************************/

int _XMPF_coarrayMsg = 0;          // default: message off

void xmpf_coarray_msg_(int *sw)
{
  _XMPF_coarrayMsg = *sw;
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

int num_images_(void)
{
  return xmp_num_nodes();
}

int this_image_(void)
{
  return xmp_node_num();
}



/*****************************************\
  coarray allocation
\*****************************************/

void xmpf_coarray_malloc_(int *serno, char **pointer, int *count, int *element)
{
  _XMPF_coarray_malloc(serno, pointer, *count, (size_t)(*element));
}

void _XMPF_coarray_malloc(int *serno, char **pointer, int count, size_t element)
{
  void *desc;
  void *orgAddr;

  // see libxmp/src/xmp_coarray_set.c
  _XMP_coarray_malloc_info_1(count, element);  
  _XMP_coarray_malloc_image_info_1();
  _XMP_coarray_malloc_do(&desc, &orgAddr);

  *pointer = orgAddr;
  *serno = _set_coarrayInfo(desc, orgAddr, count, element);
}


/*****************************************\
  synchronizations
\*****************************************/

void xmpf_sync_all_0_(void)
{
  int dummy;
  xmpf_sync_all_1_(&dummy);
}

void xmpf_sync_all_1_(int *status)
{
  _XMPF_errmsg = NULL;
  xmp_sync_all(status);

  _XMPF_errmsg = "short test";

  if (_XMPF_coarrayMsg)
    fprintf(stderr, "**** done sync_all, *status=%d (%s)\n",
            *status, __FILE__);
}

void xmpf_sync_memory_0_(void)
{
  int dummy;
  xmpf_sync_memory_1_(&dummy);
}

void xmpf_sync_memory_1_(int *status)
{
  _XMPF_errmsg = NULL;
  xmp_sync_memory(status);

  _XMPF_errmsg = "test test test. long message test.";

  if (_XMPF_coarrayMsg)
    fprintf(stderr, "**** done sync_memory, *status=%d (%s)\n",
            *status, __FILE__);
}

void xmpf_sync_image_0_(int *image)
{
  int dummy;
  xmpf_sync_image_1_(image, &dummy);
}

void xmpf_sync_image_1_(int *image, int *status)
{
  _XMPF_errmsg = NULL;

  /*** not supported ***/

  *status = NotSupportedError;
  _XMPF_errmsg = "not supported yet: sync images(<image> ...)";
}

void xmpf_sync_images_0s_(int *size, int *images)
{
  int dummy;
  xmpf_sync_images_1_(size, images, &dummy);
}

void xmpf_sync_images_1s_(int *size, int *images, int *status)
{
  _XMPF_errmsg = NULL;

  /*** not supported ***/

  *status = NotSupportedError;
  _XMPF_errmsg = "not supported yet: sync images(<image-set> ...)";
}

void xmpf_sync_images_all_0_(void)
{
  int dummy;
  xmpf_sync_images_all_1_(&dummy);
}

void xmpf_sync_images_all_1_(int *status)
{
  _XMPF_errmsg = NULL;

  /*** not supported ***/

  *status = NotSupportedError;
  _XMPF_errmsg = "not supported yet: sync images(* ...)";
}



/*****************************************\
  error message to reply to Fortran
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

  
