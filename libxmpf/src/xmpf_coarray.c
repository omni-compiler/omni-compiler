#include "xmpf_internal.h"

#define DESCR_ID_MAX   250
#define BOOL int
#define TRUE   1
#define FALSE  0

typedef struct {
  BOOL is_used;
  void *co_desc;
  void *co_addr;
} _info_t;

static _info_t _descTab[DESCR_ID_MAX] = {};
static int _nextId = 0;

static int _set_descTab(void *co_desc, void *co_addr);
static int _getNextUnusedIdx();

void xmpf_coarray_malloc_(int *descrId, void **pointer, int *size, int *unit)
{
  int n_elems = *size;
  size_t elem_size = (size_t)(*unit);
  void *co_desc;
  void *co_addr;
  int idx;

  _XMP_coarray_malloc_info_1(n_elems, elem_size);   // in xmp_coarray_set.c
  _XMP_coarray_malloc_image_info_1();            // in xmp_coarray_set.c
  _XMP_coarray_malloc_do(&co_desc, &co_addr);    // in xmp_coarray_set.c
  *pointer = co_addr;

  /* set table */
  idx = _set_descTab(co_desc, co_addr);
  *descrId = _set_descTab(co_desc, co_addr);
}



static int _set_descTab(void *co_desc, void *co_addr)
{
  int idx;

  idx = _getNextUnusedIdx();
  if (idx < 0) {         /* fail */
    _XMP_fatal("xmpf_coarray.c: no more descriptor table.");
    return idx;
  }

  _descTab[idx].is_used = TRUE;
  _descTab[idx].co_desc = co_desc;
  _descTab[idx].co_addr = co_addr;
  return idx;
}

  
static int _getNextUnusedIdx() {
  int i;

  /* try white area */
  for (i = _nextId; i < DESCR_ID_MAX; i++) {
    if (! _descTab[i].is_used) {
      ++_nextId;
      return i;
    }
  }

  /* try reuse */
  for (i = 0; i < _nextId; i++) {
    if (! _descTab[i].is_used) {
      _nextId = i + 1;
      return i;
    }
  }

  /* error: no room */
  return -1;
}


