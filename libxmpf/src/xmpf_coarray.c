#include "xmpf_internal.h"

#define DESCR_ID_MAX   250
#define SMALL_WORK_SIZE_KB  10

#define BOOL int
#define TRUE   1
#define FALSE  0


static coarray_info_t _descTab[DESCR_ID_MAX] = {};
static int _nextId = 0;

static int _set_descTab(void *co_desc, void *co_addr, int n_elems, size_t elem_size);
static int _getNextUnusedIdx();


void xmpf_coarray_malloc_(int *descrId, void **pointer, int *size, int *unit)
{
  xmpf_coarray_malloc(descrId, pointer, *size, (size_t)(*unit));
}

void xmpf_coarray_malloc(int *descrId, void* *pointer, int n_elems, size_t elem_size)
{
  void* co_desc;
  void* co_addr;

  _XMP_coarray_malloc_info_1(n_elems, elem_size);  // see libxmp/src/xmp_coarray_set.c
  _XMP_coarray_malloc_image_info_1();              // see libxmp/src/xmp_coarray_set.c
  _XMP_coarray_malloc_do(&co_desc, &co_addr);      // see libxmp/src/xmp_coarray_set.c
  *pointer = co_addr;

  /* set table */
  *descrId = _set_descTab(co_desc, co_addr, n_elems, elem_size);
}


coarray_info_t *get_coarray_info(int idx)
{
#if 1
  fprintf(stdout, "get idx=%d, is_used=%d, co_desc=%p, "
          "co_addr=%p, n_elems=%d, elem_size=%ld\n",
          idx, _descTab[idx].is_used, _descTab[idx].co_desc,
          _descTab[idx].co_addr, _descTab[idx].n_elems,
          _descTab[idx].elem_size);
#endif

  return &(_descTab[idx]);
}


static int _set_descTab(void *co_desc, void *co_addr, int n_elems, size_t elem_size)
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
  _descTab[idx].n_elems = n_elems;
  _descTab[idx].elem_size = elem_size;

#if 1
  fprintf(stdout, "set idx=%d, is_used=%d, co_desc=%p, "
          "co_addr=%p, n_elems=%d, elem_size=%ld\n",
          idx, _descTab[idx].is_used, _descTab[idx].co_desc,
          _descTab[idx].co_addr, _descTab[idx].n_elems,
          _descTab[idx].elem_size);
#endif

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


