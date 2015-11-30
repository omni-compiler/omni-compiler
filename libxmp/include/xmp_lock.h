#ifndef _XMP_LOCK_H
#define _XMP_LOCK_H
typedef struct xmp_lock{
  _Bool  islocked;
  void   *hsl;           /* the data type is gasnet_hsl_t defined in gasnet.h */
  int    wait_size;      /* How many elements in wait_list */
  int    wait_head;      /* Index for next dequeue */
  int    wait_tail;      /* Index for next enqueue */
  int    *wait_list;     /* Circular queue of waiting threads */
} xmp_lock_t;
typedef xmp_lock_t xmp_gasnet_lock_t;
#endif
