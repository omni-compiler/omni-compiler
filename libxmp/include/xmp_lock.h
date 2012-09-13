#ifndef _XMP_LOCK_H
#define _XMP_LOCK_H

#ifdef _XMP_COARRAY_GASNET
#include <gasnet.h>
typedef struct xmp_lock{
  int islocked;
  gasnet_hsl_t  inchsl;
  int  wait_size;   /* How many elements in wait_list */
  int  wait_head;   /* Index for next dequeue */
  int  wait_tail;   /* Index for next enqueue */
  int *wait_list;   /* Circular queue of waiting threads */
} xmp_gasnet_lock_t;
#define xmp_lock xmp_gasnet_lock_t
#endif

#endif
