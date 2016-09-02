#include <limits.h>

#define XMP_ASYNC_NONE INT_MIN

typedef enum {

  // 4 event may occur; callback always invoked when event occurs

  XMPT_EVENT_FULLY_SUPPORTED,

  // 3 event may occur; callback invoked when convenient
  xmpt_event_reflect_begin,
  xmpt_event_reflect_end,
  xmpt_event_bcast_begin,
  xmpt_event_bcast_end,
  xmpt_event_wait_async_begin,
  xmpt_event_wait_async_end,

  XMPT_EVENT_PARTIALLY_SUPPORTED,

  // 2 event will never occur in this runtime

  XMPT_EVENT_NEVER,
  
  // 1 event may occur; no callback is possible (e.g., not yet implemented)
  
  XMPT_EVENT_ALL
} xmpt_event_t;

typedef void (*xmpt_reflect_begin_callback_t)(
  void *adesc,
  int *lwidth,
  int *uwidth,
  int *is_periodic,
  int async_id
);

typedef void (*xmpt_reflect_end_callback_t)(
  void *adesc,
  int *lwidth,
  int *uwidth,
  int *is_periodic,
  int async_id
);

typedef void (*xmpt_bcast_begin_callback_t)(
  int async_id
);

typedef void (*xmpt_bcast_end_callback_t)(
  int async_id
);

typedef void (*xmpt_wait_async_begin_callback_t)(
  int async_id
);

typedef void (*xmpt_wait_async_end_callback_t)(
  int async_id
);

/* typedef void (*xmpt_callback_t)( */
/*   void */
/*   ); */
typedef void (*xmpt_callback_t);

int xmpt_set_callback(xmpt_event_t, xmpt_callback_t);

/* extern xmpt_bcast_begin_callback_t  xmpt_bcast_begin_callback; */
/* extern xmpt_bcast_end_callback_t    xmpt_bcast_end_callback; */
/* extern xmpt_wait_async_callback_t   xmpt_wait_async_callback; */
extern xmpt_callback_t xmpt_callback[XMPT_EVENT_ALL];
extern int xmpt_enabled;
