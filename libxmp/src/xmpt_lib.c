#include "xmp_internal.h"

/* xmpt_bcast_begin_callback_t xmpt_bcast_begin_callback = NULL; */
/* xmpt_bcast_end_callback_t   xmpt_bcast_end_callback   = NULL; */
/* xmpt_wait_async_callback_t  xmpt_wait_async_callback  = NULL; */
xmpt_callback_t xmpt_callback[XMPT_EVENT_ALL] = { 0 };

int __attribute__((weak)) xmpt_initialize(){
  return 0;
}

int xmpt_set_callback(xmpt_event_t event, xmpt_callback_t callback){

  /* switch (event){ */
  /* case xmpt_bcast_begin: */
  /*   xmpt_bcast_begin_callback = (xmpt_bcast_begin_callback_t)callback; */
  /*   break; */
  /* case xmpt_bcast_end: */
  /*   xmpt_bcast_end_callback = (xmpt_bcast_end_callback_t)callback; */
  /*   break; */
  /* case xmpt_wait_async: */
  /*   xmpt_wait_async_callback = (xmpt_wait_async_callback_t)callback; */
  /*   break; */
  /* default: */
  /*   _XMP_warning("callback not supported."); */
  /* } */

  /* 0 callback registration error (e.g., callbacks cannot be registered at this time). */
  /* 1 event may occur; no callback is possible (e.g., not yet implemented) */
  /* 2 event will never occur in this runtime */
  /* 3 event may occur; callback invoked when convenient */
  /* 4 event may occur; callback always invoked when event occurs */

  if (!callback || event < 0 || event >= XMPT_EVENT_ALL) return 0;

  if (event < XMPT_EVENT_FULLY_SUPPORTED){
    xmpt_callback[event] = callback;
    return 4;
  }
  else if (event < XMPT_EVENT_PARTIALLY_SUPPORTED){
    xmpt_callback[event] = callback;
    return 3;
  }
  else if (event < XMPT_EVENT_NEVER){
    return 2;
  }
  else if (event < XMPT_EVENT_ALL){
    return 1;
  }
  else { // == XMPT_EVENT_XXX
    return 0;
  }

}
