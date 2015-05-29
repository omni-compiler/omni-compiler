/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_lock.c
 */
#include "ompclib.h"

/* Lock/Unlock */

void ompc_init_lock(ompc_lock_t *lp)
{
    ABT_mutex_create((ABT_mutex *)lp);
}

void ompc_lock(volatile ompc_lock_t *lp)
{
    ABT_mutex_lock(*(ABT_mutex *)lp);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    ABT_mutex_unlock(*(ABT_mutex *)lp);
}

void ompc_destroy_lock(volatile ompc_lock_t *lp)
{
    ABT_mutex_free((ABT_mutex *)lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    return ABT_mutex_trylock(*(ABT_mutex *)lp) == ABT_SUCCESS;
}

void ompc_init_nest_lock (ompc_nest_lock_t *lp)
{
  ompc_init_lock (&lp->lock);
  ompc_init_lock (&lp->wait);
  lp->count = 0;
}

void ompc_nest_lock (volatile ompc_nest_lock_t *lp)
{
  ompc_proc_t  id = _OMPC_PROC_SELF;

  int           wl;

 retry:
  if (lp->count != 0  &&  lp->id != id) {
    ompc_lock (&lp->wait);
    if (ompc_test_lock (&lp->lock) == 0) {
      ompc_unlock (&lp->wait);
      goto retry;
    }
    wl = 1;
  } else {
    ompc_lock (&lp->lock);
    wl = 0;
  }
  if (lp->count != 0) {
    if (id == lp->id) {               /* already lock by own thread */
      lp->count ++;
    } else {                          /* already lock by othre thread, fail */
      ompc_unlock (&lp->lock);
      if (wl == 1) {
        ompc_unlock (&lp->wait);
      }
      goto retry;
    }
  } else {                            /* no thread lock, yet */
    if (wl == 0) {
      ompc_lock (&lp->wait);
    }
    lp->id    = id;
    lp->count = 1;
  }
  ompc_unlock (&lp->lock);
}

void ompc_nest_unlock (volatile ompc_nest_lock_t *lp)
{
  ompc_lock (&lp->lock);
  lp->count -= 1;
  if (lp->count == 0) {
    ompc_unlock (&lp->lock);
    ompc_unlock (&lp->wait);
  } else {
    ompc_unlock (&lp->lock);
  }
}

void ompc_destroy_nest_lock (volatile ompc_nest_lock_t *lp)
{
  ompc_destroy_lock (&lp->lock);
  ompc_destroy_lock (&lp->wait);
}

int ompc_test_nest_lock (volatile ompc_nest_lock_t *lp)
{
  ompc_proc_t  id = _OMPC_PROC_SELF;


  if (lp->count != 0  &&  lp->id != id) {
    return 0;
  }
  if (ompc_test_lock (&lp->lock) == 0) {
    return 0;
  }
  if (lp->count != 0) {
    if (id == lp->id) {               /* already lock by own thread */
      lp->count ++;
    } else {                          /* already lock by othre thread, fail */
      ompc_unlock (&lp->lock);
      return 0;
    }
  } else {                            /* no thread lock, yet */
    ompc_lock (&lp->wait);
    lp->id    = id;
    lp->count = 1;
  }
  ompc_unlock (&lp->lock);
  return lp->count;
}
