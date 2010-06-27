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
#ifndef USE_SPIN_LOCK

# ifdef USE_SOL_THREAD
void ompc_init_lock(ompc_lock_t *lp)
{
    mutex_init(lp, NULL, NULL);
}

void ompc_lock(volatile ompc_lock_t *lp)
{
    mutex_lock((mutex_t *)lp);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    mutex_unlock((mutex_t *)lp);
}

void ompc_destroy_lock(volatile ompc_lock_t *lp)
{
    mutex_destroy((mutex_t *)lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    return (mutex_trylock((mutex_t *)lp) == 0); 
}

# endif /* USE_SOL_THREAD */

# ifdef USE_PTHREAD
void ompc_init_lock(ompc_lock_t *lp)
{
    pthread_mutex_init((pthread_mutex_t *)lp, NULL);
}

void ompc_lock(volatile ompc_lock_t *lp)
{
    pthread_mutex_lock((pthread_mutex_t *)lp);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    pthread_mutex_unlock((pthread_mutex_t *)lp);
}

void ompc_destroy_lock(volatile ompc_lock_t *lp)
{
    pthread_mutex_destroy((pthread_mutex_t *)lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    return (pthread_mutex_trylock((pthread_mutex_t *)lp) == 0);
}

# endif /* USE_PTHREAD */

#else /* !USE_SPIN_LOCK */

void ompc_init_lock(ompc_lock_t *lp)
{
    *lp = 0;
}

void ompc_destroy_lock(volatile ompc_lock_t *lp){
    /* do nothing */
}

# ifdef OMNI_CPU_SPARC
extern void LockWithLdstUB _ANSI_ARGS_((volatile int *));
extern void UnlockWithLdstUB _ANSI_ARGS_((volatile int *));
extern int TestLockWithLdstUB _ANSI_ARGS_((volatile int *));

void ompc_lock(volatile ompc_lock_t *lp)
{
    LockWithLdstUB(lp);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    UnlockWithLdstUB(lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    return !TestLockWithLdstUB(lp);
}
# endif /* OMNI_CPU_SPARC */

# ifdef OMNI_CPU_I386
int _xchg_1 (volatile int *p);

void _dummy ()
{
  asm ("        .align 4                        ");
#  ifdef OMNI_OS_CYGWIN32
  asm (".def    __xchg_1                        ");
  asm ("        .scl    2                       ");
  asm ("        .type   32                      ");
  asm (".endef                                  ");
  asm (".globl __xchg_1                         ");
  asm ("__xchg_1:                               ");
#  else
#ifndef __INTEL_COMPILER
  asm ("        .type    _xchg_1,@function      ");
#endif
  asm (".globl _xchg_1                          ");
#  endif /* OMNI_OS_CYGWIN32 */
  asm ("_xchg_1:                                ");
  asm ("        pushl %ebp                      ");
  asm ("        movl %esp,%ebp                  ");
  asm ("        movl 8(%ebp),%edx               ");
  asm ("        movl $1,%eax                    ");
  asm ("        xchgl 0(%edx),%eax              ");
  asm ("        leave                           ");
  asm ("        ret                             ");
}

void ompc_lock(volatile ompc_lock_t *lp)
{
 again:
    while(*lp != 0) /* spin wait */;
    if(_xchg_1(lp) != 0) goto again;
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    *lp = 0;
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    if(_xchg_1(lp) != 0) return 0;
    else return 1;
}

# endif /* OMNI_CPU_I386 */

# ifdef OMNI_CPU_MIPS
/* call SGI library */
void ompc_lock(volatile ompc_lock_t *lp)
{
  while (__lock_test_and_set(lp, 1) != 0);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
  __lock_release(lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
  return __lock_test_and_set(lp, 1);
}

# endif /* OMNI_CPU_MIPS */

# ifdef OMNI_CPU_ALPHA
extern int      __alpha_spin_lock _ANSI_ARGS_((volatile int *lock));
extern void     __alpha_spin_unlock _ANSI_ARGS_((volatile int *lock));
extern int      __alpha_spin_test_lock _ANSI_ARGS_((volatile int *lock));

void ompc_lock(volatile ompc_lock_t *lp)
{
    __alpha_spin_lock(lp);
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    __alpha_spin_unlock(lp);
}

int ompc_test_lock(volatile ompc_lock_t *lp)
{
    return __alpha_spin_test_lock(lp);
}

# endif /* OMNI_CPU_ALPHA */

# ifdef OMNI_CPU_RS6000
#include <sys/atomic_op.h>

void ompc_lock(volatile ompc_lock_t *lp)
{
    while (_check_lock((atomic_p)lp, 0, 1));
}

void ompc_unlock(volatile ompc_lock_t *lp)
{
    _clear_lock((atomic_p)lp, 0);
}

int ompc_test_lock(volatile ompc_lock_t *lp) {
    return !_check_lock((atomic_p)lp, 0, 1);
}

# endif /* OMNI_CPU_RS6000 */

#endif /* !USE_SPIN_LOCK */

void ompc_init_nest_lock (ompc_nest_lock_t *lp)
{
  ompc_init_lock (&lp->lock);
#ifndef USE_SPIN_LOCK
  ompc_init_lock (&lp->wait);
#endif
  lp->count = 0;
}

void ompc_nest_lock (volatile ompc_nest_lock_t *lp)
{
  ompc_proc_t  id = _OMPC_PROC_SELF;

#ifndef USE_SPIN_LOCK
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
#else 
 retry:
  ompc_lock (&lp->lock);
  if (lp->count != 0) {
    if (id == lp->id) {               /* already lock by own thread */
      lp->count ++;
    } else {                          /* already lock by othre thread, fail */
      ompc_unlock (&lp->lock);       /* dirty implement */
      goto retry;
    }
  } else {                            /* no thread lock, yet */
    lp->id    = id;
    lp->count = 1;
  }
  ompc_unlock (&lp->lock);
#endif
}

void ompc_nest_unlock (volatile ompc_nest_lock_t *lp)
{
  ompc_lock (&lp->lock);
  lp->count -= 1;
#ifndef USE_SPIN_LOCK
  if (lp->count == 0) {
    ompc_unlock (&lp->lock);
    ompc_unlock (&lp->wait);
  } else {
    ompc_unlock (&lp->lock);
  }
#else
  ompc_unlock (&lp->lock);
#endif
}

void ompc_destroy_nest_lock (volatile ompc_nest_lock_t *lp)
{
  ompc_destroy_lock (&lp->lock);
#ifndef USE_SPIN_LOCK
  ompc_destroy_lock (&lp->wait);
#endif
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
#ifndef USE_SPIN_LOCK
    ompc_lock (&lp->wait);
#endif
    lp->id    = id;
    lp->count = 1;
  }
  ompc_unlock (&lp->lock);
  return lp->count;
}
