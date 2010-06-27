/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * ompclib.h
 */

#ifndef _OMPC_THREAD_H
#define _OMPC_THREAD_H

#define USE_LOG 1

#define TRUE 1
#define FALSE 0

#include "exc_platform.h"
#include "ompc_reduction.h"

#ifdef OMNI_OS_CYGWIN32
# define SIMPLE_SPIN
#endif /* OMNI_OS_CYGWIN32 */

#if 0
#define USE_PTHREAD_BARRIER
#endif

#ifdef USE_PTHREAD      /* pthread: set gcc include path to 2.6(pdph3) */
# include <pthread.h>
# ifndef OMNI_OS_CYGWIN32
#  include <sched.h>
# endif /* !OMNI_OS_CYGWIN32 */
typedef pthread_t ompc_proc_t;
# if 0
#  define OMPC_WAIT(cond)      while(cond) sched_yield()
# else
#  define MAX_COUNT 10000
#  ifdef SIMPLE_SPIN
#   define _YIELD_ME_ sleep(0)
#  else
#   define _YIELD_ME_ sched_yield()
#  endif /* SIMPLE_SPIN */
#  define OMPC_WAIT(cond) \
        { \
          if (cond) { \
            volatile int c = 0; \
            while (cond) { \
              if (c++ > MAX_COUNT) { \
                _YIELD_ME_; \
                c = 0; \
              } \
            } \
          } \
        }
# endif /* 0 */
# define _OMPC_PROC_SELF                pthread_self()
#endif /* USE_PTHREAD */

#ifdef USE_SOL_THREAD   /* solaris thread */
#ifndef _REENTRANT
#define _REENTRANT
#endif /* !_REENTRANT */
#include <sys/types.h>
#include <thread.h>
#include <synch.h>
#include <sys/processor.h>
#include <sys/procset.h>
typedef thread_t ompc_proc_t;
#ifdef not
#define OMPC_WAIT(cond)        while(cond) thr_yield()
#else
#define MAX_COUNT 10000
#define OMPC_WAIT(cond)\
{if(cond){ volatile int c = 0; while(cond){ if(c++>MAX_COUNT){ thr_yield(); c = 0; }}}}
#endif
#define _OMPC_PROC_SELF         thr_self()
#endif

#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
#ifndef NO_RESOURCE_H
#include <sys/resource.h>
#endif /* !NO_RESOURCE_H */
#include <sys/prctl.h>
#include <signal.h>
#include <ulocks.h>
typedef pid_t ompc_proc_t;
#ifdef not
#define OMPC_WAIT(cond)        while(cond) sched_yield()
#else
#define MAX_COUNT 10000
#define OMPC_WAIT(cond)\
{if(cond){ volatile int c = 0; while(cond){ if(c++>MAX_COUNT){ sched_yield(); c = 0; }}}}
#endif
#define _OMPC_PROC_SELF         getpid()
#endif /* USE_SPROC && OMNI_OS_IRIX */

#ifdef USE_SPIN_LOCK
typedef int ompc_lock_t;
#else
# ifdef USE_PTHREAD
typedef pthread_mutex_t ompc_lock_t;
# endif
# ifdef USE_SOL_THREAD
typedef mutex_t ompc_lock_t;
# endif
#endif

typedef void* indvar_t;

typedef struct {
    ompc_lock_t  lock, wait;
    ompc_proc_t  id;
    int           count;
} ompc_nest_lock_t;

#define N_PROC_DEFAULT  4        /* default */
#define MAX_PROC        256

#if defined(OMNI_CPU_MIPS)
#   define CACHE_LINE_SIZE 128
#elif defined(OMNI_CPU_X86_64)
#   define CACHE_LINE_SIZE 64
#else
#   define CACHE_LINE_SIZE 32
#endif /* OMNI_CPU_MIPS */

#ifdef USE_SPROC
typedef void (*cfunc)();
#else
typedef void* (*cfunc)();
#endif

extern volatile int ompc_nested;       /* nested enable/disable */
extern volatile int ompc_dynamic;      /* dynamic enable/disable */
extern volatile int ompc_max_threads;  /* max number of thread */
extern int ompc_n_proc;

/* OMP processor structure */
struct ompc_proc {
    ompc_proc_t pid;           /* thread id give by [p]thread_self() */
    unsigned int pe;
    struct ompc_proc *link;     /* hash link */
    struct ompc_proc *next;     /* link */
    struct ompc_thread *thr;    /* thr != NULL, running */
    struct ompc_thread *free_thr;
    int is_used;                /* allocated or not */
};

struct ompc_thread {
    struct ompc_thread *parent;         /*  */
    struct ompc_thread *freelist;       /* freelist next */
    int num;            /* the thread number of this thread in team */
    int num_thds;       /* current running thread, refenced by children */
    int in_parallel;    /* current thread executes the region in parellel */
    cfunc func;
    int nargs;
    void *args;

    /* used for 'sections' */
    int section_id; 
    int last_section_id;

    /* used for schedule */
    int loop_chunk_size;
    indvar_t loop_end;
    indvar_t loop_sched_index;
    int loop_stride;              /* used for static scheduling */
    volatile indvar_t dynamic_index;   /* shared in children */

    indvar_t loop_id;
    volatile indvar_t ordered_id;    /* shared in team */
    volatile int ordered_step;  /* shared in team */

    /* for 'lastprivate' */
    int is_last;

    /* sync for shared data, used for 'single' directive */
    /* shared by children */
    volatile struct {
        int _v;
        char _padding[CACHE_LINE_SIZE-sizeof(int)];
    } in_flags[MAX_PROC];
    volatile int in_count;
    volatile int out_count;

    /* structure for barrier in this team */
    volatile int barrier_sense;
    volatile struct barrier_flag {
        int _v;
        any_type r_v;  /* for reduction */
        char _padding[CACHE_LINE_SIZE-sizeof(int)-sizeof(any_type)];
    } barrier_flags[MAX_PROC];
};


/* library prototypes */
void ompc_init(int argc,char *argv[]);
void ompc_init_proc_num(int);
void ompc_do_parallel(cfunc f,void *args);
void ompc_finalize(void);
void ompc_fatal(char *msg);
void ompc_terminate (int);

void ompc_init_lock(ompc_lock_t *);
void ompc_lock(volatile ompc_lock_t *);
void ompc_unlock(volatile ompc_lock_t *);
void ompc_destroy_lock(volatile ompc_lock_t *);
int ompc_test_lock(volatile ompc_lock_t *);
void ompc_init_nest_lock(ompc_nest_lock_t *);
void ompc_nest_lock(volatile ompc_nest_lock_t *);
void ompc_nest_unlock(volatile ompc_nest_lock_t *);
void ompc_destroy_nest_lock(volatile ompc_nest_lock_t *);
int ompc_test_nest_lock(volatile ompc_nest_lock_t *);
void ompc_thread_barrier(int i, struct ompc_thread *tp);

void ompc_atomic_init_lock ();
void ompc_atomic_lock ();
void ompc_atomic_unlock ();
void ompc_atomic_destroy_lock ();

void ompc_critical_init ();
void ompc_critical_destroy ();
void ompc_enter_critical (ompc_lock_t **);
void ompc_exit_critical (ompc_lock_t **);

void ompc_set_runtime_schedule(char *s);

#ifndef OMNI_CPU_ALPHA
#ifndef __GNUC__
#ifndef __PGI
# define asm(X) __asm(X)
#endif /* __PGI */
#else /* __GNUC__ */
# define asm(X) __asm volatile (X)
#endif /* __GNUC__ */
#endif


/* GNUC and Intel Fortran supports __sync_synchronize */
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#   define MBAR() __sync_synchronize()
#elif defined(OMNI_CPU_I386)
#   define MBAR() { /* asm("cpuid"); */ }
#elif defined(OMNI_CPU_SPARC)
#   define MBAR() asm("stbar")
#elif defined(OMNI_CPU_ALPHA)
extern void     __alpha_mbar _ANSI_ARGS_((void));
#   define MBAR() __alpha_mbar()
#else
#   define MBAR()
#endif /* __GNUC__ || __INTEL_COMPILER */


extern int ompc_debug_flag;
extern volatile int ompc_num_threads;

extern int ompc_in_parallel (struct ompc_thread *);
extern int ompc_get_num_threads (struct ompc_thread *);
extern void ompc_do_parallel_main (int nargs, int cond, int nthds,
    cfunc f, void *args);

#ifdef USE_LOG
extern int ompc_log_flag;
void tlog_init(char *name);
void tlog_slave_init(void);
void tlog_finalize(void);
void tlog_parallel_IN(int id);
void tlog_parallel_OUT(int id);
void tlog_barrier_IN(int id);
void tlog_barrier_OUT(int id);
void tlog_loop_init_EVENT(int id);
void tlog_loop_next_EVENT(int id);
void tlog_section_EVENT(int id);
void tlog_single_EVENT(int id);
void tlog_critial_IN(int id);
void tlog_critial_OUT(int id);
#endif

extern ompc_lock_t ompc_proc_lock_obj, ompc_thread_lock_obj;

#define OMPC_THREAD_LOCK()      ompc_lock(&ompc_thread_lock_obj)
#define OMPC_THREAD_UNLOCK()    ompc_unlock(&ompc_thread_lock_obj)
#define OMPC_PROC_LOCK()        ompc_lock(&ompc_proc_lock_obj)
#define OMPC_PROC_UNLOCK()      ompc_unlock(&ompc_proc_lock_obj)

#endif /* _OMPC_THREAD_H */


