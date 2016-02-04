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

//#define USE_LOG 1

#define TRUE 1
#define FALSE 0

#include "exc_platform.h"
#include "ompc_reduction.h"

#include <abt.h>
typedef ABT_xstream ompc_proc_t;
#define _YIELD_ME_ ABT_thread_yield()
#define OMPC_WAIT(cond) while (cond) { _YIELD_ME_; }
#define MAX_SPIN_COUNT 0
#define OMPC_SIGNAL(statement, condvar, mutex) \
    ABT_mutex_lock(mutex); \
    statement; \
    ABT_cond_signal(condvar); \
    ABT_mutex_unlock(mutex)
#define OMPC_WAIT_UNTIL(condexpr, condvar, mutex) \
    for (int c = 0; c <= MAX_SPIN_COUNT; c++) { \
        if (condexpr) { \
            break; \
        } \
        if (c == MAX_SPIN_COUNT) { \
            ABT_mutex_lock(mutex); \
            while (!(condexpr)) { \
                ABT_cond_wait(condvar, mutex); \
            } \
            ABT_mutex_unlock(mutex); \
        } \
    }
#define _OMPC_PROC_SELF ompc_xstream_self()

typedef ABT_mutex ompc_lock_t;

typedef void* indvar_t;

typedef struct {
    ompc_lock_t  lock, wait;
    ompc_proc_t  id;
    int           count;
} ompc_nest_lock_t;

#define N_PROC_DEFAULT  4        /* default */
#define LOG_MAX_PROC    8
#define MAX_PROC        (1 << LOG_MAX_PROC)  // 256

#define CACHE_LINE_SIZE 64  // x86-64

typedef void* (*cfunc)();

extern volatile int ompc_nested;       /* nested enable/disable */
extern volatile int ompc_dynamic;      /* dynamic enable/disable */
extern volatile int ompc_max_threads;  /* max number of thread */

/* OMP processor structure */
struct ompc_proc {
    ompc_proc_t pid;           /* thread id give by [p]thread_self() */
    unsigned int pe;
};

struct ompc_tree_barrier_node
{
    int num_children;
    int count;
    _Bool volatile sense;
    ABT_mutex mutex;
    ABT_cond cond;
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct ompc_tree_barrier
{
    int num_threads;
    int depth;
    struct ompc_tree_barrier_node nodes[MAX_PROC - 1];
};

struct ompc_ult_pool {
    ABT_thread *ult_list;
    int size_allocated;
    int size_created;
    int size_used;
};

struct ompc_tasklet_pool {
    ABT_task *tasklet_list;
    int size_allocated;
    int size_created;
    int size_used;
};

struct ompc_thread {
    ABT_thread *ult_ptr;
    ABT_task *tasklet_ptr;
    
    struct ompc_thread *parent;         /*  */
    int num;            /* the thread number of this thread in team */
    int num_thds;       /* current running thread, refenced by children */
    int in_parallel;    /* current thread executes the region in parellel */
    int parallel_nested_level; // FIXME for logging, delete this after test
    int es_start;
    int es_length;
    int set_num_thds;
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

    /* for sync between parent and children */
    int run_children;
    ABT_mutex broadcast_mutex;
    ABT_mutex reduction_mutex;
    ABT_cond  broadcast_cond;
    ABT_cond  reduction_cond;

    /* sync for shared data, used for 'single' directive */
    /* shared by children */
    volatile struct {
        int _v;
        char _padding[CACHE_LINE_SIZE-sizeof(int)];
    } in_flags[MAX_PROC];
    volatile int in_count;
    volatile int out_count;

    /* structure for barrier in this team */
    //_Bool barrier_sense;
    volatile int barrier_sense;
    struct ompc_tree_barrier_node *node_stack[LOG_MAX_PROC];
    volatile struct barrier_flag {
        int _v;
        any_type r_v;  /* for reduction */
        char _padding[CACHE_LINE_SIZE-sizeof(int)-sizeof(any_type)];
    } barrier_flags[MAX_PROC];
    
    struct ompc_tree_barrier tree_barrier;
};


/* library prototypes */
void ompc_init(int argc,char *argv[]);
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

ompc_proc_t ompc_xstream_self();

void ompc_tree_barrier_init(struct ompc_tree_barrier *barrier,
                            int num_threads);
void ompc_tree_barrier_finalize(struct ompc_tree_barrier *barrier);
void ompc_tree_barrier_wait(struct ompc_tree_barrier *barrier,
                            struct ompc_thread *thread);

/* GNUC and Intel Fortran supports __sync_synchronize */
#define MBAR() __sync_synchronize()

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
