/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_thread.c
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "exc_platform.h"
#include "ompclib.h"

#include <hwloc.h>
#include "abt_logger.h"
#include <errno.h>

//#define __ABTL_LOG_ENABLE
// __ABTL_LOG_LEVEL
// 0: outer loop, used for flat OpenMP version
// 1: inner loop, used for nested OpenMP version
#define __ABTL_LOG_LEVEL 1
//#define __TEST_WORK_STEALING

// FIXME temporary impl, needs refactoring
static ABT_xstream xstreams[MAX_PROC];

static ABT_key tls_key;
static void tls_free(void *value) {
    free(value);
}

static hwloc_topology_t topo;
static hwloc_const_cpuset_t allset;
static void thread_affinity_setup(int i) {
    hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i);
    hwloc_cpuset_t set = hwloc_bitmap_dup(core->cpuset);

    int res;
    res = hwloc_set_cpubind(topo, set, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    if (res) {
        int err = errno;
        printf("[%d] error: hwloc_set_cpubind(): %d\n", i, err);
    }

    res = hwloc_set_membind(topo, set, HWLOC_MEMBIND_FIRSTTOUCH, HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT);
    if (res) {
        int err = errno;
        printf("[%d] error: hwloc_set_membind(): %d\n", i, err);
    }

    hwloc_bitmap_free(set);
}

#ifdef __TEST_WORK_STEALING
static ABT_sched scheds[MAX_PROC];
static ABT_pool  pools[MAX_PROC];

typedef struct {
    uint32_t event_freq;
} sched_data_t;

static int sched_init(ABT_sched sched, ABT_sched_config config) {
    sched_data_t *p_data = (sched_data_t *)calloc(1, sizeof(sched_data_t));

    ABT_sched_config_read(config, 1, &p_data->event_freq);
    ABT_sched_set_data(sched, (void *)p_data);

    return ABT_SUCCESS;
}

static int sched_free(ABT_sched sched) {
    void *p_data;

    ABT_sched_get_data(sched, &p_data);
    free(p_data);

    return ABT_SUCCESS;
}

static void sched_run(ABT_sched sched) {
    sched_data_t *p_data;
    ABT_sched_get_data(sched, (void **)&p_data);

    int num_pools;
    ABT_sched_get_num_pools(sched, &num_pools);

    ABT_pool *sched_pools;
    sched_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_pools);
    ABT_sched_get_pools(sched, num_pools, 0, sched_pools);

    uint32_t work_count = 0;
    ABT_unit unit;
    while (1) {
        // Execute one work unit from the scheduler's pool
        ABT_pool_pop(sched_pools[0], &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, sched_pools[0]);
        }
        else if (num_pools > 1) {
            // Steal a work unit from other pools
            // RANDOM
            unsigned seed = time(NULL);
            int target_pool_idx = (num_pools == 2) ? 1 : (rand_r(&seed) % (num_pools - 1) + 1);
            ABT_pool target_pool = sched_pools[target_pool_idx];

            size_t target_pool_size;
            ABT_pool_get_size(target_pool, &target_pool_size);
            if (target_pool_size > 0) {
                // Pop one work unit
                ABT_pool_pop(target_pool, &unit);
                if (unit != ABT_UNIT_NULL) {
                    ABT_xstream_run_unit(unit, target_pool);
                }
            }
        }

        if (++work_count >= p_data->event_freq) {
            work_count = 0;
            ABT_bool stop;
            ABT_sched_has_to_stop(sched, &stop);
            if (stop == ABT_TRUE) break;
            ABT_xstream_check_events(sched);
        }
    }

    free(sched_pools);
}

static void create_scheds(void)
{
    ABT_sched_config config;
    ABT_pool *my_pools;
    int i, k;

    ABT_sched_config_var cv_event_freq = {
        .idx = 0,
        .type = ABT_SCHED_CONFIG_INT
    };

    ABT_sched_def sched_def = {
        .type = ABT_SCHED_TYPE_ULT,
        .init = sched_init,
        .run = sched_run,
        .free = sched_free,
        .get_migr_pool = NULL
    };

    /* Create a scheduler config */
    ABT_sched_config_create(&config, cv_event_freq, 10,
                            ABT_sched_config_var_end);

    my_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * ompc_max_threads);
    for (i = 0; i < ompc_max_threads; i++) {
        for (k = 0; k < ompc_max_threads; k++) {
            my_pools[k] = pools[(i + k) % ompc_max_threads];
        }

        ABT_sched_create(&sched_def, ompc_max_threads, my_pools, config, &scheds[i]);
    }
    free(my_pools);

    ABT_sched_config_free(&config);
}

static void sched_setup(void *arg) {
    int idx = (int)(size_t)arg;
    ABT_xstream_set_main_sched(xstreams[idx], scheds[idx]);
}
#endif // __TEST_WORK_STEALING

#define PROC_HASH_SIZE  0x100L
#define PROC_HASH_MASK (PROC_HASH_SIZE-1)
#define PROC_HASH_IDX(ID) ((unsigned long int)((unsigned long int)(ID) & (PROC_HASH_MASK)))

#define DEF_STACK_SIZE  1*1024*1024     /* default stack size */

int ompc_debug_flag = 0;       /* debug output control */
int ompc_log_flag = 0;         /* log */

volatile int ompc_nested;      /* nested enable/disable */
volatile int ompc_dynamic;     /* dynamic enable/disable */
volatile int ompc_task_end;    /* slave task end */
volatile int ompc_proc_counter = 0;    /* thread generation counter */

volatile int ompc_max_threads; /* max number of thread */
volatile int ompc_num_threads; /* number of team member? */

/* system lock variables */
ompc_lock_t ompc_proc_lock_obj, ompc_thread_lock_obj;

/* proc table */
struct ompc_proc *ompc_procs;
static int proc_last_used = 0;

static ompc_proc_t ompc_master_proc_id;

/* prototype */
static void ompc_xstream_setup();
static void ompc_thread_wrapper_func(void *arg);
static ompc_thread_t ompc_thread_self();
static struct ompc_proc *ompc_new_proc(int i);
static struct ompc_proc *ompc_get_proc(struct ompc_thread *par, struct ompc_thread *cur,
                                       int thread_num, int num_threads);
static struct ompc_thread *ompc_alloc_thread(void);
/*static*/ struct ompc_thread *ompc_current_thread(void);

extern void ompc_call_fsub(struct ompc_thread *tp);

/* 
 * initialize library
 */
void
ompc_init(int argc,char *argv[])
{
    ABT_init(argc, argv);
    tls_key = ABT_KEY_NULL;
    ABT_key_create(tls_free, &tls_key);

    // hwloc init
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
    allset = hwloc_topology_get_complete_cpuset(topo);
/*
    hwloc_cpuset_t set = hwloc_bitmap_dup(allset);
    int res = hwloc_set_membind(topo, set, HWLOC_MEMBIND_FIRSTTOUCH, HWLOC_MEMBIND_PROCESS);
    if (res) {
        int err = errno;
        printf("error: hwloc_set_membind(): %d\n", err);
    }
    hwloc_bitmap_free(set);
*/
//    int num_numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    // hwloc init end

    {
      char buff[BUFSIZ];
      FILE *fp;
      int  npes;
      char procfile[] = "/proc/stat";

      fp = fopen (procfile, "r");
      if ( fp == NULL ){
        fprintf (stderr, "cannot open \"%s\".\n"
                 "cannot get maximum number of processors.\n", procfile);
        ompc_max_threads = 1;
      }
      else {
        npes = 0;
        while( fgets(buff, BUFSIZ, fp) != NULL ){
          if ( !strncmp(buff, "cpu", 3) && isdigit(buff[3]) ){
            npes += 1;
          }
        }
        fclose (fp);
        ompc_max_threads = (npes == 0)? 1: npes;
      }
    }

    atexit(ompc_finalize);

    char *cp;
    int val;
    cp = getenv("OMPC_DEBUG");
    if(cp != NULL){
        ompc_debug_flag = TRUE;
        fprintf(stderr,"debug flag on ...\n");
    }

#ifdef USE_LOG
    cp = getenv("OMPC_LOG");
    if(cp != NULL){
        ompc_log_flag = TRUE;
        tlog_init(argv[0]);
    }
#endif /* USE_LOG */

    cp = getenv("OMP_SCHEDULE");
    if(cp != NULL)
        ompc_set_runtime_schedule(cp);

    cp = getenv("OMP_DYNAMIC");
    if(cp != NULL && (strcmp(cp, "TRUE") == 0 || strcmp(cp, "true") == 0))
        ompc_dynamic = 1;
    else
        ompc_dynamic = 0;      /* dynamic enable/disable */

    cp = getenv("OMP_NESTED");
    if(cp != NULL && (strcmp(cp,"TRUE") == 0 || strcmp(cp,"true") == 0))
        ompc_nested = 1;
    else
        ompc_nested = 0;       /* nested enable/disable */

    cp = getenv("OMPC_NUM_PROCS");   /* processor */
    if ( cp != NULL ){
        sscanf(cp, "%d", &val);
        if(val <= 0) ompc_fatal("bad OMPC_NUM_PROCS(<= 0)");
        ompc_max_threads = val;
    }

    cp = getenv("OMP_NUM_THREADS");     /* a number of team member */
    if ( cp == NULL )
        ompc_num_threads = ompc_max_threads;
    else {
        sscanf(cp, "%d", &val);
        if(val <= 0) ompc_fatal("bad OMP_NUM_THREADS(<= 0)");
        ompc_num_threads = val;
    }

    // FIXME not available
    /*
    cp = getenv("OMPC_STACK_SIZE");   // stack size of threads
    if ( cp != NULL ){
        char lstr[64];
        size_t len = strlen(cp);
        int unit = 1;

        if ( strncmp(&cp[len-1], "k", 1) == 0 || strncmp(&cp[len-1], "K", 1) == 0 ){
            len -= 1;
            unit *= 1024;
        }
        else if ( strncmp(&cp[len-1], "m", 1) == 0 || strncmp(&cp[len-1], "M", 1) == 0 ){
            len -= 1;
            unit *= 1024*1024;
        }
        strncpy(lstr, cp, len);
        sscanf(lstr, "%d", &val);
        if ( val <= 0 ) ompc_fatal("bad OMPC_STACK_SIZE(<= 0)");
        size_t maxstack = val*unit;
        if ( maxstack < DEF_STACK_SIZE ){
            maxstack = 0;       // default/
            printf("Stack size is not change, because it is less than the default(=1MB).\n");
        }
    }
    */

    ompc_task_end = 0;

    /* init system lock */
    ompc_init_lock(&ompc_proc_lock_obj);
    ompc_init_lock(&ompc_thread_lock_obj);
    ompc_critical_init ();     /* initialize critical lock */
    ompc_atomic_init_lock ();  /* initialize atomic lock */

    // allocate proc structure
    ompc_procs = (struct ompc_proc *)malloc(sizeof(struct ompc_proc) * ompc_max_threads);
    if (ompc_procs == NULL) ompc_fatal("Cannot allocate proc table.");
    bzero(ompc_procs, sizeof(struct ompc_proc) * ompc_max_threads);

    // inig ompc_master_proc_id
    ompc_master_proc_id = _OMPC_PROC_SELF;

#ifdef __TEST_WORK_STEALING
    // work stealing scheduler setup
    for (int i = 0; i < ompc_max_threads; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                              ABT_TRUE, &pools[i]);
    }

    create_scheds();
#endif

    // ES setup
    ABT_thread threads[MAX_PROC];
    if (ompc_debug_flag) fprintf(stderr, "Creating %d slave thread ...\n", ompc_max_threads - 1);
    for (int i = 0; i < ompc_max_threads; i++) {
        if (ompc_debug_flag) fprintf(stderr, "Creating slave %d  ...\n", i);

        if (i == 0) {
            ABT_xstream_self(&xstreams[0]);
            ompc_xstream_setup(0);
            continue;
        }

        int res = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        if (res) {
            extern int errno;
            fprintf(stderr, "thread create fails at id %d:%d errno=%d\n", i, res, errno);
            perror("thread creation");
            exit(1);
        }

        size_t tid = (size_t)i;
        ABT_thread_create_on_xstream(xstreams[i], ompc_xstream_setup,
                                     (void *)tid, ABT_THREAD_ATTR_NULL, &(threads[i]));
    }

    for (int i = 1; i < ompc_max_threads; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }

#ifdef __TEST_WORK_STEALING
    for (int i = 0; i < ompc_max_threads; i++) {
        if (i == 0) {
            sched_setup(0);
            continue;
        }

        size_t tid = (size_t)i;
        ABT_thread_create_on_xstream(xstreams[i], sched_setup,
                                     (void *)tid, ABT_THREAD_ATTR_NULL, &(threads[i]));
    }

    for (int i = 1; i < ompc_max_threads; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }
#endif // __TEST_WORK_STEALING

    // setup master root thread
    struct ompc_thread *tp = ompc_alloc_thread();
    tp->num             = 0;    /* team master */
    tp->in_parallel     = 0;
    tp->parent          = NULL;
    tp->tid = ompc_thread_self();
    ABT_key_set(tls_key, (void *)tp);

    // argobots logger init
    ABTL_init(ompc_max_threads);

    if (ompc_debug_flag) fprintf(stderr, "init end(Master)\n");
}


/* finalize */
void
ompc_finalize()
{
#ifdef USE_LOG
    if(ompc_log_flag){
        tlog_finalize();
    }
#endif /* USE_LOG */
}

void
ompc_fatal(char * msg)
{
    fprintf(stderr, "OMPC FATAL: %s\n", msg);
    exit(1);
}

int
ompc_is_master_proc()
{
    return ompc_master_proc_id == _OMPC_PROC_SELF;
}

/* setup new ompc_proc: master is always at first proc table */
static struct ompc_proc *
ompc_new_proc(int i)
{
    struct ompc_proc *p = &ompc_procs[i];
    p->pid = _OMPC_PROC_SELF;

    OMPC_PROC_LOCK();
    ompc_proc_counter++;
    OMPC_PROC_UNLOCK();

    return p;
}

/*static*/ struct ompc_thread *
ompc_current_thread()
{
    struct ompc_thread *tp;
    ABT_key_get(tls_key, (void **)&tp);
    return tp;
}

/* get thread from free list */
static struct ompc_proc *
ompc_get_proc(struct ompc_thread *par, struct ompc_thread *cur,
              int thread_num, int num_threads)
{
///*
    int es_start = par->es_start;
    int es_length = par->es_length;

    if (num_threads > es_length) {
        cur->es_start = es_start + (thread_num % es_length);
        cur->es_length = 1;
    }
    else {
        int chunk_size = es_length / num_threads;
        cur->es_start = es_start + (thread_num * chunk_size);
        cur->es_length = chunk_size;
    }

    // FIXME for debug
    // printf("par[%d:%d] -> cur[%d:%d] | %d th thread in %d threads\n", es_start, es_length, cur->es_start, cur->es_length, thread_num, num_threads);

    return &ompc_procs[cur->es_start];
//*/
/* round-robin
    struct ompc_proc *p;
    int i;

    OMPC_PROC_LOCK();
    if(++proc_last_used >= ompc_max_threads) proc_last_used = 0;
    p = &ompc_procs[proc_last_used];
    cur->es_start = proc_last_used;
    OMPC_PROC_UNLOCK();

    return p;
*/
}

/* allocate/get thread entry */
static struct ompc_thread *
ompc_alloc_thread(void)
{
    struct ompc_thread *p;

    p = (struct ompc_thread *)malloc(sizeof(struct ompc_thread));
    if (p == NULL) {
        ompc_fatal("ompc_alloc_thread: malloc failed");
    }

    p->parallel_nested_level = 0;
    p->es_start = 0;
    p->es_length = ompc_max_threads;
    p->set_num_thds = -1;

    return p;
}

static void ompc_xstream_setup(void *arg)
{
    int es_idx = (int)(size_t)arg;

#ifdef USE_LOG
    if (ompc_log_flag && (es_idx != 0)) {
      tlog_slave_init();
    }
#endif /* USE_LOG */

    ompc_new_proc(es_idx);
    thread_affinity_setup(es_idx);
}

static void ompc_thread_wrapper_func(void *arg)
{
    struct ompc_thread *cthd = (struct ompc_thread *)arg;
    ABT_key_set(tls_key, (void *)cthd);

    struct ompc_thread *tp = cthd->parent;

/*
#ifdef __ABTL_LOG_ENABLE
    int event_wrapper_wait;
    if (tp->parallel_nested_level == 1) event_wrapper_wait = ABTL_log_start(1);
#endif
*/

    if (!tp->run_children) {
        ABT_mutex_lock(tp->broadcast_mutex);
        while (!tp->run_children) {
            ABT_cond_wait(tp->broadcast_cond, tp->broadcast_mutex);
        }
        ABT_mutex_unlock(tp->broadcast_mutex);
    }

/*
#ifdef __ABTL_LOG_ENABLE
    if (tp->parallel_nested_level == 1) ABTL_log_end(event_wrapper_wait);
#endif
*/

#ifdef __ABTL_LOG_ENABLE
    int event_wrapper_func;
    if (tp->parallel_nested_level == __ABTL_LOG_LEVEL) {
        event_wrapper_func = ABTL_log_start(2 + __ABTL_LOG_LEVEL);
    }
#endif

# ifdef USE_LOG
    if(ompc_log_flag) tlog_parallel_IN(i);
# endif /* USE_LOG */

    if ( tp->nargs < 0) {
        /* call C function */
        if ( tp->args != NULL )
            (*tp->func)(tp->args, cthd);
        else
            (*tp->func)(cthd);
    } else {
        /* call Fortran function */
        ompc_call_fsub(tp);
    }

# ifdef USE_LOG
    if(ompc_log_flag) tlog_parallel_OUT(i);
# endif /* USE_LOG */

#ifdef __ABTL_LOG_ENABLE
    if (tp->parallel_nested_level == __ABTL_LOG_LEVEL) {
        ABTL_log_end(event_wrapper_func);
    }
#endif
}

/* called from compiled code. */
void
ompc_do_parallel_main (int nargs, int cond, int nthds,
    cfunc f, void *args)
{
    struct ompc_thread *cthd = ompc_current_thread();

/*
#ifdef __ABTL_LOG_ENABLE
    int event_parallel_exec;
    if (cthd->parallel_nested_level == 1) event_parallel_exec = ABTL_log_start(7);
#endif
*/

    int n_thds, in_parallel;
    if (cond == 0) { /* serialized by parallel if(false) */
        n_thds = 1;
        in_parallel = cthd->in_parallel;
    } else {
// FIXME old (pthread) impl
//        n_thds = (nthds < ompc_num_threads) ? (nthds) : (ompc_num_threads);
// FIXME temporary impl
// assume OMP_NESTED=TRUE
// read OpenMP specification
// nthds is not used, modify runtime API
        if ((cthd->set_num_thds) == -1) {
            n_thds = ompc_num_threads;
        }
        else {
            n_thds = cthd->set_num_thds;
        }
        in_parallel = 1;
    }

    /* initialize parent thread */
    cthd->num_thds = n_thds;
    cthd->nargs = nargs;
    cthd->args = args;
    cthd->func = f;

    /* initialize flag, mutex, and cond */
    cthd->run_children = 0;
    ABT_mutex_create(&cthd->broadcast_mutex);
    ABT_mutex_create(&cthd->reduction_mutex);
    ABT_cond_create(&cthd->broadcast_cond);
    ABT_cond_create(&cthd->reduction_cond);

    /* initialize barrier structure */
    cthd->out_count = 0;
    cthd->in_count = 0;
    for (int i = 0; i < n_thds; i++ ) {
        cthd->barrier_flags[i]._v = cthd->barrier_sense;
        cthd->in_flags[i]._v = 0;
    }
    
    ompc_tree_barrier_init(&cthd->tree_barrier, n_thds);

    ompc_thread_t *children = (ompc_thread_t *)malloc(sizeof(ompc_thread_t) * n_thds);

    /* assign thread to proc */
    for (int i = 0; i < n_thds; i++ ) {
        struct ompc_thread *tp = ompc_alloc_thread();
        struct ompc_proc *p = ompc_get_proc(cthd, tp, i, n_thds);
        tp->parent = cthd;
        tp->num = i;                        /* set thread_num */
        tp->in_parallel = in_parallel;
        tp->parallel_nested_level = cthd->parallel_nested_level + 1;
        tp->barrier_sense = 0;

        if (i == 0) {
            tp->nargs = nargs;
            tp->args = args;
        }

        ABT_thread_create_on_xstream((ABT_xstream)(p->pid), ompc_thread_wrapper_func,
            (void *)tp, ABT_THREAD_ATTR_NULL, (ABT_thread *)(&tp->tid));

        children[i] = tp->tid;
    }

    ABT_mutex_lock(cthd->broadcast_mutex);
    cthd->run_children = 1;
    ABT_cond_broadcast(cthd->broadcast_cond);
    ABT_mutex_unlock(cthd->broadcast_mutex);

/*
#ifdef __ABTL_LOG_ENABLE
    if (cthd->parallel_nested_level == 1) ABTL_log_end(event_parallel_exec);

    int event_parallel_join;
    if (cthd->parallel_nested_level == 1)  event_parallel_join = ABTL_log_start(9);
#endif
*/

    for (int i = 0; i < n_thds; i++) {
        ABT_thread_join(children[i]);
        ABT_thread_free(&children[i]);
    }

    ABT_cond_free(&cthd->broadcast_cond);
    ABT_cond_free(&cthd->reduction_cond);
    ABT_mutex_free(&cthd->broadcast_mutex);
    ABT_mutex_free(&cthd->reduction_mutex);
    free(children);
    
    ompc_tree_barrier_finalize(&cthd->tree_barrier);

    if (cthd->parent == NULL) {
        proc_last_used = 0;
    }

/*
#ifdef __ABTL_LOG_ENABLE
    if (cthd->parallel_nested_level == 1) ABTL_log_end(event_parallel_join);
#endif
*/
}


void
ompc_do_parallel(cfunc f, void *args)
{
    ompc_do_parallel_main (-1, 1, ompc_num_threads, f, args);
}

void
ompc_do_parallel_if (int cond, cfunc f, void *args)
{
    ompc_do_parallel_main (-1, cond, ompc_num_threads, f, args);
}


/* 
 * Barrier 
 */
void
ompc_thread_barrier(int id, struct ompc_thread *tpp)
{
    int sen0,n;

    if(tpp == NULL) return; /* not in parallel */
#ifdef USE_LOG
    if(ompc_log_flag) tlog_barrier_IN(id);
#endif // USE_LOG

#if 1  // USE_ARGOBOTS
    sen0 = ~tpp->barrier_sense;
    n = tpp->num_thds;

    if (id == 0) {
        for (int i = 1; i < n; i++) {
            OMPC_WAIT_UNTIL((volatile int)tpp->barrier_flags[i]._v == sen0,
                tpp->reduction_cond, tpp->reduction_mutex);
        }

        ABT_mutex_lock(tpp->broadcast_mutex);
        tpp->barrier_sense = sen0;
        ABT_cond_broadcast(tpp->broadcast_cond);
        ABT_mutex_unlock(tpp->broadcast_mutex);
    } else {
        ABT_mutex_lock(tpp->reduction_mutex);
        tpp->barrier_flags[id]._v = sen0;
        ABT_cond_signal(tpp->reduction_cond);
        ABT_mutex_unlock(tpp->reduction_mutex);

        OMPC_WAIT_UNTIL((volatile int)tpp->barrier_sense == sen0,
            tpp->broadcast_cond, tpp->broadcast_mutex);
    }
#else
    sen0 = tpp->barrier_sense ^ 1;
    n = tpp->num_thds;
    if (id == 0){
        int j;
        for ( j = 1 ; j < n ; j++ )
          OMPC_WAIT((volatile int)tpp->barrier_flags[j]._v != sen0);
        tpp->barrier_sense = sen0;
        MBAR();
    } else {
        tpp->barrier_flags[id]._v = sen0;
        MBAR();
        OMPC_WAIT ((volatile int)tpp->barrier_sense != sen0);
    }
#endif  // USE_ARGOBOTS

#ifdef USE_LOG
    if(ompc_log_flag) tlog_barrier_OUT(id);
#endif // USE_LOG
}

void
ompc_current_thread_barrier()
{
    int id;
    struct ompc_thread *tp = ompc_current_thread();
    struct ompc_thread *tpp = tp->parent;

    if(tpp == NULL)
        return;

    if(ompc_get_num_threads(tp) == 1) {
        id = 0;
    } else {
        id = tp->num;
    }

    ompc_thread_barrier(id, tpp);
}


void
ompc_terminate(int exitcode)
{
    for (int i = 1; i < ompc_max_threads; i++) {
        ABT_xstream_join(ompc_procs[i].pid);
        ABT_xstream_free((ABT_xstream *)&(ompc_procs[i].pid));
    }

#ifdef __TEST_WORK_STEALING
    // scheds[0] will be deallocated by the argobots runtime
    for (int i = 1; i < ompc_max_threads; i++) {
        ABT_sched_free(&scheds[i]);
    }
#endif

    free(ompc_procs);

    // FIXME this causes segmentation fault on ABT_finalize():
    // incorrect destructor pointer for primary ULT
    // ABT_key_free(&tls_key);

#ifdef __ABTL_LOG_ENABLE
    ABTL_dump_log();
#endif
    ABTL_finalize();
    ABT_finalize();

    hwloc_topology_destroy(topo);
    hwloc_bitmap_free(allset);

    exit (exitcode);
}


int
ompc_in_parallel (struct ompc_thread *tp)
{
    return tp->in_parallel;
}


int
ompc_get_num_threads (struct ompc_thread *tp)
{
    if((tp = tp->parent) == NULL)
        return 1;
    else 
        return tp->num_thds;
}

// FIXME this function is used for num_threads clause
// not for omp_set_num_threads()
// modify runtime API, e.g. omp_set_num_threads_clause()
void ompc_set_num_threads(int n) {
    // FIXME assumes that OMP_NESTED=TRUE
    // add error check
    struct ompc_thread *tp = ompc_current_thread();
    tp->set_num_thds = n;
}

int
ompc_get_thread_num()
{
    extern int omp_get_thread_num();
    return omp_get_thread_num();
}


int
ompc_get_max_threads()
{
    return ompc_max_threads;
}

ompc_proc_t
ompc_xstream_self()
{
    ABT_xstream xstream;
    ABT_xstream_self(&xstream);
    return xstream;
}

static ompc_thread_t
ompc_thread_self()
{
    ABT_thread thread;
    ABT_thread_self(&thread);
    return thread;
}
