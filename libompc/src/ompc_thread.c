/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_thread.c
 */

//#define __TEST_WORK_STEALING  // FIXME do not use these now
//#define __OMNI_TEST_TASKLET__

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "exc_platform.h"
#include "ompclib.h"

#include <hwloc.h>
#include <errno.h>

#define ULT_POOL_SIZE 1024
#define TASKLET_POOL_SIZE 1024

// FIXME temporary impl, needs refactoring
static ABT_xstream              xstreams[MAX_PROC];
static ABT_pool                 pools[MAX_PROC];
static struct ompc_ult_pool     ult_pools[MAX_PROC];
static struct ompc_tasklet_pool tasklet_pools[MAX_PROC];

static ABT_key tls_key;
static void tls_free(void *value) {
    // do nothing
}

static hwloc_topology_t topo;
static int hwloc_ncores;
static void thread_affinity_setup(int i) {
    int thread_num;
    if (ompc_max_threads >= hwloc_ncores) {
        thread_num = i % hwloc_ncores;
    }
    else { // ompc_max_threads < hwloc_ncores
        int chunk_size = hwloc_ncores / ompc_max_threads;
        thread_num = i * chunk_size;
    }

    hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, thread_num);
    hwloc_cpuset_t set = hwloc_bitmap_dup(core->cpuset);
//    hwloc_bitmap_singlify(set);

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

static ompc_proc_t ompc_master_proc_id;

/* prototype */
static void ompc_xstream_setup();
static void ompc_thread_wrapper_func(void *arg);
static struct ompc_proc *ompc_new_proc(int i);
static int ompc_get_ES_num(struct ompc_thread *par, struct ompc_thread *cur,
                           int thread_num, int num_threads);
static void ompc_start_ult(ABT_pool pool, int es_num, void (*thread_func)(void *),
               void *arg, ABT_thread **newthread);
static void ompc_start_tasklet(struct ompc_thread *tp, void (*thread_func)(void *));
static void ompc_end_ult(ABT_thread *thread, int es_num);
static void ompc_end_tasklet(struct ompc_thread *tp);
static void ompc_init_thread(struct ompc_thread *tp);
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
    hwloc_ncores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    
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
    // FIXME variable pools is used for main pools for ESs
    // work stealing scheduler setup
    for (int i = 0; i < ompc_max_threads; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                              ABT_TRUE, &pools[i]);
    }

    create_scheds();
#endif

    for (int i = 0; i < ompc_max_threads; i++) {
        ABT_pool_create_basic(ABT_POOL_DEQUE, ABT_POOL_ACCESS_SPMC, ABT_TRUE, &pools[i]);
    }

    // ES setup
    if (ompc_debug_flag) fprintf(stderr, "Creating %d slave thread ...\n", ompc_max_threads - 1);
    for (int i = 0; i < ompc_max_threads; i++) {
        if (ompc_debug_flag) fprintf(stderr, "Creating slave %d  ...\n", i);

        ult_pools[i].ult_list = (ABT_thread *)malloc(sizeof(ABT_thread) * ULT_POOL_SIZE);
        ult_pools[i].idle_ult_list = (ABT_thread **)malloc(sizeof(ABT_thread *) * ULT_POOL_SIZE);
        ult_pools[i].size_allocated = ULT_POOL_SIZE;
        ult_pools[i].size_created = 0;
        ult_pools[i].size_idle = 0;

        tasklet_pools[i].tasklet_list = (ABT_task *)malloc(sizeof(ABT_task) * TASKLET_POOL_SIZE);
        tasklet_pools[i].size_allocated = TASKLET_POOL_SIZE;
        tasklet_pools[i].size_created = 0;
        tasklet_pools[i].size_used = 0;

        if (i == 0) {
            ABT_xstream_self(&xstreams[0]);
            ABT_xstream_set_main_sched_basic(xstreams[0], ABT_SCHED_RANDWS, ompc_max_threads, pools);
            ompc_xstream_setup(0, xstreams[0]);
            continue;
        }

        ABT_pool tmp;
        tmp = pools[0];
        pools[0] = pools[i];
        pools[i] = tmp;
        int res = ABT_xstream_create_basic(ABT_SCHED_RANDWS, ompc_max_threads, pools, ABT_SCHED_CONFIG_NULL, &xstreams[i]);
        if (res) {
            extern int errno;
            fprintf(stderr, "thread create fails at id %d:%d errno=%d\n", i, res, errno);
            perror("thread creation");
            exit(1);
        }

        ompc_xstream_setup(i, xstreams[i]);
    }

#ifdef __TEST_WORK_STEALING
    for (int i = 0; i < ompc_max_threads; i++) {
        if (i == 0) {
            sched_setup(0);
            continue;
        }

        size_t tid = (size_t)i;
        ABT_thread_create(pools[i], sched_setup,
                          (void *)tid, ABT_THREAD_ATTR_NULL, &(threads[i]));
    }

    for (int i = 1; i < ompc_max_threads; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }
#endif // __TEST_WORK_STEALING

    // setup master root thread
    struct ompc_thread *tp = (struct ompc_thread *)malloc(sizeof(struct ompc_thread));
    ompc_init_thread(tp);
    tp->num             = 0;    /* team master */
    tp->in_parallel     = 0;
    tp->parent          = NULL;
    tp->implicit_task.child_task_ptrs = NULL;
    tp->implicit_task.child_task_count = 0;
    tp->implicit_task.child_task_capacity = 0;
    tp->implicit_task.depth = 0;
    tp->implicit_task.es_num = 0;
    tp->implicit_task.avail_es_count = ompc_max_threads;
    ABT_key_set(tls_key, (void *)&tp->implicit_task);
    current_thread = tp;

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
ompc_new_proc(int i, ABT_xstream xstream)
{
    struct ompc_proc *p = &ompc_procs[i];
    p->pid = xstream;

    ompc_proc_counter++;

    return p;
}

static struct ompc_task *
ompc_current_task()
{
    struct ompc_task *task;
    ABT_key_get(tls_key, (void **)&task);
    return task;
}

struct ompc_thread *
ompc_current_thread()
{
    return current_thread;
}

static int
ompc_get_ES_num(struct ompc_thread *par, struct ompc_thread *cur,
                int thread_num, int num_threads)
{
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

    return cur->es_start;
}

static void
ompc_start_ult(ABT_pool pool,
               int es_num,
               void (*thread_func)(void *),
               void *arg,
               ABT_thread **newthread)
{
    struct ompc_ult_pool *ult_pool = &(ult_pools[es_num]);
    ABT_thread *ult_ptr;
    if (ult_pool->size_idle == 0) {
        if (ult_pool->size_created == ULT_POOL_SIZE) {
            ompc_fatal("cannot create new ULT");
        }

        int idx = ult_pool->size_created;
        ult_ptr = &(ult_pool->ult_list[idx]);
        ABT_thread_create(pool, thread_func,
                          (void *)arg, ABT_THREAD_ATTR_NULL, ult_ptr);
        (ult_pool->size_created)++;
    }
    else { // ult_pool->size_idle > 0
        int idx = ult_pool->size_idle - 1;
        ult_ptr = ult_pool->idle_ult_list[idx];
        (ult_pool->size_idle)--;
        ABT_thread_revive(pool, thread_func,
                          (void *)arg, ult_ptr);
    }

    *newthread = ult_ptr;
}

static void
ompc_start_tasklet(struct ompc_thread *tp,
                   void (*thread_func)(void *))
{
    int ES_num = tp->es_start;
    struct ompc_tasklet_pool *tasklet_pool = &(tasklet_pools[ES_num]);
    ABT_task *tasklet_ptr;
    if (tasklet_pool->size_created == tasklet_pool->size_used) {
        if (tasklet_pool->size_created == TASKLET_POOL_SIZE) {
            ompc_fatal("cannot create new Tasklet");
        }

        int idx = tasklet_pool->size_created;
        tasklet_ptr = &(tasklet_pool->tasklet_list[idx]);
        ABT_task_create(pools[ES_num], thread_func,
                        (void *)tp, tasklet_ptr);
        (tasklet_pool->size_created)++;
        (tasklet_pool->size_used)++;
    }
    else { // tasklet_pool->size_created > tasklet_pool->size_used
        int idx = tasklet_pool->size_used;
        tasklet_ptr = &(tasklet_pool->tasklet_list[idx]);
        ABT_task_revive(pools[ES_num], thread_func,
                        (void *)tp, tasklet_ptr);
        (tasklet_pool->size_used)++;
    }

    tp->tasklet_ptr = tasklet_ptr;
}

static void
ompc_end_ult(ABT_thread *thread, int es_num)
{
    struct ompc_ult_pool *ult_pool = &(ult_pools[es_num]);

    int idle_idx = ult_pool->size_idle;
    ult_pool->idle_ult_list[idle_idx] = thread;
    (ult_pool->size_idle)++;
}

static void
ompc_end_tasklet(struct ompc_thread *tp)
{
    ABT_task_state state;
    do {
        ABT_task_get_state(*(tp->tasklet_ptr), &state);
        ABT_thread_yield();
    } while (state != ABT_TASK_STATE_TERMINATED);
    (tasklet_pools[tp->es_start].size_used)--;
}

static void ompc_init_thread(struct ompc_thread *p) {
    p->parallel_nested_level = 0;
    p->es_start = 0;
    p->es_length = ompc_max_threads;
    p->set_num_thds = -1;
}

static void ompc_xstream_setup(int es_idx, ABT_xstream xstream)
{
#ifdef USE_LOG
    if (ompc_log_flag && (es_idx != 0)) {
      tlog_slave_init();
    }
#endif /* USE_LOG */

    ompc_new_proc(es_idx, xstream);
    thread_affinity_setup(es_idx);
}

struct divconq_args
{
    cfunc func;
    void *args;
    uint64_t lower;
    uint64_t upper;
    int step;
    int num_tasks;
};

static void divide_conquer_wrapper(struct ompc_task *curr_task);

static void loop_divide_conquer_impl(struct divconq_args *a)
{
    if (a->upper - a->lower <= 1 || a->num_tasks == 1) {
        a->func(a->lower, a->upper, a->step, a->args);
    } else {
        struct divconq_args a_other = {
            .func = a->func,
            .args = a->args,
            .lower = a->lower + (a->upper - a->lower) / 2
            .upper = a->upper,
            .step = a->step,
            .num_tasks = a->num_tasks / 2
        };

        a->upper = a_other.lower;
        a->num_tasks -= a_other.num_tasks;
        
        ABT_xstream self;
        ABT_xstream_self(&self);
        ABT_thread thread_other;
        ABT_thread_create_on_xstream(self, divide_conquer_impl, &a_other, ABT_THREAD_ATTR_NULL, &thread_other);
        loop_divide_conquer_impl(a);
        ABT_thread_join(thread_other);
        ABT_thread_free(&thread_other);
    }
}

// num_tasks == 0 means no constraint
void ompc_loop_divide_conquer(cfunc func, int nargs, void *args,
                              uint64_t lower, uint64_t upper, int step, int num_tasks)
{
    struct divconq_args a = {
        .func = func,
        .args = args,
        .lower = lower,
        .upper = upper,
        .step = step,
        .num_tasks = num_tasks == 0 ? ompc_max_threads * 8 : num_tasks
    };

    loop_divide_conquer_impl(&a);
}

static void ompc_thread_wrapper_func(void *arg)
{
    struct ompc_thread *cthd = (struct ompc_thread *)arg;
    ABT_key_set(tls_key, (void *)&cthd->implicit_task);

    struct ompc_thread *tp = cthd->parent;

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

    ABT_sched_finish(current_thread->scheduler);
}

/* called from compiled code. */
void
ompc_do_parallel_main (int nargs, int cond, int nthds,
    cfunc f, void *args)
{
    struct ompc_thread *cthd = ompc_current_thread();

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

    /* initialize barrier structure */
    cthd->out_count = 0;
    cthd->in_count = 0;
    for (int i = 0; i < n_thds; i++ ) {
        cthd->in_flags[i]._v = 0;
    }

    ompc_tree_barrier_init(&cthd->tree_barrier, n_thds);

    ABT_pool sched_pools[2];  // 0: private pool, 1: shared pool
    ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE, &sched_pools[1]);

    struct ompc_thread *tp_list = (struct ompc_thread *)malloc(sizeof(struct ompc_thread) * n_thds);
    /* assign thread to proc */
    for (int i = n_thds - 1; i >= 0; --i) {
        struct ompc_thread *tp = &(tp_list[i]);
        ompc_init_thread(tp);
        ompc_get_ES_num(cthd, tp, i, n_thds);
        tp->parent = cthd;
        tp->num = i;                        /* set thread_num */
        tp->in_parallel = in_parallel;
        tp->parallel_nested_level = cthd->parallel_nested_level + 1;
        tp->barrier_sense = 0;

#ifdef __OMNI_TEST_TASKLET__
        if (cthd->parallel_nested_level == 0) {
            ompc_start_ult(tp, ompc_thread_wrapper_func);
        }
        else {
            ompc_start_tasklet(tp, ompc_thread_wrapper_func);
        }
#else
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPSC,
                              ABT_TRUE, &sched_pools[0]);
        ABT_sched_create(&task_sched_def, 2, sched_pools, ABT_SCHED_CONFIG_NULL, &tp->scheduler);
        ompc_event_init(&tp->sched_finished);
        ABT_sched_set_data(tp->scheduler, tp);

        tp->implicit_task.child_task_ptrs = NULL;
        tp->implicit_task.child_task_count = 0;
        tp->implicit_task.child_task_capacity = 0;
        ompc_start_ult(sched_pools[0], tp->es_start, ompc_thread_wrapper_func, tp, &tp->ult_ptr);
        ABT_pool_add_sched(pools[tp->es_start], tp->scheduler);
#endif
    }

    for (int i = 0; i < n_thds; i++) {
        struct ompc_thread *tp = &(tp_list[i]);
#ifdef __OMNI_TEST_TASKLET__
        if (cthd->parallel_nested_level == 0) {
            ompc_end_ult(tp);
        }
        else {
            ompc_end_tasklet(tp);
        }
#else
        ompc_event_wait(&tp_list[i].sched_finished);
        ompc_event_finalize(&tp_list[i].sched_finished);
        ompc_end_ult(tp->ult_ptr, tp->es_start);
#endif
    }

    free(tp_list);

    ompc_tree_barrier_finalize(&cthd->tree_barrier);
}

static void
task_wrapper_func(void *arg)
{
    ABT_thread thread;
    ABT_thread_self(&thread);
    struct ompc_task *task = (struct ompc_task *)arg;
    ABT_key_set(tls_key, task);
    
    if (task->args != NULL) {
        task->func(task->args);
    } else {
        task->func();
    }

    for (int i = 0; i < task->child_task_count; i++) {
        ABT_thread_join(*task->child_task_ptrs[i]);
        ompc_end_ult(task->child_task_ptrs[i], current_thread->es_start);
    }
    free(task->child_task_ptrs);
    free(task->args);
    free(task);
}

static void
do_task_main(int nargs, _Bool cond, cfunc func, void *args, _Bool tied)
{
    struct ompc_task *task = malloc(sizeof(struct ompc_task));
    task->func = func;
    task->nargs = nargs;
    void *args_dup = malloc(nargs * sizeof(void *));
    memcpy(args_dup, args, nargs * sizeof(void *));
    task->args = args_dup;
    task->child_task_ptrs = NULL;
    task->child_task_count = 0;
    task->child_task_capacity = 0;

    ABT_pool shared_pool;
    ABT_thread *task_thread_ptr;
    ABT_sched_get_pools(current_thread->scheduler, 1, 1, &shared_pool);
    ompc_start_ult(shared_pool, current_thread->es_start, task_wrapper_func,
                   task, &task_thread_ptr);

    if (cond) {
        struct ompc_task *curr_task = ompc_current_task();
        if (curr_task->child_task_count == curr_task->child_task_capacity) {
            curr_task->child_task_capacity = curr_task->child_task_capacity == 0
                ? 10 : curr_task->child_task_capacity * 2;
            curr_task->child_task_ptrs = realloc(curr_task->child_task_ptrs,
                curr_task->child_task_capacity * sizeof(ABT_thread *));
        }
        curr_task->child_task_ptrs[curr_task->child_task_count++] = task_thread_ptr;
    } else {
        ABT_thread_join(*task_thread_ptr);
        ompc_end_ult(task_thread_ptr, current_thread->es_start);
    }
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

void
ompc_do_task(int nargs, cfunc func, void *args, _Bool tied)
{
    do_task_main(nargs, 1, func, args, tied);
}

void
ompc_do_task_if(int nargs, _Bool cond, cfunc func, void *args, _Bool tied)
{
    do_task_main(nargs, cond, func, args, tied);
}

void
ompc_taskwait()
{
    struct ompc_task *task = ompc_current_task();
    for (int i = 0; i < task->child_task_count; i++) {
        ABT_thread_join(*task->child_task_ptrs[i]);
        ompc_end_ult(task->child_task_ptrs[i], current_thread->es_start);
    }
    task->child_task_count = 0;
}

void
ompc_taskyield()
{
    ABT_thread_yield();
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

    ompc_tree_barrier_wait(&tpp->tree_barrier, tp);
}


void
ompc_terminate(int exitcode)
{
//     for (int i = 1; i < ompc_max_threads; i++) {
//         ABT_xstream_join(ompc_procs[i].pid);
//         ABT_xstream_free((ABT_xstream *)&(ompc_procs[i].pid));
//     }

// #ifdef __TEST_WORK_STEALING
//     // scheds[0] will be deallocated by the argobots runtime
//     for (int i = 1; i < ompc_max_threads; i++) {
//         ABT_sched_free(&scheds[i]);
//     }
// #endif

//     free(ompc_procs);

//     // removing master root thread
//     struct ompc_thread *tp = ompc_current_thread();
//     free(tp);

//     ABT_key_free(&tls_key);

//     for (int i = 0; i < ompc_max_threads; i++) {
//         for (int j = 0; j < ult_pools[i].size_created; j++) {
//             if (ult_pools[i].ult_list[j] != ABT_THREAD_NULL) {
//                 ABT_thread_free(&(ult_pools[i].ult_list[j]));
//             }
//         }
//         free(ult_pools[i].ult_list);
//         free(ult_pools[i].idle_ult_list);

//         for (int j = 0; j < tasklet_pools[i].size_created; j++) {
//             if (tasklet_pools[i].tasklet_list[j] != ABT_TASK_NULL) {
//                 ABT_task_free(&(tasklet_pools[i].tasklet_list[j]));
//             }
//         }
//         free(tasklet_pools[i].tasklet_list);
//     }

//     ABT_finalize();

    hwloc_topology_destroy(topo);

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

int ompc_get_proc_num(void)
{
    int rank;
    ABT_xstream_self_rank(&rank);
    return rank;
}

ompc_proc_t
ompc_xstream_self()
{
    ABT_xstream xstream;
    ABT_xstream_self(&xstream);
    return xstream;
}
