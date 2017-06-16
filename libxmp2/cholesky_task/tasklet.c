/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tasklet.h"

#ifdef USE_ABT
#include <abt.h>
#endif

#define TASKLET_MAX_ARGS  20

typedef struct _tasklet_list
{
    struct _tasklet_list *next;
    struct _tasklet *tasklet;
    void *key_addr;
    int is_out;
} *Tasklet_list;

static Tasklet_list tasklet_list_free;

static Tasklet_list new_tasklet_list();
static void free_tasklet_list(Tasklet_list lp);

typedef struct _tasklet {
    struct _tasklet *next;  // for ready list and free list
    struct _tasklet_list *depends;
    unsigned int depend_count;  // input count
    unsigned int ref_count;  // reference count
    int done;  // already executed
    cfunc f;
    void *args[TASKLET_MAX_ARGS];
#ifdef USE_ABT
    ABT_mutex mutex;
#endif
} Tasklet;

static Tasklet *tasklet_free_list = NULL;

// for checking
static int tasklet_n_created = 0;
static int tasklet_n_executed = 0;
static int tasklet_n_reclaimed = 0;

#ifdef USE_ABT
#define _ENABLE_SHARED_POOL 1

static ABT_mutex tasklet_g_mutex;   // global lock
static ABT_mutex tasklet_count_mutex;
static ABT_cond tasklet_count_cond;
static int tasklet_running_count = 0;

int _xmp_num_xstreams = 4;  // number of execution stream
static ABT_xstream *_xmp_ess; // execution stream

#ifdef _ENABLE_SHARED_POOL
static ABT_pool _xmp_shared_pool;  // shared global pool
#else 
static int _xmp_num_pools;
static ABT_pool *_xmp_private_pools;
#endif

#else
static Tasklet *tasklet_ready_list = NULL;
#endif

static Tasklet *new_tasklet();
static void free_tasklet(Tasklet *tp);
static void tasklet_put_ready(Tasklet *tp);
static Tasklet *tasklet_get_ready();

static void tasklet_make_dependency(Tasklet *from_tp, Tasklet *to_tp);
static void tasklet_put_keyaddr(int index,Tasklet *tp,void *key_addr,int is_out);
static void tasklet_remove_keyaddr(int index,void *key_addr);

static void tasklet_g_lock();
static void tasklet_g_unlock();
static void tasklet_lock(Tasklet *tp);
static void tasklet_unlock(Tasklet *tp);
static void tasklet_ref_inc(Tasklet *tp);
static void tasklet_ref_dec(Tasklet *tp);

/* hash table for dependency */
#define N_HASH_TBL_SIZE (1 << 9)   /* 512: must be the power of 2 */
static Tasklet_list tasklet_hash_table[N_HASH_TBL_SIZE];

unsigned int tasklet_hash_index(void *a)
{
    unsigned int h = (unsigned long long int)(a);
    h = h >> 4;
    h = h + (h>>16);
    return h & (N_HASH_TBL_SIZE-1);
}
    
#define HASH_ADDR_INDEX(a)  ((a)) & (N_DEPEND_TBL_SIZE-1))

void tasklet_create(cfunc f, int narg, void **args, 
                    int n_in, void **in_data, int n_out, void **out_data)
{
    Tasklet *tp;
    Tasklet_list lp;
    int i,h;
    void *key_addr;

    tasklet_g_lock();
    tp = new_tasklet();
    tp->ref_count = 1;  // just created
    tp->done = FALSE;
    tp->f = f;
    if(narg >= TASKLET_MAX_ARGS){
        fprintf(stderr,"too many task arg: %d > %d\n",narg,TASKLET_MAX_ARGS);
        exit(1);
    }
#ifdef USE_ABT
    ABT_mutex_create(&tp->mutex);
#endif
    for(i = 0; i < narg; i++) tp->args[i] = args[i];

    if(n_in > 0){
        // handling IN-dependency, search conresponding OUT
        for(i = 0; i < n_in; i++){
            key_addr = in_data[i];
            h = tasklet_hash_index(key_addr);
            for(lp = tasklet_hash_table[h]; lp != NULL; lp = lp->next){
                if(lp->is_out && lp->key_addr == key_addr){
                    tasklet_make_dependency(lp->tasklet,tp);
                    tasklet_put_keyaddr(h,tp,key_addr,FALSE);
                    break;
                }
            }
        }
    }
    if(n_out > 0){
        // handling OUT dependecy, serch IN and OUT
        for(i = 0; i < n_out; i++){
            key_addr = out_data[i];
            h = tasklet_hash_index(key_addr);
            for(lp = tasklet_hash_table[h]; lp != NULL; lp = lp->next){
                if(lp->key_addr == key_addr){
                    tasklet_make_dependency(lp->tasklet,tp);
                }
            }
            tasklet_remove_keyaddr(h,key_addr);
            tasklet_put_keyaddr(h,tp,key_addr,TRUE);
        }
    }
    tasklet_n_created++;
    tasklet_g_unlock();
    
    // if no dependecy, ready to execute
    if(tp->depend_count == 0) tasklet_put_ready(tp);
}

/**
 * Tasklet management
 */
static Tasklet *new_tasklet()
{
    Tasklet *tp;

    if((tp = tasklet_free_list) != NULL){
        tasklet_free_list = tasklet_free_list->next;
    } else {
        if((tp = (Tasklet *)malloc(sizeof(Tasklet))) == NULL){
            fprintf(stderr,"malloc failed: Tasklet\n");
            exit(1);
        }
    }
    memset((void *)tp,0,sizeof(*tp));
    return tp;
}

static void free_tasklet(Tasklet *tp)
{
    tp->next = tasklet_free_list;
    tasklet_free_list = tp;
#ifdef USE_ABT
    ABT_mutex_free(&tp->mutex);
#endif
}

/**
 * dependency management
 */
Tasklet_list new_tasklet_list()
{
    Tasklet_list lp;
    if((lp = tasklet_list_free) != NULL){
        tasklet_list_free = lp->next;
    } else {
        if((lp = (Tasklet_list)malloc(sizeof(*lp))) == NULL){
            fprintf(stderr,"malloc failed: Tasklet_list\n");
            exit(1);
        }
    }
    memset((void *)lp,0,sizeof(*lp));
    return lp;
}

static void free_tasklet_list(Tasklet_list lp)
{
    lp->next = tasklet_list_free;
    tasklet_list_free = lp;
}

static void tasklet_make_dependency(Tasklet *from_tp, Tasklet *to_tp)
{
    Tasklet_list lp;

    if(from_tp->done) return;

    lp = new_tasklet_list();
    lp->tasklet = to_tp;
    lp->next = from_tp->depends;
    from_tp->depends = lp;
    to_tp->depend_count++;
    tasklet_ref_inc(to_tp); // inc reference count
}

static void tasklet_put_keyaddr(int index,Tasklet *tp,void *key_addr,int is_out)
{
    Tasklet_list lp;

    lp = new_tasklet_list();
    lp->next = tasklet_hash_table[index];
    tasklet_hash_table[index] = lp;
    lp->tasklet = tp;
    lp->key_addr = key_addr;
    lp->is_out = is_out;
    tasklet_ref_inc(tp);  // inc reference count
}

static void tasklet_remove_keyaddr(int index,void *key_addr)
{
    Tasklet_list lp, lq, lpp;
    
    lp = tasklet_hash_table[index]; 
    lq = NULL;
    while(lp != NULL){
        if(lp->key_addr == key_addr){ // remove this entry
            lpp = lp;
            lp = lp->next;
            if(lq == NULL) // head
                tasklet_hash_table[index] = lp;
            else
                lq->next = lp;
            tasklet_ref_dec(lpp->tasklet); // dec reference count
            free_tasklet_list(lpp);
        } else {
            lq = lp;
            lp = lp->next;
        }
    }
}

static void tasklet_ref_inc(Tasklet *tp)
{
    tasklet_lock(tp);
    tp->ref_count++;
    tasklet_unlock(tp);
}

static void tasklet_ref_dec(Tasklet *tp)
{
    tasklet_lock(tp);
    tp->ref_count--;
    tasklet_unlock(tp);

    if(tp->ref_count == 0) {
        free_tasklet(tp);
        tasklet_n_reclaimed++;
    }
}

#ifdef USE_ABT

static void ABT_error(char *msg)
{
    fprintf(stderr,"ABT error: %s\n",msg);
    exit(1);
}

void tasklet_initialize(int argc, char *argv[])
{
    int i;

    /* initialization */
    ABT_init(argc, argv);

    char *cp;
    int val;
    cp = getenv("XMP_NUM_THREADS");
    if (cp != NULL) {
        sscanf(cp, "%d", &val);
        _xmp_num_xstreams = val;
    } else {
        _xmp_num_xstreams = 4;
    }
    printf("_xmp_num_xstreams = %d\n",_xmp_num_xstreams);

    _xmp_ess = (ABT_xstream *)malloc(sizeof(ABT_xstream) * _xmp_num_xstreams);

#ifdef _ENABLE_SHARED_POOL
    /* shared pool creation */
    ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE, &_xmp_shared_pool);

    ABT_xstream_self(&_xmp_ess[0]);
    ABT_xstream_set_main_sched_basic(_xmp_ess[0], ABT_SCHED_DEFAULT,1, &_xmp_shared_pool);
    for (i = 1; i < _xmp_num_xstreams; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &_xmp_shared_pool,
                                 ABT_SCHED_CONFIG_NULL, &_xmp_ess[i]);
        ABT_xstream_start(_xmp_ess[i]);
    }

    for (i = 0; i < _xmp_num_xstreams; i++) {
        int rank;
        ABT_xstream_get_rank(_xmp_ess[i],&rank);
    }
#else
    _xmp_num_pools = (_xmp_num_threads != 1) ? _xmp_num_threads-1 : 1;
    _xmp_private_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * _xmp_num_pools);

    ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE, &_xmp_private_pools[0]);
    for (int i = 1; i < _xmp_num_threads; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE, &_xmp_private_pools[i-1]);
    }
    ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &_xmp_private_pools[0], ABT_SCHED_CONFIG_NULL, 
                           &_xmp_scheds[0]);
    for (int i = 1; i < _xmp_num_threads; i++) {
        ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &_xmp_private_pools[i-1], 
                               ABT_SCHED_CONFIG_NULL, &_xmp_scheds[i]);
    }
#endif

    if(ABT_mutex_create(&tasklet_g_mutex) != ABT_SUCCESS) goto err;
    if(ABT_mutex_create(&tasklet_count_mutex) != ABT_SUCCESS) goto err;
    if(ABT_cond_create(&tasklet_count_cond) != ABT_SUCCESS) goto err;
    tasklet_running_count = 0;

    return;
 err:
    ABT_error("tasklet_initialize");
}

void tasklet_exec_main(cfunc main_func)
{
    ABT_thread thread;
    int dummy;

    /* lauch main proram */
    ABT_thread_create(_xmp_shared_pool, (void (*)(void *))main_func, (void *)&dummy, 
                      ABT_THREAD_ATTR_NULL, &thread);

    /* wait and join the thread */
    ABT_thread_join(thread);
    ABT_thread_free(&thread);

    printf("tasklet_exec_main: main_func end created=%d executed=%d reclaimed=%d ...\n",
           tasklet_n_created, tasklet_n_executed, tasklet_n_reclaimed);
}

void tasklet_finalize()
{
    int i;
    /* join ESs */
    for (i = 1; i < _xmp_num_xstreams; i++){
        ABT_xstream_join(_xmp_ess[i]);
        ABT_xstream_free(&_xmp_ess[i]);
    }

    ABT_finalize();
}

static void tasklet_wrapper(Tasklet *tp)
{
    Tasklet *tq;
    Tasklet_list lp, lq;

    (*tp->f)(tp->args);  // exec tasklet

    tasklet_g_lock();
    tp->done = TRUE;  // executed
    // resolve dependency
    for(lp = tp->depends; lp != NULL; lp = lp->next){
        tq = lp->tasklet;

        tasklet_lock(tq);
        tq->depend_count--;
        // printf("tasklet(%p) depend_count %d\n",tq,tq->depend_count);
        assert(tq->depend_count >= 0);
        if(tq->depend_count < 0){
            fprintf(stderr,"depend_count (%d) < 0\n",tq->depend_count);
            exit(1);
        }
        tq->ref_count--;
        if(tq->ref_count <= 0){ // must be > 0 
            fprintf(stderr,"ref_count (%d) <= 0\n",tq->ref_count);
            exit(1);
        }
        tasklet_unlock(tq);
        if(tq->depend_count == 0) tasklet_put_ready(tq);
    }

    lp = tp->depends;
    while(lp != NULL){
        lq = lp->next;
        free_tasklet_list(lp);
        lp = lq;
    }
    tasklet_n_executed++;
    tasklet_g_unlock();

    tasklet_ref_dec(tp);  // dec refcount, executed

    ABT_mutex_lock(tasklet_count_mutex);
    tasklet_running_count--;
    if(tasklet_running_count == 0) 
        ABT_cond_signal(tasklet_count_cond);
    ABT_mutex_unlock(tasklet_count_mutex);
}

static void tasklet_put_ready(Tasklet *tp)
{
    /* lauch main proram */
    if(ABT_mutex_lock(tasklet_count_mutex) != ABT_SUCCESS) goto err;
    tasklet_running_count++;
    if(ABT_mutex_unlock(tasklet_count_mutex) != ABT_SUCCESS) goto err;

    ABT_thread_create(_xmp_shared_pool, (void (*)(void *))tasklet_wrapper, tp, ABT_THREAD_ATTR_NULL, NULL);
    
    return;
 err:
    ABT_error("task_put_ready");
}

void tasklet_wait_all()
{
    if(ABT_mutex_lock(tasklet_count_mutex) != ABT_SUCCESS) goto err;
    while(tasklet_running_count != 0 || tasklet_n_executed < tasklet_n_created) {
        if(ABT_cond_wait(tasklet_count_cond,tasklet_count_mutex) != ABT_SUCCESS) goto err;
    }
    if(ABT_mutex_unlock(tasklet_count_mutex) != ABT_SUCCESS) goto err;

    // clean up hash table
    for(int h = 0; h < N_HASH_TBL_SIZE; h++) {
        Tasklet_list lp;
        for(lp = tasklet_hash_table[h]; lp != NULL; lp = lp-> next)
            tasklet_ref_dec(lp->tasklet);
    }
    
    // printf("tasklet_wait_all end: taslet_running count=%d\n",tasklet_running_count);
    return;
 err:
    ABT_error("task_wait_all");
}

static void tasklet_g_lock()
{
    if(ABT_mutex_lock(tasklet_g_mutex) != ABT_SUCCESS) 
        ABT_error("tasklet_g_lock");
}

static void tasklet_g_unlock()
{
    if(ABT_mutex_unlock(tasklet_g_mutex) != ABT_SUCCESS) 
        ABT_error("tasklet_g_unlock");
}

static void tasklet_lock(Tasklet *tp)
{
    if(ABT_mutex_lock(tp->mutex) != ABT_SUCCESS) 
        ABT_error("taslet_lock");
}

static void tasklet_unlock(Tasklet *tp)
{
    if(ABT_mutex_unlock(tp->mutex) != ABT_SUCCESS)
        ABT_error("tasklet_unlock");
}

#else

void tasklet_initialize(int argc, char *argv[])
{
    /* nothing */
}

void tasklet_exec_main(cfunc main_func)
{
    (*main_func)(NULL);

    printf("tasklet_exec_main: main_func end created=%d executed=%d...\n",
           tasklet_n_created, tasklet_n_executed);
}

void tasklet_finalize()
{
    /* nothing */
}

void tasklet_wait_all()
{
    Tasklet *tp, *tq;
    Tasklet_list lp, lq;

    while(tasklet_ready_list != NULL){
        tp = tasklet_get_ready();
        // printf("exec tasklet(%p)\n",tp);

        (*tp->f)(tp->args);  // exec tasklet

        // resolve dependency
        lp = tp->depends; 
        while(lp != NULL){
            tq = lp->tasklet;
            tq->depend_count--;
            // printf("tasklet(%p) depend_count %d\n",tq,tq->depend_count);
            assert(tq->depend_count >= 0);
            if(tq->depend_count < 0){
                fprintf(stderr,"depend_count (%d) < 0\n",tq->depend_count);
                exit(1);
            }
            if(tq->depend_count == 0) tasklet_put_ready(tq);
            lq = lp->next;
            free_tasklet_list(lp);
            lp = lq;
            tasklet_n_executed++;
        }

        free_tasklet(tp);
    }
}

static void tasklet_put_ready(Tasklet *tp)
{
    tp->next = tasklet_ready_list;
    tasklet_ready_list = tp;
}

static Tasklet *tasklet_get_ready()
{
    Tasklet *tp = tasklet_ready_list;
    if(tp != NULL) tasklet_ready_list = tp->next;
    return tp;
}

static void tasklet_g_lock()
{
    /* nothing */
}

static void tasklet_g_unlock()
{
    /* nothing */
}

static void tasklet_lock(Tasklet *tp)
{
    /* nothing */
}

static void tasklet_unlock(Tasklet *tp)
{
    /* nothing */
}

#endif
