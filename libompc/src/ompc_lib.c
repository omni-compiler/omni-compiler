/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_lib.c
 */
#include <stdlib.h>
#include "ompclib.h"
#include "omp.h"

/*
 * OMP standard library function
 */
extern struct ompc_thread *ompc_current_thread();

int omp_get_thread_num()
{
    struct ompc_thread *tp;
    tp = ompc_current_thread();
    return tp->num;
}


int omp_get_num_threads()
{
    struct ompc_thread *tp;

    tp = ompc_current_thread();
    return ompc_get_num_threads (tp);
}


void omp_set_num_threads(int num)
{
#ifdef USE_PTHREAD_BARRIER
    extern pthread_barrier_t ompc_thd_bar;
#endif
    struct ompc_thread *tp;

    if(num <= 0){
        printf("omp_set_num_threads: argument must be positive.");
        ompc_fatal("ompc_set_num_threads");
    }
    tp = ompc_current_thread();
    if( tp->num != 0 )
        return;
    if ( ompc_max_threads < num ){
        printf("set fail: %d exceeds max parallel number %d\n", num, ompc_max_threads);
        ompc_fatal("ompc_set_num_threads");
    }
    /* printf(" -set a number of parallelism [%d]\n", num); */
    OMPC_PROC_LOCK();
    ompc_num_threads = num;
#ifdef USE_PTHREAD_BARRIER
    pthread_barrier_init(&ompc_thd_bar, 0, ompc_num_threads);
#endif
    OMPC_PROC_UNLOCK();
}


int omp_get_max_threads()
{
    return ompc_num_threads;
}


int omp_get_num_procs()
{
    return ompc_n_proc;
}


int omp_in_parallel()
{
    struct ompc_thread *tp;

    tp = ompc_current_thread();
    return ompc_in_parallel (tp);
}


void omp_set_dynamic(int dynamic_thds)
{
    ompc_dynamic = dynamic_thds;
}


int omp_get_dynamic()
{
    return 0 /*ompc_dynamic*/; /* not implmented */
}


void omp_set_nested(int n_nested)
{
    ompc_nested = n_nested;
}


int omp_get_nested()
{
    return ompc_nested; 
}



/*
 * Lock Functions
 */
void omp_init_lock(omp_lock_t *lock)
{
    ompc_lock_t *lp;
    OMPC_THREAD_LOCK();
    if((lp = (ompc_lock_t *)malloc(sizeof(ompc_lock_t))) == NULL)
        ompc_fatal("cannot allocate lock memory");
    ompc_init_lock(lp);
    OMPC_THREAD_UNLOCK();
    *lock = (omp_lock_t)lp;
}


void omp_init_nest_lock(omp_nest_lock_t *lock)
{
    ompc_nest_lock_t *lp;

    OMPC_THREAD_LOCK();
    if ((lp = (ompc_nest_lock_t *)malloc(sizeof(ompc_nest_lock_t))) == NULL)
        ompc_fatal("cannot allocate lock memory");
    ompc_init_nest_lock (lp);
    OMPC_THREAD_UNLOCK();
    *lock = (omp_nest_lock_t)lp;
}


void omp_destroy_lock(omp_lock_t *lock)
{
    ompc_destroy_lock((ompc_lock_t *)*lock);
    OMPC_THREAD_LOCK();
    free((ompc_lock_t *)*lock);
    OMPC_THREAD_UNLOCK();
}


void omp_destroy_nest_lock(omp_nest_lock_t *lock)
{
    ompc_destroy_nest_lock((ompc_nest_lock_t *)*lock);
    OMPC_THREAD_LOCK();
    free((ompc_nest_lock_t *)*lock);
    OMPC_THREAD_UNLOCK();
}


void omp_set_lock(omp_lock_t *lock)
{
    ompc_lock((ompc_lock_t *)*lock);
}


void omp_set_nest_lock(omp_nest_lock_t *lock)
{
    ompc_nest_lock((ompc_nest_lock_t *)*lock);
}


void omp_unset_lock(omp_lock_t *lock)
{
    ompc_unlock((ompc_lock_t *)*lock);
}


void omp_unset_nest_lock(omp_nest_lock_t *lock)
{
    ompc_nest_unlock((ompc_nest_lock_t *)*lock);
}


int omp_test_lock(omp_lock_t *lock)
{
    return ompc_test_lock((ompc_lock_t *)*lock);
}


int omp_test_nest_lock(omp_nest_lock_t *lock)
{
    return ompc_test_nest_lock((ompc_nest_lock_t *)*lock);
}


/* 
 * Timer routine
 */
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

double omp_get_wtime()
{
    double t;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = (double)(tv.tv_sec) + ((double)(tv.tv_usec))/1.0e6;
    return t ;
}

double omp_get_wtick()
{
    double t1,t2;
    t1 = omp_get_wtime();
 again:
    t2 = omp_get_wtime();
    if(t1 == t2) goto again;
    return t2-t1;
}

/**
 * Fortran wrapper functions
 */

int omp_get_thread_num_() { return omp_get_thread_num(); }
int omp_get_num_threads_() { return omp_get_num_threads(); }
void omp_set_num_threads_(int *num) {  omp_set_num_threads(*num); }
int omp_get_max_threads_() { return omp_get_max_threads(); }
int omp_get_num_procs_() { return omp_get_num_procs(); }
int omp_in_parallel_() { return omp_in_parallel(); }
void omp_set_dynamic_(int *dynamic_thds) { omp_set_dynamic(*dynamic_thds); }
int omp_get_dynamic_() { return omp_get_dynamic(); }
void omp_set_nested_(int *n_nested){ omp_set_nested(*n_nested); }
int omp_get_nested_() { return omp_get_nested(); }

double omp_get_wtime_() { return omp_get_wtime(); }
double omp_get_wtick_() { return omp_get_wtick(); }

void omp_init_lock_(omp_nest_lock_t *lock) { omp_init_lock(lock); }
void omp_init_nest_lock_(omp_nest_lock_t *lock) { omp_init_nest_lock(lock); }
void omp_destroy_lock_(omp_lock_t *lock) { omp_destroy_nest_lock(lock); }
void omp_destroy_nest_lock_(omp_nest_lock_t *lock) { omp_destroy_nest_lock(lock); }
void omp_set_lock_(omp_lock_t *lock) { omp_set_lock(lock); }
void omp_set_nest_lock_(omp_nest_lock_t *lock) { omp_set_nest_lock(lock); }
void omp_unset_lock_(omp_lock_t *lock) { omp_unset_lock(lock); }
void omp_unset_nest_lock_(omp_nest_lock_t *lock) { omp_unset_nest_lock(lock); }
int omp_test_lock_(omp_lock_t *lock) { return omp_test_lock(lock); }
int omp_test_nest_lock_(omp_nest_lock_t *lock) { return omp_test_nest_lock(lock); }

