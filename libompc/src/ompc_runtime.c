/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompc_runtime.c
 */
#include <stdlib.h>
#include "exc_platform.h"
#include "ompclib.h"

extern struct ompc_thread *ompc_current_thread(void);
static void ompc_static_bsched_tid(struct ompc_thread *tp, 
        indvar_t *lb, indvar_t *up, int ubofs, int *step);
static void ompc_static_csched_tid(struct ompc_thread *tp, 
        indvar_t *lb, indvar_t *up, int ubofs, int *step);
static void ompc_static_sched_init_tid(struct ompc_thread *tp,
        indvar_t lb, indvar_t up, int ubofs, int step, int chunk_size);
static void ompc_dynamic_sched_init_tid(struct ompc_thread *tp,
        indvar_t lb, indvar_t up, int ubofs, int step, int chunk_size);
static int ompc_static_sched_next_tid(struct ompc_thread *tp,
        indvar_t *lb, indvar_t *up, int ubofs);
static int ompc_dynamic_sched_next_tid(struct ompc_thread *tp,
        indvar_t *lb, indvar_t *up, int ubofs, int guided);
static int ompc_do_single_tid(struct ompc_thread *tp);


#define UBOFS_C 0


/*
 * compiler runtime
 */

int
ompc_thread_num_tid(struct ompc_thread *tp)
{
    return tp->num;
}


void
ompc_barrier()
{
    struct ompc_thread *tp;

    tp = ompc_current_thread();
    ompc_thread_barrier(tp->num, tp->parent);
}


void
ompc_barrier_tid(struct ompc_thread *tp)
{
    ompc_thread_barrier(tp->num, tp->parent);
}


void
_ompc_default_sched(indvar_t *lb, indvar_t *up, int ubofs, int *step)
{
    struct ompc_thread *tp = ompc_current_thread();
#ifdef DEBUG
    if(ompc_debug_flag)
        printf("default_sched: +id=%ld, lb=%ld, ub=%ld, step=%ld\n",
               (long)tp->num,(long)*lb,(long)*up,(long)*step);
#endif
    ompc_static_bsched_tid(tp, lb, up, ubofs, step);
#ifdef DEBUG
    if(ompc_debug_flag)
        printf("default_sched: -id=%ld, lb=%ld, ub=%ld, step=%ld\n",
               (long)tp->num,(long)*lb,(long)*up,(long)*step); 
#endif
}


void
ompc_default_sched(indvar_t *lb, indvar_t *up, int *step)
{
    _ompc_default_sched(lb, up, UBOFS_C, step);
}


/* static scheduling: cyclic */
void
_ompc_static_csched(indvar_t *lb, indvar_t *up, int ubofs, int *step)
{
    struct ompc_thread *tp = ompc_current_thread();
    ompc_static_csched_tid(tp, lb, up, ubofs, step);
}


void
ompc_static_csched(indvar_t *lb, indvar_t *up, int *step)
{
    _ompc_static_csched(lb, up, UBOFS_C, step);
}


void
ompc_static_csched_tid(struct ompc_thread *tp,
    indvar_t *lb, indvar_t *up, int ubofs, int *step)
{
    int n, s, n_thd, id;

    if ((n_thd = ompc_get_num_threads (tp)) == 1) { /* not in parallel, do nothing */
        return;
    }

#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_init_EVENT(tp->num);
#endif

    id = tp->num;
    tp->is_last = 0;
    s = *step;

    /* how many iteration */
    if(s > 0) n = ((char *)*up + ubofs - (char *)*lb + s - 1) / s;
    else n = ((char *)*up + ubofs - (char *)*lb + s + 1) / s;

    *lb = (char *)*lb + id * s;   /* adjust low bound */
    *step = s * n_thd;
    if(n > 0 && ((n - 1) % n_thd) == id) tp->is_last = 1;
}


/* static scheduling: block */
void
_ompc_static_bsched(indvar_t *lb, indvar_t *up, int ubofs, int *step)
{
    struct ompc_thread *tp = ompc_current_thread();
    ompc_static_bsched_tid(tp, lb, up, ubofs, step);
}


void
ompc_static_bsched(indvar_t *lb, indvar_t *up, int *step)
{
    _ompc_static_bsched(lb, up, UBOFS_C, step);
}


static void
ompc_static_bsched_tid(struct ompc_thread *tp,
    indvar_t *lb, indvar_t *up, int ubofs, int *step)
{
    indvar_t b, e, ee;
    int n_thd, id;
    int s, blk_s;

    if ((n_thd = ompc_get_num_threads (tp)) == 1) { /* not in parallel, do nothing */
      return;
    }

#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_init_EVENT(tp->num);
#endif

    s = *step;
    b = *lb;
    id = tp->num;
    tp->is_last = 0;

    if(s > 0){
        ee = e = (char *)*up + ubofs;
        blk_s = ((char *)e - (char *)b + n_thd - 1) / n_thd;
        blk_s = ((blk_s + s - 1) / s) * s;
        b = (char *)b + blk_s * id;
        e = (char *)b + blk_s;
        if((intptr_t)e >= (intptr_t)ee){
            e = ee;
            if(ee > b) tp->is_last = 1;
        }
        *up = (char *)e - ubofs;
    } else if(s < 0){
        ee = e = (char *)*up - ubofs;
        blk_s = ((char *)e - (char *)b - n_thd + 1) / n_thd;
        blk_s = ((blk_s + s + 1) / s) * s;
        b = (char *)b + blk_s * id;
        e = (char *)b + blk_s;
        if((intptr_t)e <= (intptr_t)ee){
            e = ee;
            if((intptr_t)ee < (intptr_t)b) tp->is_last = 1;
        }
        *up = (char *)e + ubofs;
    } else return;

    *lb = b;
}


/*
 * static schedule
 */
void
_ompc_static_sched_init(indvar_t lb, indvar_t up, int ubofs, int step,
    int chunk_size)
{
    struct ompc_thread *tp = ompc_current_thread();
    ompc_static_sched_init_tid(tp, lb, up, ubofs, step, chunk_size);
}


void
ompc_static_sched_init(indvar_t lb, indvar_t up, int step,
    int chunk_size)
{
    _ompc_static_sched_init(lb, up, UBOFS_C, step, chunk_size);
}


void
ompc_static_sched_init_tid(struct ompc_thread *tp,
    indvar_t lb, indvar_t up, int ubofs, int step, int chunk_size)
{
    int nthds;


    if ((nthds = ompc_get_num_threads (tp)) == 1) { /* not in parallel */
        tp->loop_sched_index = lb;
        tp->loop_end = (char *)up + ubofs;
        return;        
    }

#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_init_EVENT(tp->num);
#endif

    if(chunk_size <= 0){
        printf("check size is non-positive\n");
        ompc_fatal("ompc_static_sched_init");
    }

    chunk_size *= step;
    tp->loop_sched_index = (char *)lb + chunk_size * tp->num;
    tp->loop_chunk_size = chunk_size;
    tp->loop_stride = chunk_size * nthds;
    tp->loop_end = (char *)up + ubofs;
    tp->is_last = 0;
}


int
_ompc_static_sched_next(indvar_t *lb, indvar_t *up, int ubofs)
{
    struct ompc_thread *tp = ompc_current_thread();
    return ompc_static_sched_next_tid(tp, lb, up, ubofs);
}


int
ompc_static_sched_next(indvar_t *lb, indvar_t *up)
{
    return _ompc_static_sched_next(lb, up, UBOFS_C);
}


int
ompc_static_sched_next_tid(struct ompc_thread *tp,
    indvar_t *lb, indvar_t *up, int ubofs)
{
    indvar_t b,e;

    b = tp->loop_sched_index;
    if (ompc_get_num_threads (tp) == 1) { /* not in parallel */
        e = tp->loop_end;
        if(b == e) return FALSE;
        *lb = b;
        *up = (char *)e - ubofs;
        tp->loop_sched_index = e;
        return TRUE;
    }

#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_next_EVENT(tp->num);
#endif

    tp->loop_sched_index = (char*)tp->loop_sched_index + tp->loop_stride;
    e = (char *)b + tp->loop_chunk_size;

    if(tp->loop_chunk_size > 0){
        if(b >= tp->loop_end) return FALSE; 
        if(e >= tp->loop_end){
            e = tp->loop_end;
            tp->is_last = 1;
        }
    } else {
        if(b <= tp->loop_end) return FALSE;
        if(e <= tp->loop_end){
            e = tp->loop_end;
            tp->is_last = 1;
        }
    }

    *lb = b;
    *up = (char *)e - ubofs;
    return TRUE;
}


/* 
 * dynamic schedule
 */
void
_ompc_dynamic_sched_init(
    indvar_t lb, indvar_t up, int ubofs, int step, int chunk_size)
{
    struct ompc_thread *tp = ompc_current_thread();
    ompc_dynamic_sched_init_tid(tp, lb, up, ubofs, step, chunk_size);
}


void
ompc_dynamic_sched_init(
    indvar_t lb, indvar_t up, int step, int chunk_size)
{
    _ompc_dynamic_sched_init(lb, up, UBOFS_C, step, chunk_size);
}


void
ompc_dynamic_sched_init_tid(struct ompc_thread *tp,
    indvar_t lb, indvar_t up, int ubofs, int step, int chunk_size)
{
    struct ompc_thread *tpp;
    int id;

    if (ompc_get_num_threads (tp) == 1) { /* not in parallel */
        tp->loop_sched_index = lb;
        tp->loop_end = (char *)up + ubofs;
        return;        /* stride is not used */
    }
#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_init_EVENT(tp->num);
#endif
    if(chunk_size <= 0){
        printf("check size is non-positive\n");
        ompc_fatal("ompc_dynamic_sched_init");
    }
    tp->loop_chunk_size = chunk_size*step;
    tp->loop_end = (char *)up + ubofs;
    tp->loop_sched_index = lb;
    tp->is_last = 0;

    id  = tp->num;
    tpp = tp->parent;
    OMPC_WAIT((volatile int)tpp->in_flags[id]._v);
    OMPC_THREAD_LOCK();
    if(tpp->in_count == 0) tpp->dynamic_index = lb;
    tpp->in_count++;
    OMPC_THREAD_UNLOCK();
    tpp->in_flags[id]._v = 1;
}


int
_ompc_dynamic_sched_next(indvar_t *lb, indvar_t *up, int ubofs)
{
    struct ompc_thread *tp = ompc_current_thread();
    return ompc_dynamic_sched_next_tid(tp, lb, up, ubofs, FALSE);
}


int
ompc_dynamic_sched_next(indvar_t *lb, indvar_t *up)
{
    return _ompc_dynamic_sched_next(lb, up, UBOFS_C);
}


int
ompc_dynamic_sched_next_tid(struct ompc_thread *tp,
    indvar_t *lb, indvar_t *up, int ubofs, int guided)
{
    struct ompc_thread *tpp;
    int id,exit_flag;
    indvar_t b,e;
    int l,c;

    if (ompc_get_num_threads (tp) == 1) { /* not in parallel */
        b = tp->loop_sched_index;
        e = tp->loop_end;
        if(b == e) return FALSE;
        *lb = b;
        *up = (char *)e - ubofs;
        tp->loop_sched_index = e;
        return TRUE;
    } 
#ifdef USE_LOG
    if(ompc_log_flag) tlog_loop_next_EVENT(tp->num);
#endif
    c   = tp->loop_chunk_size;
    tpp = tp->parent;

    /* get my chunk, set b and e */
    OMPC_THREAD_LOCK();
    b = tpp->dynamic_index;
    if(guided){
      l = ((char *)tp->loop_end - (char *)b)/tpp->num_thds;
        l = ((l + c) / c) * c;
        if(c > 0){
            if(c > l) l = c;
        } else {
            if(c < l) l = c;
        }
        e = (char *)b + l;
    } else {
      e = (char *)b + c;
    }
    tpp->dynamic_index = e;
    OMPC_THREAD_UNLOCK();

    exit_flag = FALSE;
    if(c > 0){
        if(tp->loop_sched_index >= tp->loop_end) return FALSE;
        if(e >= tp->loop_end) exit_flag = TRUE;
    } else {
        if(tp->loop_sched_index <= tp->loop_end) return FALSE;
        if(e <= tp->loop_end) exit_flag = TRUE;
    }

    if(exit_flag){
        OMPC_THREAD_LOCK();
        tpp->out_count++;
        if(tpp->out_count == tpp->num_thds){   /* all thread exit */
            tpp->out_count = 0;
            tpp->in_count = 0;
            for(id = 0; id < tpp->num_thds; id++)
              tpp->in_flags[id]._v = 0;
            MBAR();
        }
        OMPC_THREAD_UNLOCK();
    }
    tp->loop_sched_index = e;

    /* adjust the last iteration */
    if(c > 0){
        if(b >= tp->loop_end) return FALSE; 
        if(e >= tp->loop_end){
            e = tp->loop_end;
            tp->is_last = 1;
        }
    } else {
        if(b <= tp->loop_end) return FALSE;
        if(e <= tp->loop_end){
            e = tp->loop_end;
            tp->is_last = 1;
        }
    }
    *lb = b;
    *up = (char *)e - ubofs;
    return TRUE;
}


/* 
 * guided schedule
 */
void
_ompc_guided_sched_init(indvar_t lb, indvar_t up, int ubofs,
    int step, int chunk_size)
{
    struct ompc_thread *tp = ompc_current_thread();
    ompc_dynamic_sched_init_tid(tp, lb, up, ubofs, step, chunk_size);
}


void
ompc_guided_sched_init(indvar_t lb, indvar_t up,
    int step, int chunk_size)
{
    _ompc_guided_sched_init(lb, up, UBOFS_C, step, chunk_size);
}


int
_ompc_guided_sched_next(indvar_t *lb, indvar_t *up, int ubofs)
{
    struct ompc_thread *tp = ompc_current_thread();
    return ompc_dynamic_sched_next_tid(tp, lb, up, ubofs, TRUE);
}


int
ompc_guided_sched_next(indvar_t *lb, indvar_t *up)
{
    return _ompc_guided_sched_next(lb, up, UBOFS_C);
}


/* 
 * runtime schedule 
 */
static enum { SCHED_NONE = 0, SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED }
ompc_runtime_sched_kind;
static int ompc_runtime_chunk_size = 0; /* default */

void
ompc_set_runtime_schedule(char *s)
{
    char *cp;
    cp = s;
    while(isspace((int)*cp)) cp++;
    if(*cp == 0) return;
    if(strncmp(cp,"static",6) == 0){
        cp += 6;
        ompc_runtime_sched_kind = SCHED_STATIC;
    } else if(strncmp(cp,"dynamic",7) == 0){
        cp += 7;
        ompc_runtime_sched_kind = SCHED_DYNAMIC;
    } else if(strncmp(cp,"guided",6) == 0){
        cp += 6;
        ompc_runtime_sched_kind = SCHED_GUIDED;
    }
    while(isspace((int)*cp)) cp++;
    if(*cp == 0) return;
    if(*cp != ',') goto err;
    cp++;
    while(isspace((int)*cp)) cp++;
    if(!isdigit((int)*cp)) goto err;
    sscanf(cp,"%d",&ompc_runtime_chunk_size);
    if(ompc_runtime_chunk_size <= 0){
        ompc_runtime_sched_kind = SCHED_NONE;
        goto err;
    }
    return;
err:
    fprintf(stderr,"OMP_SCHEDULE ='%s'",s);
    ompc_fatal("bad OMP_SCHEDULE");
}


void
_ompc_runtime_sched_init(indvar_t lb, indvar_t up, int ubofs, int step)
{
    struct ompc_thread *tp = ompc_current_thread();
    int chunk_size,n_thd;

    chunk_size = ompc_runtime_chunk_size;
    switch(ompc_runtime_sched_kind){
    case SCHED_DYNAMIC:
    case SCHED_GUIDED:
        if(chunk_size <= 0) chunk_size = 1;
        ompc_dynamic_sched_init_tid(tp, lb, up, ubofs, step, chunk_size);
        break;
    case SCHED_STATIC:
    case SCHED_NONE:
    default:
        n_thd = ompc_get_num_threads (tp);
        if(chunk_size <= 0){
	  chunk_size = ((char *)up + ubofs- (char *)lb) / (step * n_thd)
	    + ((((char *)up + ubofs - (char *)lb) % (step * n_thd)) ? 1 : 0);
            if(chunk_size <= 0) chunk_size = 1;
        }
        ompc_static_sched_init_tid(tp, lb, up, ubofs, step, chunk_size);
        break;
    }
}


void
ompc_runtime_sched_init(indvar_t lb, indvar_t up, int step)
{
    _ompc_runtime_sched_init(lb, up, UBOFS_C, step);
}


int
_ompc_runtime_sched_next(indvar_t *lb, indvar_t *up, int ubofs)
{
    struct ompc_thread *tp = ompc_current_thread();

    switch(ompc_runtime_sched_kind){
    case SCHED_DYNAMIC:
        return ompc_dynamic_sched_next_tid(tp, lb, up, ubofs, FALSE);
    case SCHED_GUIDED:
        return ompc_dynamic_sched_next_tid(tp, lb, up, ubofs, TRUE);
    case SCHED_STATIC:
    case SCHED_NONE:
    default:
        return ompc_static_sched_next_tid(tp, lb, up, ubofs);
    }
}


int
ompc_runtime_sched_next(indvar_t *lb, indvar_t *up)
{
    return _ompc_runtime_sched_next(lb, up, UBOFS_C);
}


/* 
 * ordered
 */
void
ompc_set_loop_id(indvar_t i)
{
    struct ompc_thread *tp = ompc_current_thread();
    tp->loop_id = i;
}


void
ompc_init_ordered(indvar_t lb, int step)
{
    int n,id;
    struct ompc_thread *tpp;
    struct ompc_thread *tp = ompc_current_thread();

    if ((n = ompc_get_num_threads (tp)) == 1) { /* not in parallel, do nothing */
      return;
    }
    tpp = tp->parent;
    
    id = tp->num;
    OMPC_WAIT((volatile int)tpp->in_flags[id]._v);
    OMPC_THREAD_LOCK();
    if(tpp->out_count == 0){ /* first visitor execute it */
        tpp->ordered_id = lb;
        tpp->ordered_step = step;
    }
    tpp->out_count++;
    if(tpp->out_count == n){
        /* if all threads comes, clear flags */
        for(id = 0; id < n; id++) tpp->in_flags[id]._v = 0;
        tpp->out_count = 0;
    } else tpp->in_flags[id]._v = 1;
    OMPC_THREAD_UNLOCK();
}


void
ompc_ordered_begin()
{
    struct ompc_thread *tpp;
    struct ompc_thread *tp = ompc_current_thread();
    
    if (ompc_get_num_threads (tp) == 1) { /* serialized */
      return;
    }
    tpp = tp->parent;
    OMPC_WAIT((volatile indvar_t)tp->loop_id != (volatile indvar_t)tpp->ordered_id);
}


void
ompc_ordered_end()
{
    struct ompc_thread *tpp;
    struct ompc_thread *tp = ompc_current_thread();

    if (ompc_get_num_threads (tp) == 1) { /* serialized */
      return;
    }
    tpp = tp->parent;
    tpp->ordered_id = (char*)tpp->ordered_id + tpp->ordered_step;
    MBAR();
}


/*
 * sections directives. section_id is allocated in round-robin manner.
 */
void
ompc_section_init(int n_sections)
{
    struct ompc_thread *tp;
    tp = ompc_current_thread();
    tp->section_id = tp->num;
    tp->last_section_id = n_sections - 1;
    tp->is_last = 0;
}


int
ompc_section_id()
{
    struct ompc_thread *tp;
    int id;

    tp = ompc_current_thread();
#ifdef USE_LOG
    if(ompc_log_flag) tlog_section_EVENT(tp->num);
#endif
    id = tp->section_id;
    tp->section_id += ompc_get_num_threads (tp);

    if (id == tp->last_section_id) {
      tp->is_last = 1;
    }
    return id;
}


int
ompc_is_last()
{
    struct ompc_thread *tp;
    tp = ompc_current_thread();
    return tp->is_last || (ompc_get_num_threads (tp) == 1);
}


/*
 * single construct
 */
int
ompc_do_single()
{
    return ompc_do_single_tid(ompc_current_thread());
}


int
ompc_do_single_tid(struct ompc_thread *tp)
{
    struct ompc_thread *tpp;
    int n,id;
    int ret = 0;

    if ((n = ompc_get_num_threads (tp)) == 1) { /* not in parallel */
        return 1;
    }

#ifdef USE_LOG
    if(ompc_log_flag) tlog_single_EVENT(tp->num);
#endif

    id  = tp->num;
    tpp = tp->parent;
    OMPC_WAIT((volatile int)tpp->in_flags[id]._v);
    OMPC_THREAD_LOCK();
    if(tpp->out_count == 0) ret = 1;        /* first visitor execute it */
    tpp->out_count++;
    if(tpp->out_count == n){
        /* if all threads comes, clear flags */
        for(id = 0; id < n; id++) tpp->in_flags[id]._v = 0;
        tpp->out_count = 0;
    } else tpp->in_flags[id]._v = 1;
    OMPC_THREAD_UNLOCK();
    return ret;
}


int
ompc_is_master()
{
    struct ompc_thread *tp;
    tp = ompc_current_thread();
    return tp->num == 0;
    /* return omp_get_thread_num() == 0; */
}


int
ompc_is_master_tid(struct ompc_thread *tp)
{
    return tp->num == 0;
}



static ompc_lock_t        _critical_lock;

void
ompc_critical_init ()
{
    ompc_init_lock (&_critical_lock);
}


void
ompc_critical_destroy ()
{
    ompc_destroy_lock (&_critical_lock);
}


void
ompc_enter_critical(ompc_lock_t **p)
{
#ifdef USE_LOG
    if(ompc_log_flag) tlog_critial_IN(ompc_current_thread()->num);
#endif
    if (*p == NULL) {
        ompc_lock (&_critical_lock);
        if ((ompc_lock_t volatile *)*p == NULL) {
            if((*p = (ompc_lock_t *)malloc(sizeof(ompc_lock_t))) == NULL) {
                ompc_fatal("cannot allocate lock memory");
            }
            ompc_init_lock (*p);
        }
        ompc_unlock (&_critical_lock);
    }
    ompc_lock((ompc_lock_t volatile *)*p);
}


void
ompc_exit_critical(ompc_lock_t **p)
{
    ompc_unlock(*p);
#ifdef USE_LOG
    if(ompc_log_flag) tlog_critial_OUT(ompc_current_thread()->num);
#endif
}


static ompc_lock_t _atomic_lock;

void
ompc_atomic_init_lock ()
{
    ompc_init_lock (&_atomic_lock);
}


void
ompc_atomic_lock()
{
    ompc_lock(&_atomic_lock);
}


void
ompc_atomic_unlock()
{
    ompc_unlock(&_atomic_lock);
}


void
ompc_thread_lock()
{
    OMPC_THREAD_LOCK();
}


void
ompc_thread_unlock()
{
    OMPC_THREAD_UNLOCK();
}


void
ompc_atomic_destroy_lock ()
{
    ompc_destroy_lock (&_atomic_lock);
}


void
ompc_bcopy(char *dst,char *src,int nbyte)
{
    bcopy(src,dst,nbyte);
}


void
ompc_flush()
{
    MBAR();
}


void *
ompc_get_thdprv(void ***thdprv_p,int size,void *datap)
{
    void **pp,*p;
    struct ompc_thread *tp;
    tp = ompc_current_thread();
    if((pp = *thdprv_p) == NULL){
        OMPC_THREAD_LOCK();
        if((pp = *thdprv_p) == NULL){
            pp = (void *)malloc(sizeof(void *)*ompc_max_threads);
            bzero(pp,sizeof(void *)*ompc_max_threads);
            if(pp == NULL) ompc_fatal("cannot allocate memory");
            *thdprv_p = pp;
        }
        OMPC_THREAD_UNLOCK();
    }
    if((p = pp[tp->num]) == NULL){
        if(tp->num == 0) p = datap;
        else {
            p = (void *)malloc(size);
        }
        pp[tp->num] = p;
    }
    return p;
}


void
ompc_copyin_thdprv(void *datap,void *global_datap,int size)
{
    if(global_datap != datap) bcopy(global_datap,datap,size);
    ompc_barrier();
}


/* 
 * reduction operation
 */
#define DO_REDUCTION_INTEGRAL(type_t,t) {\
    vals[id].r_v.t = *((type_t *)in_p); \
    if(tpp != NULL) ompc_thread_barrier(id,tpp); \
    if(id == 0) { \
        any_type v; int i; \
        v.t = *((type_t *)out_p); \
        switch(op){ \
        case OMPC_REDUCTION_PLUS: \
        case OMPC_REDUCTION_MINUS: \
            for(i = 0; i < n_thd; i++) v.t += vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_MUL: \
            for(i = 0; i < n_thd; i++) v.t *= vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_BITAND: \
            for(i = 0; i < n_thd; i++) v.t &= vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_BITOR: \
            for(i = 0; i < n_thd; i++) v.t |= vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_BITXOR: \
            for(i = 0; i < n_thd; i++) v.t ^= vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_LOGAND: \
            if(!v.t) break; \
            for(i = 0; i < n_thd; i++) \
                if(!vals[i].r_v.t) { v.t = 0; break; } \
            break; \
        case OMPC_REDUCTION_LOGOR: \
            if(v.t) break; \
            for(i = 0; i < n_thd; i++) \
                if(vals[i].r_v.t) { v.t = 1; break; } \
            break; \
        case OMPC_REDUCTION_MIN: \
            for(i = 0; i < n_thd; i++) \
                if(v.t>vals[i].r_v.t) v.t = vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_MAX: \
            for(i = 0; i < n_thd; i++) \
                if(v.t<vals[i].r_v.t) v.t = vals[i].r_v.t;\
            break; \
        default: \
            ompc_fatal("ompc_reduction: bad op\n"); \
        } \
        *((type_t *)out_p) = v.t; \
    } \
    if(tpp != NULL) ompc_thread_barrier(id,tpp); \
}

#define DO_REDUCTION_FLOAT(type_t,t) { \
    vals[id].r_v.t = *((type_t *)in_p); \
    if(tpp != NULL) ompc_thread_barrier(id,tpp); \
    if(id == 0){ \
        any_type v; int i; \
        v.t = *((type_t *)out_p); \
        switch(op){ \
        case OMPC_REDUCTION_PLUS: \
        case OMPC_REDUCTION_MINUS: \
            for(i = 0; i < n_thd; i++) v.t += vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_MUL: \
            for(i = 0; i < n_thd; i++) v.t *= vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_LOGAND: \
            if(!v.t) break; \
            for(i = 0; i < n_thd; i++) \
                if(!vals[i].r_v.t) { v.t = 0; break; } \
            break; \
        case OMPC_REDUCTION_LOGOR: \
            if(v.t) break; \
            for(i = 0; i < n_thd; i++) \
                if(vals[i].r_v.t) { v.t = 1; break; } \
            break; \
        case OMPC_REDUCTION_MIN: \
            for(i = 0; i < n_thd; i++) \
               if(v.t > vals[i].r_v.t) v.t=vals[i].r_v.t;\
            break; \
        case OMPC_REDUCTION_MAX: \
            for(i = 0; i < n_thd; i++) \
                if(v.t < vals[i].r_v.t) v.t=vals[i].r_v.t;\
            break; \
        default: \
            ompc_fatal("ompc_reduction: bad op\n"); \
        } \
        *((type_t *)out_p) = v.t; \
    } \
    if(tpp != NULL) ompc_thread_barrier(id,tpp); \
}

#define DO_REDUCTION_COMPLEX(type_t,t) { \
    vals[id].r_v.t = *((type_t *)in_p); \
    if(tpp != NULL) ompc_thread_barrier(id,tpp); \
    if(id == 0){ \
        any_type v; int i; \
        v.t = *((type_t *)out_p); \
        switch(op){ \
        case OMPC_REDUCTION_PLUS: \
        case OMPC_REDUCTION_MINUS: \
            for(i = 0; i < n_thd; i++)\
              { v.t -= vals[i].r_v.t;} \
            break; \
        case OMPC_REDUCTION_MUL: \
            for(i = 0; i < n_thd; i++)\
                { v.t *= vals[i].r_v.t; }\
            break; \
        default: \
            ompc_fatal("ompc_reduction: bad op\n"); \
        } \
        *((type_t *)out_p) = v.t; \
     } \
     if(tpp != NULL) ompc_thread_barrier(id,tpp); \
}


void
ompc_reduction(void *in_p,void *out_p,int type, int op)
{
    struct ompc_thread *tp = ompc_current_thread();
    struct ompc_thread *tpp = tp->parent;
    struct barrier_flag volatile *vals;
    int id,n_thd;

    if((n_thd = ompc_get_num_threads (tp)) == 1) {
        id = 0;
        vals = tp->barrier_flags;
    } else {
        id = tp->num;
        vals = tpp->barrier_flags;
    }

    switch(type){
    case OMPC_REDUCTION_CHAR:
        DO_REDUCTION_INTEGRAL(char,c);
        break;
    case OMPC_REDUCTION_UNSIGNED_CHAR:
        DO_REDUCTION_INTEGRAL(unsigned char,uc);
        break;

    case OMPC_REDUCTION_SHORT:
        DO_REDUCTION_INTEGRAL(short,s);
        break;
    case OMPC_REDUCTION_UNSIGNED_SHORT:
        DO_REDUCTION_INTEGRAL(unsigned short,us);
        break;

    case OMPC_REDUCTION_SIGNED:
    case OMPC_REDUCTION_INT:
        DO_REDUCTION_INTEGRAL(int, i);
        break;
    case OMPC_REDUCTION_UNSIGNED_INT:
        DO_REDUCTION_INTEGRAL(unsigned int, ui);
        break;

    case OMPC_REDUCTION_LONG:
        DO_REDUCTION_INTEGRAL(long,l);
        break;
    case OMPC_REDUCTION_UNSIGNED_LONG:
        DO_REDUCTION_INTEGRAL(unsigned long,ul);
        break;

    case OMPC_REDUCTION_LONGLONG:
        DO_REDUCTION_INTEGRAL(long long,ll);
        break;

    case OMPC_REDUCTION_UNSIGNED_LONGLONG:
        DO_REDUCTION_INTEGRAL(unsigned long long,ull);
        break;

    case OMPC_REDUCTION_FLOAT:
        DO_REDUCTION_FLOAT(float,f);
        break;

    case OMPC_REDUCTION_DOUBLE:
        DO_REDUCTION_FLOAT(double,d);
        break;

    case OMPC_REDUCTION_LONG_DOUBLE:
        DO_REDUCTION_FLOAT(long double,d);
        break;

    case OMPC_REDUCTION_FLOAT_COMPLEX:
        DO_REDUCTION_COMPLEX(float _Complex,fcx);
        break;
        
    case OMPC_REDUCTION_DOUBLE_COMPLEX:
        DO_REDUCTION_COMPLEX(double _Complex,dcx);
        break;
        
    case OMPC_REDUCTION_LONG_DOUBLE_COMPLEX:
        DO_REDUCTION_COMPLEX(long double _Complex,ldcx);
        break;

    default:
        ompc_fatal("ompc_reduction: bad type");
    }
}


void
ompc_reduction_init(void *in_p, int type, int op)
{
    ompc_fatal("ompc_reduction_init: bad op\n");
}


