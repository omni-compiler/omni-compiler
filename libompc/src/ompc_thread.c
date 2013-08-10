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

int ompc_n_proc = N_PROC_DEFAULT;      /* number of PE */
int ompc_bind_procs = FALSE;   /* default */

/* system lock variables */
ompc_lock_t ompc_proc_lock_obj, ompc_thread_lock_obj;

#ifdef USE_PTHREAD
# ifndef SIMPLE_SPIN
pthread_mutex_t ompc_proc_mutex;
pthread_cond_t ompc_proc_cond;
pthread_cond_t ompc_mainwait_cond;
pthread_mutex_t ompc_mainwait_mutex;
#if _POSIX_BARRIERS > 0
pthread_barrier_t ompc_thd_bar;
#endif
# endif /* !SIMPLE_SPIN */
#endif /* USE_PTHREAD */

#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
ompc_proc_t ompc_sproc_pid[MAX_PROC];
#endif /* USE_SPROC && OMNI_OS_IRIX */

/* hash table */
struct ompc_proc *ompc_proc_htable[PROC_HASH_SIZE];
/* proc table */
struct ompc_proc *ompc_procs;

static ompc_proc_t ompc_master_proc_id;

/* prototype */
#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
static void *ompc_slave_proc(void *, size_t);
#else
static void *ompc_slave_proc(void *);
#endif /* USE_SPROC && OMNI_OS_IRIX */
static struct ompc_proc *ompc_new_proc(void);
static struct ompc_proc *ompc_current_proc(void);
static struct ompc_proc *ompc_get_proc(int hint);
static void ompc_free_proc(struct ompc_proc *p);
static struct ompc_thread *ompc_alloc_thread(struct ompc_proc *proc);
static void ompc_free_thread(struct ompc_proc *proc, struct ompc_thread *p);
/*static*/ struct ompc_thread *ompc_current_thread(void);
static void ompc_thread_barrier2(int id, struct ompc_thread *tpp);

extern void ompc_call_fsub(struct ompc_thread *tp);

#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
/* SGI sproc() special care */
static size_t getProcessStackSize(void);

size_t
getProcessStackSize()
{
    struct rlimit limit;
    size_t size;
    if (getrlimit(RLIMIT_STACK, &limit) < 0) {
        perror("getrlimit");
        fprintf(stderr, "can't get stack limit max.\n");
        exit(1);
    }
    /*
     * FIXME:
     * Factor 1/2 is not the best value.
     * Are there any good way to compute stack size of each shared
     * process?
     */
    size = limit.rlim_max / 2;
    return size;
}
#endif /* USE_SPROC && OMNI_OS_IRIX */

/* 
 * initialize library
 */
#ifdef not
void
ompc_init_proc_num(int pnum)
{
    ompc_n_proc = pnum;
    ompc_init();
}
#endif

void
ompc_init(int argc,char *argv[])
{
    char  * cp;
    int t, r, val;
    long lnp;
    struct ompc_thread *tp;
    struct ompc_proc *cproc;
    size_t maxstack = 0;
#if defined(OMNI_OS_IRIX) && defined(USE_SPROC)
    size_t thdStackSize;
#endif /* OMNI_OS_IRIX && USE_SPROC */

#ifdef USE_PTHREAD
    static pthread_t thds[MAX_PROC];
    static pthread_attr_t attr;

    pthread_attr_init(&attr);

# ifndef SIMPLE_SPIN
    pthread_mutex_init(&ompc_proc_mutex,NULL);
    pthread_cond_init(&ompc_proc_cond,NULL);
    pthread_mutex_init(&ompc_mainwait_mutex,NULL);
    pthread_cond_init(&ompc_mainwait_cond,NULL);
# endif /* SIMPLE_SPIN */

# if !defined(OMNI_OS_FREEBSD) && !defined(OMNI_OS_IRIX) && !defined(OMNI_OS_CYGWIN32)
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
# endif /* !OMNI_OS_FREEBSD ... */
# if 0
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
# endif
#endif /* USE_PTHREAD */

#ifdef OMNI_OS_SOLARIS
    lnp = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(OMNI_OS_IRIX)
    lnp = sysconf(_SC_NPROC_ONLN);
#elif defined(OMNI_OS_LINUX)
    {
      char buff[BUFSIZ];
      FILE *fp;
      int  npes;
      char procfile[] = "/proc/stat";

      fp = fopen (procfile, "r");
      if ( fp == NULL ){
        fprintf (stderr, "cannot open \"%s\".\n"
                 "cannot get maximum number of processors.\n", procfile);
        lnp = 1;
      }
      else {
        npes = 0;
        while( fgets(buff, BUFSIZ, fp) != NULL ){
          if ( !strncmp(buff, "cpu", 3) && isdigit(buff[3]) ){
            npes += 1;
          }
        }
        fclose (fp);
        lnp = (npes == 0)? 1: npes;
      }
    }
#else
    lnp = ompc_n_proc;
#endif /* OMNI_OS_SOLARIS */

    if (ompc_n_proc != lnp)
        ompc_n_proc = lnp;
    
#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
    __ateachexit(ompc_finalize);
#else    
    atexit(ompc_finalize);
#endif /* USE_SPROC && OMNI_OS_IRIX */

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

    cp = getenv("OMPC_BIND_PROCS");
    if(cp != NULL && (strcmp(cp, "TRUE") == 0 || strcmp(cp, "true") == 0))
        ompc_bind_procs = TRUE;

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
        ompc_n_proc = val;
    }

    ompc_max_threads = ompc_n_proc;   /* max number of thread, default */

    cp = getenv("OMP_NUM_THREADS");     /* a number of team member */
    if ( cp == NULL )
        ompc_num_threads = ompc_n_proc;
    else {
        sscanf(cp, "%d", &val);
        if(val <= 0) ompc_fatal("bad OMP_NUM_THREADS(<= 0)");
        ompc_num_threads = val;

        if(ompc_num_threads > ompc_max_threads)
            ompc_max_threads = ompc_num_threads;
    }

    /* ompc_num_threads cannot be different from
     * ompc_max_threads in Omni. 
     */
    ompc_max_threads = ompc_num_threads;

#ifdef USE_PTHREAD
#if _POSIX_BARRIERS > 0
    pthread_barrier_init(&ompc_thd_bar, 0, ompc_num_threads);
#endif
#endif

#if (defined(OMNI_OS_IRIX) || defined(OMNI_OS_DARWIN)) && defined(USE_PTHREAD)
    pthread_setconcurrency(ompc_max_threads);
#endif /* OMNI_OS_IRIX && USE_PTHREAD */

    cp = getenv("OMPC_STACK_SIZE");   /* stack size of threads */
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
        maxstack = val*unit;
        if ( maxstack < DEF_STACK_SIZE ){
            maxstack = 0;       /* default */
            printf("Stack size is not change, because it is less than the default(=1MB).\n");
        }
    }
#ifdef USE_PTHREAD
    pthread_attr_setstacksize(&attr, maxstack);
#endif /* USE_PTHREAD */

    ompc_task_end = 0;

    /* hash table initialize */
    bzero(ompc_proc_htable, sizeof(ompc_proc_htable));

    /* allocate proc structure */
    ompc_procs =
        (struct ompc_proc *)malloc(sizeof(struct ompc_proc)*ompc_max_threads);
    if(ompc_procs == NULL) ompc_fatal("Cannot allocate proc table.");
    bzero(ompc_procs,sizeof(struct ompc_proc)*ompc_max_threads);

    /* init system lock */
    ompc_init_lock(&ompc_proc_lock_obj);
    ompc_init_lock(&ompc_thread_lock_obj);
    ompc_critical_init ();     /* initialize critical lock */
    ompc_atomic_init_lock ();  /* initialize atomic lock */

        /* add (and init proc table) this as master thread */
    cproc = ompc_new_proc();
    cproc->is_used = TRUE;  /* this is always used */
    ompc_master_proc_id = _OMPC_PROC_SELF;

    if(ompc_debug_flag)
        fprintf(stderr, "Creating %d slave thread ...\n", ompc_max_threads-1);

#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
    (void)usconfig(CONF_INITUSERS, (unsigned int)ompc_max_threads);
    ompc_sproc_pid[0] = getpid();
    thdStackSize = getProcessStackSize();
#endif /* USE_SPROC && OMNI_OS_IRIX */

    for( t = 1; t < ompc_max_threads; t++ ){
        if(ompc_debug_flag) fprintf(stderr, "Creating slave %d  ...\n", t);

#ifdef USE_SOL_THREAD
        r = thr_create(NULL, maxstack, ompc_slave_proc, (void *)t,
                       THR_BOUND, NULL);
#elif defined(USE_PTHREAD)
        r = pthread_create(&thds[t],
                           &attr, (cfunc)ompc_slave_proc, (void *)((_omAddrInt_t)t));
#elif defined(USE_SPROC) && defined(OMNI_OS_IRIX)
        if (getpid() == (pid_t)ompc_sproc_pid[0]) {
            ompc_sproc_pid[t] = (ompc_proc_t)sprocsp((cfunc)ompc_slave_proc,
                                                       PR_SALL,
                                                       (void *)NULL,
                                                       (caddr_t)NULL,
                                                       thdStackSize);
            if (ompc_sproc_pid[t] > 0) {
                r = 0;
            } else {
                r = -1;
            }
        }
#else
        ompc_fatal("no thread library!!");
#endif /* USE_SOL_THREAD */

        if ( r ){
            extern int errno;
            fprintf(stderr, "thread create fails at id %d:%d errno=%d\n", t, r, errno);
            perror("thread creation");
            exit(1);
        }
    }

    OMPC_WAIT((volatile int)ompc_proc_counter != (volatile int)ompc_max_threads);

    /* setup master root thread */
    tp = ompc_alloc_thread(cproc);
    tp->num             = 0;    /* team master */
    tp->in_parallel     = 0;
    tp->parent          = NULL;
    cproc->thr          = tp;

    if(ompc_debug_flag) fprintf(stderr, "init end(Master)\n");
}


/* finalize */
void
ompc_finalize()
{
#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
    if(ompc_sproc_pid[0] == getpid()){
        int t;
        for( t = 1; t < ompc_max_threads; t++ )
            kill(ompc_sproc_pid[t], SIGTERM);
    
        for( t = 1; t < ompc_max_threads; t++ )
            wait(NULL);
    }
#endif /* USE_SPROC && OMNI_OS_IRIX */
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

#ifdef USE_SOL_THREAD
/* find on-line processor id */
# define P_MAX 256
static int cpu_id = -1;
int
getpeid()
{
    int  i, err;
    processor_info_t  pt;

    for( i = cpu_id+1 ; i < P_MAX ; i++ ){
        err = processor_info((processorid_t)i, &pt);
        if ( err < 0 )
            continue;
        cpu_id = i;
        return cpu_id;
    }
    exit(1);/*return -1;*/
}
#endif /* USE_SOL_THREAD */

int
ompc_is_master_proc()
{
    return ompc_master_proc_id == _OMPC_PROC_SELF;
}

/* setup new ompc_proc: master is always at first proc table */
static struct ompc_proc *
ompc_new_proc()
{
    struct ompc_proc  * p, ** pp;
    ompc_proc_t  id = _OMPC_PROC_SELF;

    OMPC_PROC_LOCK();
    p = &ompc_procs[ompc_proc_counter];
    ompc_proc_counter++;

    p->pid = id;
    p->thr = NULL;
    /* add this proc table to hash */
    pp = &ompc_proc_htable[PROC_HASH_IDX(id)];
    p->link = *pp;
    *pp = p;

#ifdef USE_SOL_THREAD
    /* if ompc_max_threads is less than the number of processor, bind it. */
    if(ompc_bind_procs && (ompc_max_threads <= ompc_n_proc)){
        int  peid, err;
        peid = getpeid();
        if(ompc_debug_flag) 
            fprintf(stderr, "thread #[%d] is bind to pe[%d]\n",
                   ompc_proc_counter-1, peid);
        err = processor_bind(P_LWPID, P_MYID, peid, NULL);
        if ( err < 0 ){
            perror("processor_bind");
            exit(1);
        }
    }
#endif /* USE_SOL_THREAD */
    OMPC_PROC_UNLOCK();
    return p;
}

/*static*/ struct ompc_thread *
ompc_current_thread()
{
    ompc_proc_t id;
    struct ompc_proc *p;
    struct ompc_thread *tp;

    id = _OMPC_PROC_SELF;
#if 0
    fprintf(stderr, "current thread: %d, idx %lud\n", (int)id, PROC_HASH_IDX(id));
#endif
    for( p = ompc_proc_htable[PROC_HASH_IDX(id)]; p != NULL; p = p->link ){
        if(p->pid == id){
            if((tp = p->thr) == NULL)
                ompc_fatal("unkonwn thread is running");
            return tp;
        }
    }
    ompc_fatal("unknown proc is running");
    return NULL;
}

static struct ompc_proc *
ompc_current_proc()
{
    ompc_proc_t id;
    struct ompc_proc *p;

    id = _OMPC_PROC_SELF;
    for(p = ompc_proc_htable[PROC_HASH_IDX(id)]; p != NULL; p = p->link){
        if ( p->pid == id )
            return p;
    }
    fprintf(stderr, "pid=%d\n", (unsigned int)id);
    ompc_fatal("unknown proc is running");
    return NULL;
}

/* get thread from free list */
static struct ompc_proc *
ompc_get_proc(int hint)
{
    struct ompc_proc *p;
    int i;
    static int last_used = 0;

    OMPC_PROC_LOCK();
    p = &ompc_procs[hint];
    if(!p->is_used)
        p->is_used = TRUE;
    else {
        for(i = 0; i < ompc_max_threads; i++){
            if(++last_used >= ompc_max_threads) last_used = 0;
            p = &ompc_procs[last_used];
            if(!p->is_used) break;
        }
        if(p->is_used) p = NULL;
        else p->is_used = TRUE;
    }
    OMPC_PROC_UNLOCK();

    return p;
}

static void
ompc_free_proc(struct ompc_proc *p)
{
    OMPC_PROC_LOCK();
    p->is_used = FALSE;
    OMPC_PROC_UNLOCK();
}

/* allocate/get thread entry */
static struct ompc_thread *
ompc_alloc_thread(struct ompc_proc *proc)
{
    struct ompc_thread *p;

    if ((p = proc->free_thr) != NULL)
        proc->free_thr = p->freelist;
    else {
        p = (struct ompc_thread *)malloc(sizeof(struct ompc_thread));
        if (p == NULL)
            ompc_fatal("ompc_alloc_thread: malloc failed");
    }

    return p;
}

static void
ompc_free_thread(struct ompc_proc *proc,struct ompc_thread *p)
{
    p->freelist = proc->free_thr;
    proc->free_thr = p;
}


#if defined(USE_SPROC) && defined(OMNI_OS_IRIX)
static void *ompc_slave_proc(void *arg, size_t stackSize)
#else
static void *ompc_slave_proc(void *arg)
#endif /* USE_SPROC && OMNI_OS_IRIX */
{
    struct ompc_proc *cproc;    /* current process */
    struct ompc_thread *tp;
    struct ompc_thread *me;
    int i;

#ifdef USE_LOG
    if(ompc_log_flag) {
      tlog_slave_init ();
    }
#endif /* USE_LOG */

    cproc = ompc_new_proc();

    for(;;) {

#if defined(USE_PTHREAD) && !defined(SIMPLE_SPIN)
        if ((struct ompc_proc * volatile)cproc->thr == NULL){
            volatile int c;
            for( c = 0 ; (struct ompc_proc * volatile)cproc->thr == NULL ; c++ ){
                if ( c > MAX_COUNT ){
                    pthread_mutex_lock(&ompc_proc_mutex);
                    while((struct ompc_proc * volatile)cproc->thr == NULL){
                        pthread_cond_wait(&ompc_proc_cond,&ompc_proc_mutex);
                    }
                    pthread_mutex_unlock(&ompc_proc_mutex);
                    c = 0;
                }
            }
        }
#else /* if defined(USE_SOL_THREAD) || defined(USE_SPROC) || defined(SIMPLE_SPIN) */
        /* wait for starting job */
        OMPC_WAIT((struct ompc_proc * volatile)cproc->thr == NULL);
#endif /* USE_PTHREAD && !SIMPLE_SPIN */

        if ( ompc_task_end > 0 ){      /* terminate */
            break;
        }
        tp = cproc->thr->parent;

        i = cproc->thr->num;
#ifdef USE_LOG
        if(ompc_log_flag) tlog_parallel_IN(i);
#endif /* USE_LOG */
        if ( tp->nargs < 0) {
            /* call C function */
            if ( tp->args != NULL )
                (*tp->func)(tp->args, cproc->thr);
            else
                (*tp->func)(cproc->thr);
        } else {
            /* call Fortran function */
            ompc_call_fsub(tp);
        }

#ifdef USE_LOG
        if(ompc_log_flag) tlog_parallel_OUT(i);
#endif /* USE_LOG */
        /* on return, clean up */
        me = cproc->thr;
        cproc->thr = NULL;
        ompc_free_thread(cproc,me);    /* free thread & put me to freelist */
        ompc_free_proc(cproc);
        ompc_thread_barrier2(i,tp);
    }

#if 0
    fprintf(stderr, "Exit slave[%d]\n", _OMPC_PROC_SELF);
#endif
    return NULL;
}

/* called from compiled code. */
void
ompc_do_parallel_main (int nargs, int cond, int nthds,
    cfunc f, void *args)
{
    struct ompc_proc *cproc, *proclist, *p;
    struct ompc_thread *cthd, *tp;
    int i, n_thds, max_thds, in_parallel;

    cproc = ompc_current_proc();
    cthd  = cproc->thr;
#if 0
    fprintf(stderr, "  parallel proc[%d] omp num[%d].\n", cproc->pid, ompc_num_threads);
#endif

    if (cond == 0) { /* serialized by parallel if(false) */
        max_thds = 1;
        in_parallel = cthd->in_parallel;
    } else if ((cthd->parent != NULL) && ompc_nested == 0) { /* serialize nested parallel region */
        max_thds = 1;
        in_parallel = 1;
    } else {
        max_thds = (nthds < ompc_num_threads) ? (nthds) : (ompc_num_threads);
        in_parallel = 1;
    }

    proclist = NULL;
    for( n_thds = 1; n_thds < max_thds; n_thds ++ ){
        if ((p = ompc_get_proc(n_thds)) == NULL){
#if 0
          fprintf(stderr, "   -cannot find thread %d\n", n_thds);
#endif
          break;
        }
        p->next = proclist;
        proclist = p;
    }

    /* initialize parent thread */
    cthd->num_thds = n_thds;
    cthd->nargs = nargs;
    cthd->args = args;
    cthd->func = f;
#if 0
    fprintf(stderr, "  thread team total[%d] from[%d]\n", cthd->num_thds, ompc_num_threads);
#endif

    /* initialize barrier structure */
    cthd->out_count = 0;
    cthd->in_count = 0;
    for( i = 0; i < n_thds; i++ ){
        cthd->barrier_flags[i]._v = cthd->barrier_sense;
        cthd->in_flags[i]._v = 0;
    }

    /* assign thread to proc */
    for( i = 1; i < n_thds; i++ ){
        p = proclist;
        proclist = proclist->next;
        tp = ompc_alloc_thread(p);
        tp->parent = cthd;
        tp->num = i;                        /* set thread_num */
        tp->in_parallel = in_parallel;
        p->thr = tp;                        /* start it ! */
        MBAR();
    }

#if defined(USE_PTHREAD) && !defined(SIMPLE_SPIN)
    if ( n_thds > 1 ){
        pthread_mutex_lock(&ompc_proc_mutex);
        pthread_cond_broadcast(&ompc_proc_cond);
        pthread_mutex_unlock(&ompc_proc_mutex);
    }
#endif /* USE_PTHREAD && !SIMPLE_SPIN */

    /* allocate master in this team */
    tp = ompc_alloc_thread(cproc);
    tp->parent = cthd;
    tp->num = 0;  /* this is master */
    tp->in_parallel = in_parallel;
    tp->nargs = nargs;
    tp->args = args;
    cproc->thr = tp;

#ifdef USE_LOG
    if(ompc_log_flag) tlog_parallel_IN(0);
#endif /* USE_LOG */
    /* execute on master */
    if ( nargs < 0) {
        /* call C function */
        if ( args == NULL )
            (*f)(tp);
        else
            (*f)(args, tp);
    } else {
        /* call Fortran function */
        ompc_call_fsub(cthd);
    }
#ifdef USE_LOG
    if(ompc_log_flag) tlog_parallel_OUT(0);
#endif /* USE_LOG */

    /* clean up this thread */
    ompc_free_thread(cproc,tp);
    ompc_thread_barrier2(0, cthd);
    cproc->thr = cthd;
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
#ifndef USE_PTHREAD_BARRIER
    int sen0,n;
#endif // USE_PTHREAD_BARRIER

    if(tpp == NULL) return; /* not in parallel */
#ifdef USE_LOG
    if(ompc_log_flag) tlog_barrier_IN(id);
#endif // USE_LOG

#ifdef USE_PTHREAD_BARRIER
    pthread_barrier_wait(&ompc_thd_bar);
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
#endif // USE_PTHREAD_BARRIER

#ifdef USE_LOG
    if(ompc_log_flag) tlog_barrier_OUT(id);
#endif // USE_LOG
}


void
ompc_thread_barrier2(int id, struct ompc_thread *tpp)
{
    int sen0,n;

    if(tpp == NULL) return; /* not in parallel */
#ifdef USE_LOG
    if(ompc_log_flag) tlog_barrier_IN(id);
#endif // USE_LOG
    sen0 = tpp->barrier_sense ^ 1;
    n = tpp->num_thds;
#ifdef USE_PTHREAD_BARRIER
    if (id == 0) {
        int j;
        const struct timespec t = { 0, 10 };
        for ( j = 1 ; j < n ; j++ ) {
            if ((volatile int)tpp->barrier_flags[j]._v != sen0) {
                pthread_mutex_lock(&ompc_mainwait_mutex);
                while((volatile int)tpp->barrier_flags[j]._v != sen0) {
                    pthread_cond_timedwait(&ompc_mainwait_cond, &ompc_mainwait_mutex, &t);
                }
                pthread_mutex_unlock(&ompc_mainwait_mutex);
            }
        }
        tpp->barrier_sense = sen0;
        MBAR();
    } else {
        //pthread_mutex_lock(&ompc_mainwait_mutex);
        tpp->barrier_flags[id]._v = sen0;
        pthread_cond_signal(&ompc_mainwait_cond);
        //pthread_mutex_unlock(&ompc_mainwait_mutex);
    }
#else
    if (id == 0) {
        int j;
        for ( j = 1 ; j < n ; j++ ) {
            OMPC_WAIT((volatile int)tpp->barrier_flags[j]._v != sen0);
        }
        tpp->barrier_sense = sen0;
        MBAR();
    } else {
        tpp->barrier_flags[id]._v = sen0;
    }
#endif // USE_PTHREAD_BARRIER

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
ompc_terminate (int exitcode)
{
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


void
ompc_set_num_threads(int n)
{
    extern void omp_set_num_threads();
    omp_set_num_threads(n);
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

