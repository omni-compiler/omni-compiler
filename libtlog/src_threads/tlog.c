#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include "exc_platform.h"
#include "tlog.h"
#include "omp.h"

#if defined(OMNI_OS_LINUX) && defined(USE_PTHREAD)
#include <pthread.h>
#endif


TLOG_HANDLE tlog_handle_table[MAX_THREADS];

static TLOG_DATA *tlog_get_data(int id);
static void tlog_dump(void);
static void tlog_block_swap_bytes(TLOG_DATA *dp);
static double start_time;

char *prog_name;

void tlog_sig_handler(int x)
{
    extern int ompc_is_master_proc();
    if (ompc_is_master_proc()) {
	exit(1);	/* exit explicitly */
    } else {
	int n = 0;
	for (;;) {
	    n++;
	}
    }
}


void tlog_init(char *name)
{
    fprintf(stderr,"log on ...\n");
    prog_name = name;
    start_time = tlog_timestamp();
    tlog_log(0,TLOG_START);
    signal(SIGINT, tlog_sig_handler);
}


void tlog_slave_init()
{
#if defined(OMNI_OS_LINUX) && defined(USE_PTHREAD)
  sigset_t	sm;

  sigemptyset (&sm);
  sigaddset (&sm, SIGINT);
  pthread_sigmask (SIG_BLOCK, &sm, NULL);
#endif /* OMNI_OS_LINUX && USE_PTHREAD */
}


void tlog_finalize()
{
    TLOG_HANDLE *hp;
    int i;
    
    /* fprintf(stderr,"finalize log by %d ...\n", omp_get_thread_num()); */
    fprintf(stderr,"finalize log ...\n");
    for(i = 0; i < MAX_THREADS; i++){
	hp = &tlog_handle_table[i];
	if(hp->block_top == NULL) continue; /* not used */
	tlog_log(i,TLOG_END);
    }
    tlog_dump();
}

static TLOG_DATA *tlog_get_data(int id)
{
    TLOG_DATA *dp;
    TLOG_HANDLE *hp;
    TLOG_BLOCK *bp;

    hp = &tlog_handle_table[id];
    if((dp = hp->free_p) == NULL || dp >= hp->end_p){
	bp = (TLOG_BLOCK *)malloc(sizeof(TLOG_BLOCK));
	bzero(bp,sizeof(*bp));
	bp->next = NULL;
	if(hp->block_top == NULL){
	    hp->block_top = hp->block_tail = bp;
	} else {
	    hp->block_tail->next = bp;
	    hp->block_tail = bp;
	}
	hp->free_p = (TLOG_DATA *)bp->data;
	hp->end_p = (TLOG_DATA *)((char *)bp->data + TLOG_BLOCK_SIZE);
	dp = hp->free_p;
    }
    hp->free_p = dp+1;
    dp->proc_id = id;
    dp->time_stamp = tlog_timestamp() - start_time;
    return dp;
}

void tlog_log(int id,enum tlog_type type)
{
    TLOG_DATA *dp;
    dp = tlog_get_data(id);
    dp->log_type = type;
}

void tlog_log1(int id,TLOG_TYPE type,int arg1)
{
    TLOG_DATA *dp;
    dp = tlog_get_data(id);
    dp->log_type = type;
    dp->arg1 = (_omInt16_t)arg1;
}

void tlog_log2(int id,TLOG_TYPE type,int arg1,int arg2)
{
    TLOG_DATA *dp;
    dp = tlog_get_data(id);
    dp->log_type = type;
    dp->arg1 = (_omInt16_t)arg1;
    dp->arg2 = (_omInt32_t)arg2;
}

static void tlog_dump()
{
    FILE *fp;
    int i;
    union {
	long i;
	char c;
    } x;
    int bigendian;
    TLOG_BLOCK *bp;
#ifdef not
    TLOG_HANDLE *hp;
#endif
    char fname[256];

    x.i = 1;
    bigendian = (x.c == 0);

    if(prog_name == NULL) strcpy(fname,TLOG_FILE_NAME);
    else {
	strcpy(fname,prog_name);
	strcat(fname,".log");
    }
    
    if((fp = fopen(fname,"w")) == NULL){
	fprintf(stderr,"cannot open '%s'\n",fname);
	return;
    }

#ifdef not
    for(i = 0; i < MAX_THREADS; i++){
	hp = &tlog_handle_table[i];
	for(bp = hp->block_top; bp != NULL; bp = bp->next){
	    if(!bigendian) tlog_block_swap_bytes((TLOG_DATA *)bp->data);
	    if(fwrite((void *)bp->data,1,TLOG_BLOCK_SIZE,fp) 
	       != TLOG_BLOCK_SIZE){
		fprintf(stderr,"write error to '%s'\n",TLOG_FILE_NAME);
		return;
	    }
	}
    }
#else
    {
      TLOG_BLOCK *bps[MAX_THREADS];
      int done;
      for(i = 0; i < MAX_THREADS; i++) bps[i] = tlog_handle_table[i].block_top;
      do {
	done = 1;
	for(i = 0; i < MAX_THREADS; i++){
	  bp = bps[i];
	  if(bp != NULL){
	    done = 0;
	    bps[i] = bp->next;
	    if(!bigendian) tlog_block_swap_bytes((TLOG_DATA *)bp->data);
	    if(fwrite((void *)bp->data,1,TLOG_BLOCK_SIZE,fp) 
	       != TLOG_BLOCK_SIZE){
	      fprintf(stderr,"write error to '%s'\n",TLOG_FILE_NAME);
	      return;
	    }
	  }
	}
      } while(!done);
    }
#endif
    fclose(fp);
}

static void tlog_block_swap_bytes(TLOG_DATA *dp)
{
    union {
	char c[8];
	short int s;
	long int l;
	double d;
    } x;
    char t;
    TLOG_DATA *end_dp;

    end_dp = (TLOG_DATA *)(((char *)dp) + TLOG_BLOCK_SIZE);
    for(; dp < end_dp; dp++){
	x.s = dp->arg1;
	t = x.c[0]; x.c[0] = x.c[1]; x.c[1] = t;
	dp->arg1 = x.s;
	x.l = dp->arg2;
	t = x.c[0]; x.c[0] = x.c[3]; x.c[3] = t;
	t = x.c[1]; x.c[1] = x.c[2]; x.c[2] = t;
	dp->arg2 = x.l;
	x.d = dp->time_stamp;
	t = x.c[0]; x.c[0] = x.c[7]; x.c[7] = t;
	t = x.c[1]; x.c[1] = x.c[6]; x.c[6] = t;
	t = x.c[2]; x.c[2] = x.c[5]; x.c[5] = t;
	t = x.c[3]; x.c[3] = x.c[4]; x.c[4] = t;
	dp->time_stamp = x.d;
    }
}

/*
 * timer routine
 */
#if 0
double tlog_timestamp()
{
    double t;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = (double)(tv.tv_sec) + ((double)(tv.tv_usec))/1.0e6;
    return t ;
}
#endif
