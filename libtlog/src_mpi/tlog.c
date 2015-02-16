#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "mpi.h"
#include "tlog_mpi.h"

int tlog_debug = FALSE;

TLOG_HANDLE tlog_handle_table[MAX_THREADS];
static TLOG_BLOCK *free_block = NULL;

static TLOG_DATA *_tlog_get_data(int id);
static void _tlog_adjust_all_time();
void _tlog_log(int tid,enum tlog_type type,int arg);
void _tlog_file_open();
void _tlog_file_close();
void _tlog_dump_data(char *datap);
void _tlog_adjust_time(double t_diff);
void _tlog_collect_and_dump();

static double start_time;

static char *_tlog_fname;

static int _tlog_on = FALSE;

static int _tlog_mpi_id;
static int _tlog_mpi_nproc;

#define TLOG_TAG 9999
#define TLOG_BARRIER()   MPI_Barrier(MPI_COMM_WORLD)
#define TLOG_SEND_DATA(dst,addr,nbytes) \
 MPI_Send((char *)(addr),nbytes,MPI_BYTE,dst,TLOG_TAG,MPI_COMM_WORLD)
#define TLOG_RECV_DATA(src,addr,nbytes) \
 MPI_Recv((char *)(addr),nbytes,MPI_BYTE,src,TLOG_TAG,MPI_COMM_WORLD,&status)

void tlog_initialize()
{
    MPI_Comm_size(MPI_COMM_WORLD,&_tlog_mpi_nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&_tlog_mpi_id);

    _tlog_fname = TLOG_DEFAULT_FILENAME;

    /* init and set timestamp */
    tlog_timestamp_init();
    start_time = tlog_timestamp();

    _tlog_adjust_all_time();
    _tlog_adjust_all_time();

#ifdef not
    { 
	int i;
	for(i = 1; i < 10; i++){
	    if(_tlog_mpi_id == 0) printf("%d:\n",i);
	    _tlog_adjust_all_time();
	    fflush(stdout);
	    TLOG_BARRIER();
	}
    }
#endif
    if(tlog_debug){
	if(_tlog_mpi_id == 0) printf("tlog on ...\n");
    }

    TLOG_BARRIER();
    _tlog_on = TRUE;
    _tlog_log(0,TLOG_START,0);
    /* scash_prf("tlog_init_end t=%g",tlog_get_time()); */
}

void tlog_log_filename(char *name)
{
    _tlog_fname = strdup(name);
}

void _tlog_adjust_time(double t_diff)
{
    start_time -= t_diff;
}

double _tlog_get_time()
{
    double t;
    t = tlog_timestamp();
    return t - start_time;
}

static void _tlog_adjust_all_time()
{
    MPI_Status status;
    double t1,t2,t0,t_diff,t12,min_t12;
    int i,k;
    char c;

    TLOG_BARRIER();
    if(_tlog_mpi_id == 0){
	for(i = 1; i < _tlog_mpi_nproc; i++){
	  min_t12 = 100000.0;
	  for(k = 0; k < 10; k++){
	    t1 = _tlog_get_time();
	    TLOG_SEND_DATA(i,&c,1);
	    TLOG_RECV_DATA(i,(char *)&t0,sizeof(double));
	    t2 = _tlog_get_time();
	    t12 = t2-t1;
	    if(min_t12 > t12){
	      min_t12 = t12;
	      t_diff = t1+(t2-t1)/2.0 - t0;
	    }
	  }
	  TLOG_SEND_DATA(i,(char *)&t_diff,sizeof(double));
	  if(tlog_debug) 
	    printf("adjust i=%d,t1=%g,t2=%g,t0=%g,diff=%g\n", i,t1,t2,t0,t_diff);
	}
    } else {
      for(k = 0; k < 10; k++){
	TLOG_RECV_DATA(0,&c,1);
	t0 = _tlog_get_time();
	TLOG_SEND_DATA(0,(char *)&t0,sizeof(double));
      }
	TLOG_RECV_DATA(0,(char *)&t_diff,sizeof(double));
	/* printf("t_diff=%g\n",t_diff); */
	_tlog_adjust_time(t_diff);
    }
    TLOG_BARRIER();
}

static TLOG_DATA *_tlog_get_data(int tid)
{
    TLOG_DATA *dp;
    TLOG_HANDLE *hp;
    TLOG_BLOCK *bp;

    hp = &tlog_handle_table[tid];
    if((dp = hp->free_p) == NULL || dp >= hp->end_p){
	if((bp = free_block) == NULL){
	    bp = (TLOG_BLOCK *)malloc(sizeof(TLOG_BLOCK));
	    if(bp == NULL) {
		fprintf(stderr,"cannot allocate tlog buffer\n");
		exit(1);
	    }
	} else {
	    free_block = bp->next;
	}

	/* clear buffer */
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
    dp->proc_id = _tlog_mpi_id;
    dp->arg1 = tid;
    dp->time_stamp = tlog_timestamp() - start_time;
    return dp;
}

void _tlog_reset_buffer()
{
    //TLOG_DATA *dp;
    TLOG_HANDLE *hp;
    //TLOG_BLOCK *bp;

    hp = &tlog_handle_table[0]; /* only for 0, no thread version */
    free_block = hp->block_top;
    bzero(hp,sizeof(*hp));
}

void _tlog_log(int tid,enum tlog_type type,int arg)
{
    TLOG_DATA *dp;

    if(!_tlog_on) return;

    dp = _tlog_get_data(tid);
    dp->log_type = type;
    dp->arg2 = arg;
}

static FILE *tlog_fp;
int tlog_is_bigendian;

void _tlog_file_open()
{
  union {
    long i;
    char c;
  } x;

  x.i = 1;
  tlog_is_bigendian = (x.c == 0);
  
  if((tlog_fp = fopen(_tlog_fname,"w")) == NULL){
    fprintf(stderr,"cannot open '%s'\n",_tlog_fname);
    return;
  }
}

void _tlog_file_close()
{
    fclose(tlog_fp);
}

void _tlog_dump_data(char *datap)
{
    if(!tlog_is_bigendian) _tlog_block_swap_bytes((TLOG_DATA *)datap);
    if(fwrite((void *)datap,1,TLOG_BLOCK_SIZE,tlog_fp) 
       != TLOG_BLOCK_SIZE){
	fprintf(stderr,"write error to '%s'\n",_tlog_fname);
	return;
    }
}

/* 
 * no thread version 
 */
void tlog_finalize()
{

    _tlog_log(0,TLOG_END,0);

    if(tlog_debug){
	if(_tlog_mpi_id == 0) printf("tlog finalizing ...\n");
    }

    _tlog_on = FALSE;
    _tlog_collect_and_dump();

    if(tlog_debug){
	if(_tlog_mpi_id == 0) printf("tlog dump done ...\n");
    }
}

void tlog_log_flush()
{
    _tlog_on = FALSE;

    _tlog_collect_and_dump();
    _tlog_reset_buffer();
    
    _tlog_on = TRUE;
}

void _tlog_collect_and_dump()
{
    TLOG_BLOCK *bp;
    char buf[TLOG_BLOCK_SIZE];
    int size,i;
    char c;	/* dummy buffer */
    MPI_Status status;

    TLOG_BARRIER(); 
    if(_tlog_mpi_id == 0){	/* master */
	_tlog_file_open();
	/* first dump my data */
	for(bp = tlog_handle_table[0].block_top; 
	    bp != NULL;  bp = bp->next)
	    _tlog_dump_data((char *)bp->data);
	for(i = 1; i < _tlog_mpi_nproc; i++){
	    while(1){
		TLOG_SEND_DATA(i,&c,1);
		TLOG_RECV_DATA(i,(char *)&size,sizeof(int));
		/* scash_prf("dump from pe=%d, size=%d",i,size); */
		if(size == 0) break;
		TLOG_RECV_DATA(i,(char *)buf,TLOG_BLOCK_SIZE);
		_tlog_dump_data(buf);
	    }
	}
	_tlog_file_close();
    } else {
	/* send log */
	size = TLOG_BLOCK_SIZE;
	for(bp = tlog_handle_table[0].block_top; 
	    bp != NULL;  bp = bp->next){
	    TLOG_RECV_DATA(0,&c,1);
	    TLOG_SEND_DATA(0,(char *)&size,sizeof(int));
	    TLOG_SEND_DATA(0,(char *)bp->data,size);
	}
	size = 0;
	TLOG_RECV_DATA(0,&c,1);
	TLOG_SEND_DATA(0,(char *)&size,sizeof(int));
    }
    TLOG_BARRIER();
}


void tlog_log(enum tlog_type type)
{
    _tlog_log(0,type,0);
}

void tlog_log2(enum tlog_type type, int arg)
{
    _tlog_log(0,type,arg);
}

/* 
 * for fortran 
 */
void tlog_log__(enum tlog_type *type)
{
    _tlog_log(0,*type,0);
}

void tlog_initialize__() { tlog_initialize(); }

void tlog_finalize__() { tlog_finalize(); }

