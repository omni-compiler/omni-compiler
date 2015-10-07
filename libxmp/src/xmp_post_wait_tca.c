#include <stdlib.h>
#include <pthread.h>
#include "xmp_internal.h"
#include <time.h>
#include <sys/time.h>

typedef struct _XMP_postreq_info{
  int node;
  int tag;
} _XMP_postreq_info_t;

typedef struct _XMP_postreq{
  _XMP_postreq_info_t *table;   /**< Table for post requests */
  int                 num;      /**< How many post requests are in table */
  int                 max_size; /**< Max size of table */
  pthread_mutex_t     lock;
  pthread_cond_t      added;
} _XMP_postreq_t;

static _XMP_postreq_t _postreq;

//static int __reqnum = 0;
//static pthread_mutex_t __lock = PTHREAD_MUTEX_INITIALIZER;
//static pthread_cond_t __notempty = PTHREAD_COND_INITIALIZER;
//static pthread_cond_t __added = PTHREAD_COND_INITIALIZER;

/**
 * Initialize environment for post/wait directives
 */
void _xmp_tca_post_wait_initialize()
{
  _postreq.num      = 0;
  _postreq.max_size = _XMP_POSTREQ_TABLE_INITIAL_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);
  pthread_mutex_init(&_postreq.lock, NULL);
  pthread_cond_init(&_postreq.added, NULL);

  _postreq.table[0].node = 0;
  _postreq.table[_postreq.max_size-1].node=0;
}

static void do_post(const int node, const int tag)
{
  //fprintf(stderr, "do_post(%d, %d)\n", node, tag);
  //lock
  pthread_mutex_lock(&_postreq.lock);

  if(_postreq.num == _postreq.max_size){
    fprintf(stderr, "reallocation\n");
    _postreq.max_size *= _XMP_POSTREQ_TABLE_INCREMENT_RATIO;
    size_t next_size = sizeof(_XMP_postreq_info_t) * _postreq.max_size;
    _XMP_postreq_info_t *tmp;
    if((tmp = realloc(_postreq.table, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _postreq.table = tmp;
  }

  //add request
  _postreq.table[_postreq.num].node = node;
  _postreq.table[_postreq.num].tag  = tag;
  _postreq.num++;

  //unlock
  pthread_mutex_unlock(&_postreq.lock);

  pthread_cond_signal(&_postreq.added);
}

static void shift_postreq(const int index)
{
  //lock
  //pthread_mutex_lock(&_postreq.lock);
  
  if(index != _postreq.num-1){  // Not tail index
    for(int i=index+1;i<_postreq.num;i++){
      _postreq.table[i-1] = _postreq.table[i];
    }
  }
  _postreq.num--;

  //unlock
  //pthread_mutex_unlock(&_postreq.lock);
}

inline static bool remove_request_noargs()
{
  if(_postreq.num > 0){
    shift_postreq(_postreq.num-1);
    return _XMP_N_INT_TRUE;
  }
  return _XMP_N_INT_FALSE;
}

inline static bool remove_request_node(const int node)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

inline static bool remove_request(const int node, const int tag)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(node == _postreq.table[i].node && tag == _postreq.table[i].tag){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;

  /* //なかったとき */
  /* while(1){ */
  /*   // pthread_cond_wait(&__added, &_postreq.lock); */
  /*   for(int i=_postreq.num-1;i>=0;i--){ */
  /*     if(node == _postreq.table[i].node && tag == _postreq.table[i].tag){ */
  /* 	shift_postreq(i); */
  /* 	//pthread_mutex_unlock(&_postreq.lock); */
  /* 	return _XMP_N_INT_TRUE; */
  /*     } */
  /*   } */
  /* } */
  
  /* while(_postreq.num == 0){ */
  /*   pthread_cond_wait(&__notempty, &_postreq.lock); */
  /* } */
  
  /* for(int i=_postreq.num-1;i>=0;i--){ */
  /*   if(node == _postreq.table[i].node && tag == _postreq.table[i].tag){ */
  /*     shift_postreq(i); */
  /*     pthread_mutex_unlock(&_postreq.lock); */
  /*     return _XMP_N_INT_TRUE; */
  /*   } */
  /* } */
  
  /* pthread_mutex_unlock(&_postreq.lock); */
  /* return _XMP_N_INT_FALSE; */
}

//for comm_thread
void _xmp_tca_postreq(const int node, const int tag)
{
  do_post(node, tag);
}

/**
 * Post operation
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_tca_post(const int node, const int tag)
{
  if(node == _XMP_world_rank){
    do_post(_XMP_world_rank, tag);
  } else{
    _XMP_tca_comm_send(node, _XMP_TCA_POSTREQ_TAG, tag);
  }
}

/**
 * Wait operation without node-ref and tag
 */
void _xmp_tca_wait_noargs()
{
  /* while(remove_request_noargs() == false){ */
  /*   continue; */
  /* } */

  pthread_mutex_lock(&_postreq.lock);

  if(_postreq.num == 0){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
  }

  if(remove_request_noargs()){
    pthread_mutex_unlock(&_postreq.lock);
    return;
  }

  while(1){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
    
    if(remove_request_noargs()){
      pthread_mutex_unlock(&_postreq.lock);
      return;
    }
  }
}

/**
 * Wait operation with node-ref
 *
 * @param[in] node node number
 */
void _xmp_tca_wait_node(const int node)
{
  /* while(remove_request_node(node) == false){ */
  /*   continue; */
  /* } */

  pthread_mutex_lock(&_postreq.lock);

  if(_postreq.num == 0){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
  }

  if(remove_request_node(node)){
    pthread_mutex_unlock(&_postreq.lock);
    return;
  }

  while(1){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
    
    if(remove_request_node(node)){
      pthread_mutex_unlock(&_postreq.lock);
      return;
    }
  }
}

//extern struct timespec begin_ts, end_ts;
//struct timespec begin_ts_2, end_ts_2;
double getElapsedTime_(struct timespec *begin, struct timespec *end);

/**
 * Wait operation with node-ref and tag
 *
 * @param[in] node node number
 * @param[in] tag  tag
 */
void _xmp_tca_wait(const int node, const int tag)
{
  pthread_mutex_lock(&_postreq.lock);

  if(_postreq.num == 0){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
  }

  if(remove_request(node, tag)){
    pthread_mutex_unlock(&_postreq.lock);
    return;
  }

  while(1){
    pthread_cond_wait(&_postreq.added, &_postreq.lock);
    
    if(remove_request(node, tag)){
      pthread_mutex_unlock(&_postreq.lock);
      return;
    }
  }

  /*
  pthread_mutex_lock(&__lock);
  while(__reqnum == 0){
    pthread_cond_wait(&__notempty, &__lock);
  }
  --__reqnum;
  pthread_mutex_unlock(&__lock);
  return;
  */
  
  /*
  volatile int *reqnum_p = &__reqnum;
  while(1){
    pthread_mutex_lock(&__lock);
    if(*reqnum_p > 0){
      --*reqnum_p;
      pthread_mutex_unlock(&__lock);
      break;
    }else{
      pthread_mutex_unlock(&__lock);
    }
  }
  return ;
  */
  /* //fprintf(stderr, "_xmp_tca_wait(%d, %d)\n", node, tag); */
  /* clock_gettime(CLOCK_MONOTONIC, &begin_ts_2); */
  /* //fprintf(stderr, "_xmp_tca_wait(%d,%d)\n", node, tag); */
  /* volatile int *num_p = &(_postreq.num); */
  /* while(*num_p < 1){ */
  /*   _mm_pause(); */
  /* } */
  /* while(remove_request(node, tag) == false){ */
  /*   //_mm_pause(); */
  /* } */
  /* clock_gettime(CLOCK_MONOTONIC, &end_ts_2); */
  /* //  if(_XMP_world_rank==1) fprintf(stderr, "wait time=%f\n", getElapsedTime_(&begin_ts_2, &end_ts_2)*1000*1000);   */
  /* //if(_XMP_world_rank==1) fprintf(stderr, "between time=%f, %f\n", getElapsedTime_(&begin_ts_2, &begin_ts)*1000*1000, getElapsedTime_(&end_ts, &end_ts_2)*1000*1000); */
}
