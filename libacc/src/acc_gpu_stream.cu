#include <stdio.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"


//funcs
//cudaStream_t _ACC_gpu_get_stream(int id);
//void _ACC_gpu_init_stream_map(int table_size);
//void _ACC_gpu_finalize_stream_map();
//void _ACC_gpu_wait(int id);
//void _ACC_gpu_wait_all();
//int _ACC_gpu_test(int id);
//int _ACC_gpu_test_all();

static int calc_hash(int id);
static void create_stream(cudaStream_t *stream);
static void destroy_stream(cudaStream_t stream);


typedef struct Cell
{
  int id;
  cudaStream_t stream;
  void *mpool;
  unsigned *block_count;
  struct Cell *next;
}Cell;

typedef Cell** StreamMap;

static StreamMap stream_map = NULL;
static int table_size = 0;
static void* default_mpool;
static unsigned* default_count;

void _ACC_gpu_init_stream_map(int size)
{
  table_size = size;
  if(stream_map != NULL){
    _ACC_gpu_finalize_stream_map();
  }
  stream_map = (StreamMap)_ACC_alloc(table_size * sizeof(Cell *));
  int i;
  for(i=0;i<table_size;i++) stream_map[i] = NULL;
  _ACC_gpu_mpool_alloc_block(&default_mpool);
  _ACC_gpu_calloc((void**)&default_count, sizeof(unsigned));
}

void _ACC_gpu_finalize_stream_map()
{
  int i;
  for(i=0;i<table_size;i++){
    Cell *head = stream_map[i], *cur;
    for(cur = head; cur != NULL; cur = cur->next){
      destroy_stream(cur->stream);
      _ACC_gpu_mpool_free_block(cur->mpool);
      _ACC_gpu_free(cur->block_count);
      _ACC_free(cur);
    }
  }
  _ACC_free(stream_map);
  stream_map = NULL;
  _ACC_gpu_mpool_free_block(default_mpool);
  _ACC_gpu_free(default_count);
}

static Cell* alloc_stream(int id)
{
  Cell *new_cell = (Cell *)_ACC_alloc(sizeof(Cell));
  create_stream(&(new_cell->stream));
  new_cell->id = id;
  _ACC_gpu_mpool_alloc_block(&new_cell->mpool);
  _ACC_gpu_calloc((void**)&new_cell->block_count, sizeof(unsigned));
  return new_cell;
}

cudaStream_t _ACC_gpu_get_stream(int id)
{
  int hash = calc_hash(id);

  Cell *cur;
  Cell *head = stream_map[hash];
  for(cur = head; cur != NULL; cur = cur->next){
    if(cur->id == id){
      return cur->stream;
    }
  }
			  
  //if not found, create & put stream
  Cell *new_cell = alloc_stream(id);
  new_cell->next = head;

  stream_map[hash]=new_cell;

  return new_cell->stream;
}


//wait func
void _ACC_gpu_wait(int id){
  cudaStream_t stream = _ACC_gpu_get_stream(id);
  cudaError_t error = cudaStreamSynchronize(stream);

  if(error != cudaSuccess){
    _ACC_gpu_fatal(error);
    
  }
}

void _ACC_gpu_wait_all(){
  int i;
  for(i=0;i<table_size;i++){
    Cell *head = stream_map[i], *cur;
    for(cur = head; cur != NULL; cur = cur->next){
      //do something
      cudaError_t error = cudaStreamSynchronize(cur->stream);
      if(error != cudaSuccess){
	_ACC_gpu_fatal(error);
      }
    }
  }
}


//test func
int _ACC_gpu_test(int id)
{
  cudaStream_t stream = _ACC_gpu_get_stream(id);
  cudaError_t error = cudaStreamQuery(stream);
  if(error == cudaSuccess){
    return ~0;
  }else{
    return 0;
  }
}

int _ACC_gpu_test_all()
{
  int i;
  //int result = 0;
  for(i=0;i<table_size;i++){
    Cell *head = stream_map[i], *cur;
    for(cur = head; cur != NULL; cur = cur->next){
      //do something
      cudaError_t error = cudaStreamQuery(cur->stream);
      if(error != cudaSuccess){
	return 0;
      }
    }
  }
  return ~0;
}


  
//internal functions
static int calc_hash(int id)
{
  return id%table_size;
}
static void create_stream(cudaStream_t *stream)
{
  cudaError_t error = cudaStreamCreate(stream);
  //error handling
  if(error != cudaSuccess){
    _ACC_fatal("cant create stream\n");
  }
}
static void destroy_stream(cudaStream_t stream)
{
  cudaError_t error = cudaStreamDestroy(stream);
  if(error != cudaSuccess){
    _ACC_fatal("can't destroy stream\n");
  }
}



//for test
static void print()
{
  if(stream_map == NULL){
    printf("no map\n");
    return;
  }
  
  int i;
  for(i=0;i<table_size;i++){
    printf("StreamMap[%d]:", i);
    Cell *head = stream_map[i];
    Cell *cur;
    for(cur = head; cur!=NULL;cur = cur->next){
      printf("(%d, %d)->",cur->id, cur->stream);
    }
    printf("null\n");
  }
}

/*
int main(void) //for test
{
  _ACC_gpu_init_stream_map(4);
  print();

  cudaStream_t a,b,c;
  a = _ACC_gpu_get_stream(3);
  printf("id=3 stream=%lld\n", (long long)a);
  _ACC_gpu_get_stream(5);
  _ACC_gpu_get_stream(1);
  _ACC_gpu_get_stream(3);
  printf("id=3 stream=%lld\n", (long long)a);
  _ACC_gpu_get_stream(7);
  print();

  _ACC_gpu_finalize_stream_map();
  print();
}
*/

void _ACC_gpu_mpool_get(void **ptr)
{
  *ptr = default_mpool;
}
void _ACC_gpu_mpool_get_async(void **ptr, int id)
{
  int hash = calc_hash(id);

  Cell *cur;
  Cell *head = stream_map[hash];
  for(cur = head; cur != NULL; cur = cur->next){
    if(cur->id == id){
      *ptr = cur->mpool;
      return;
    }
  }
			  
  //if not found, create & put stream
  Cell *new_cell = alloc_stream(id);
  new_cell->next = head;
  stream_map[hash]=new_cell;

  *ptr = new_cell->mpool;
}

void _ACC_gpu_get_block_count(unsigned **count)
{
  *count = default_count;
}

void _ACC_gpu_get_block_count_async(unsigned **count, int id)
{
  int hash = calc_hash(id);

  Cell *cur;
  Cell *head = stream_map[hash];
  for(cur = head; cur != NULL; cur = cur->next){
    if(cur->id == id){
      *count = cur->block_count;
      return;
    }
  }
			  
  //if not found, create & put stream
  Cell *new_cell = alloc_stream(id);
  new_cell->next = head;
  stream_map[hash]=new_cell;

  *count = new_cell->block_count;
}
