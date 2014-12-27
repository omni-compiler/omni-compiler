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

//static StreamMap stream_map = NULL;
static const int table_size = 16;

static Cell* alloc_cell(int id);
static void free_cell(Cell* cell);
static void add_cell(StreamMap stream_map, int id, Cell *cell);

//static Cell* async_sync_cell;
//static Cell* async_noval_cell;

static Cell* alloc_cell(int id)
{
  Cell *new_cell = (Cell *)_ACC_alloc(sizeof(Cell));
  if(id != ACC_ASYNC_SYNC){
    create_stream(&(new_cell->stream));
  }else{
    new_cell->stream = 0;
  }
  new_cell->id = id;
  _ACC_gpu_mpool_alloc_block(&new_cell->mpool);
  _ACC_gpu_calloc((void**)&new_cell->block_count, sizeof(unsigned));
  return new_cell;
}

static void free_cell(Cell* cell)
{
  if(cell == NULL) return;
  if(cell->id != ACC_ASYNC_SYNC){
    destroy_stream(cell->stream);
  }
  _ACC_gpu_mpool_free_block(cell->mpool);
  _ACC_gpu_free(cell->block_count);
  _ACC_free(cell);
}

void* _ACC_gpu_init_stream_map(int size)
{
  _ACC_DEBUG("init_map\n")
  //table_size = size;
  StreamMap map;
  map = (StreamMap)_ACC_alloc(table_size * sizeof(Cell *));
  int i;
  for(i=0;i<table_size;i++) map[i] = NULL;

  //stream_map = map;

  Cell* async_sync_cell = alloc_cell(ACC_ASYNC_SYNC);
  Cell* async_noval_cell = alloc_cell(ACC_ASYNC_NOVAL);
  add_cell(map, ACC_ASYNC_SYNC, async_sync_cell);
  add_cell(map, ACC_ASYNC_NOVAL, async_noval_cell);

  return map;
}

void _ACC_gpu_finalize_stream_map(void* map)
{
  //printf("finalize map\n");
  int i;
  if(map == NULL) return;
  StreamMap st_map = (StreamMap)map;
  for(i=0;i<table_size;i++){
    Cell *head = st_map[i], *cur, *next;
    for(cur = head; cur != NULL; cur = next){
      next = cur->next;
      free_cell(cur);
      cur = NULL;
    }
  }
  _ACC_free(st_map);
}

// void _ACC_gpu_set_stream_map(void* map)
// {
//   //stream_map = (StreamMap)map;
  

//   int hash = calc_hash(ACC_ASYNC_SYNC);
//   for(Cell *cur = stream_map[hash]; cur != NULL; cur = cur->next){
//     if(cur->id == ACC_ASYNC_SYNC){
//       async_sync_cell = cur;
//     }
//   }

//   hash = calc_hash(ACC_ASYNC_NOVAL);
//   for(Cell *cur = stream_map[hash]; cur != NULL; cur = cur->next){
//     if(cur->id == ACC_ASYNC_NOVAL){
//       async_noval_cell = cur;
//     }
//   }
// }

static void add_cell(StreamMap stream_map, int id, Cell *cell)
{
  int hash = calc_hash(id);
  //  StreamMap stream_map = (StreamMap)_ACC_gpu_get_current_stream_map();
  cell->next = stream_map[hash];
  stream_map[hash] = cell;
}

static Cell* get_cell(int id)
{
  //printf("get_cell(%d)\n", id);
  // if(id == ACC_ASYNC_SYNC || id == ACC_ASYNC_NOVAL){
  //   return async_sync_cell;
  // }
  if(id == ACC_ASYNC_NOVAL) return get_cell(ACC_ASYNC_SYNC);
  
  int hash = calc_hash(id);
  StreamMap stream_map = (StreamMap)_ACC_gpu_get_current_stream_map();

  for(Cell *cur = stream_map[hash]; cur != NULL; cur = cur->next){
    if(cur->id == id){
      return cur;
    }
  }

  Cell *new_cell = alloc_cell(id);
  add_cell(stream_map, id, new_cell);
  return new_cell;
}



cudaStream_t _ACC_gpu_get_stream(int id)
{
  Cell *cell = get_cell(id);
  return cell->stream;
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
  StreamMap stream_map = (StreamMap)_ACC_gpu_get_current_stream_map();
  for(i=0;i<table_size;i++){
    Cell *head = stream_map[i], *cur;
    for(cur = head; cur != NULL; cur = cur->next){
      //do something
      if(cur->id == ACC_ASYNC_NOVAL) continue;
      cudaError_t error = cudaStreamSynchronize(cur->stream);
      if(error != cudaSuccess){
	_ACC_gpu_fatal(error);
      }
    }
  }
}

/*
void _ACC_gpu_wait_async(int id1, int id2){
  //id2 waits completion of id1)
  if(id1 == id2){
    _ACC_gpu_wait(id1);
    return;
  }

  cudaStream_t stream1 = _ACC_gpu_getstream(id1);
  cudaStream_t stream2 = _ACC_gpu_getstream(id2);
  cudaEvent_t waitEvent;
  cudaEventCreate(&waitEvent);
}
*/

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
  StreamMap stream_map = (StreamMap)_ACC_gpu_get_current_stream_map();
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
  int r = id%table_size;
  if(r < 0){
    r += table_size;
  }
  return r;
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


/*
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
*/

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
  //*ptr = async_sync_cell->mpool;
  _ACC_gpu_mpool_get_async(ptr, ACC_ASYNC_SYNC);
}
void _ACC_gpu_mpool_get_async(void **ptr, int id)
{
  Cell *cell = get_cell(id);
  *ptr = cell->mpool;
}

void _ACC_gpu_get_block_count(unsigned **count)
{
  //*count = async_sync_cell->block_count;
  _ACC_gpu_get_block_count_async(count, ACC_ASYNC_SYNC);
}

void _ACC_gpu_get_block_count_async(unsigned **count, int id)
{
  Cell *cell = get_cell(id);
  *count = cell->block_count;
}
