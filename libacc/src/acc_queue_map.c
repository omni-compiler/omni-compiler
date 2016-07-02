#include <stdio.h>
#include "acc_internal.h"

#define ASYNC_NOVAL_EQUALS_ASYNC_SYNC

typedef struct queue_list{
  int async_num;        //key
  _ACC_queue_t *queue;  //value
  struct queue_list *next;
}queue_list;

struct _ACC_queue_map_type{
  queue_list *queue_lists;
  int hashtable_size;
};

static int calc_hash(int n, int size);
static void add_queue(_ACC_queue_map_t *queue_map, int async_num, _ACC_queue_t *queue);

_ACC_queue_map_t* _ACC_gpu_init_stream_map(int hashtable_size) //FIXME _ACC_queue_map_create
{
  _ACC_DEBUG("init_map\n")
  _ACC_queue_map_t* queue_map = (_ACC_queue_map_t*)_ACC_alloc(sizeof(_ACC_queue_map_t));
  queue_map->queue_lists = (queue_list *)_ACC_alloc(sizeof(queue_list)*hashtable_size);
  queue_map->hashtable_size = hashtable_size;
  
  for(int i=0;i<hashtable_size;i++){
    queue_map->queue_lists[i].async_num = ACC_ASYNC_NULL;
    queue_map->queue_lists[i].next = NULL;
  }

  _ACC_queue_t* async_sync_queue = _ACC_queue_create(ACC_ASYNC_SYNC);
  _ACC_queue_t* async_noval_queue = _ACC_queue_create(ACC_ASYNC_NOVAL);
  add_queue(queue_map, ACC_ASYNC_SYNC, async_sync_queue);
  add_queue(queue_map, ACC_ASYNC_NOVAL, async_noval_queue);

  return queue_map;
}


void _ACC_gpu_finalize_stream_map(_ACC_queue_map_t* queue_map) //FIXME _ACC_queue_map_destroy
{
  if(queue_map == NULL) return;

  int hashtable_size = queue_map->hashtable_size;
  for(int i = 0; i < hashtable_size; i++){
    queue_list *head = queue_map->queue_lists[i].next, *cur, *next;
    for(cur = head; cur != NULL; cur = next){
      next = cur->next;
      _ACC_queue_destroy(cur->queue);
      _ACC_free(cur);
    }
  }
  _ACC_free(queue_map->queue_lists);
  _ACC_free(queue_map);
}

static int calc_hash(int i, int size)
{
  int r = i%size;
  if(r < 0){
    r += size;
  }
  return r;
}

static void add_queue(_ACC_queue_map_t *queue_map, int async_num, _ACC_queue_t *queue)
{
  int hash = calc_hash(async_num, queue_map->hashtable_size);

  queue_list *entry = _ACC_alloc(sizeof(queue_list));

  //set struct members
  entry->async_num = async_num;
  entry->queue = queue;
  entry->next = queue_map->queue_lists[hash].next;

  queue_map->queue_lists[hash].next = entry;
}

_ACC_queue_t* _ACC_queue_map_get_queue(int async_num)
{
#ifdef ASYNC_NOVAL_EQUALS_ASYNC_SYNC
  // this statement will be removed
  if(async_num == ACC_ASYNC_NOVAL) return _ACC_queue_map_get_queue(ACC_ASYNC_SYNC);
#endif
  
  _ACC_queue_map_t *queue_map = _ACC_get_queue_map();
  int hash = calc_hash(async_num, queue_map->hashtable_size);

  for(queue_list *entry = &(queue_map->queue_lists[hash]); entry != NULL; entry = entry->next){
    if(entry->async_num == async_num){
      return entry->queue;
    }
  }

  _ACC_queue_t *new_queue = _ACC_queue_create(async_num);
  add_queue(queue_map, async_num, new_queue);
  return new_queue;
}

void _ACC_queue_map_set_queue(int async_num, _ACC_queue_t* queue)
{
#ifdef ASYNC_NOVAL_EQUALS_ASYNC_SYNC
  // this statement will be removed
  if(async_num == ACC_ASYNC_NOVAL){
    _ACC_queue_map_set_queue(ACC_ASYNC_SYNC, queue);
    return;
  }
#endif
  
  _ACC_queue_map_t *queue_map = _ACC_get_queue_map();
  int hash = calc_hash(async_num, queue_map->hashtable_size);

  for(queue_list *entry = &(queue_map->queue_lists[hash]); entry != NULL; entry = entry->next){
    if(entry->async_num == async_num){
      _ACC_fatal("queue has been registered already");
    }
  }

  add_queue(queue_map, async_num, queue);
}

void _ACC_gpu_wait(int async_num)
{
  _ACC_queue_t* queue = _ACC_queue_map_get_queue(async_num);
  _ACC_queue_wait(queue);
}

void _ACC_gpu_wait_all()
{
  _ACC_queue_map_t *queue_map = _ACC_get_queue_map();
  int hashtable_size = queue_map->hashtable_size;
  for(int i = 0; i < hashtable_size; i++){
    for(queue_list *entry = queue_map->queue_lists[i].next; entry != NULL; entry = entry->next){
#ifdef ASYNC_NOVAL_EQUALS_ASYNC_SYNC
      // this statement will be removed
      if(entry->async_num == ACC_ASYNC_NOVAL) continue;
#endif
      
      _ACC_queue_wait(entry->queue);
    }
  }
}
int _ACC_gpu_test(int async_num)
{
  _ACC_queue_t* queue = _ACC_queue_map_get_queue(async_num);
  return _ACC_queue_test(queue);
}
int _ACC_gpu_test_all()
{
  _ACC_queue_map_t *queue_map = _ACC_get_queue_map();
  int hashtable_size = queue_map->hashtable_size;
  for(int i = 0; i < hashtable_size; i++){
    for(queue_list *entry = queue_map->queue_lists[i].next; entry != NULL; entry = entry->next){
#ifdef ASYNC_NOVAL_EQUALS_ASYNC_SYNC
      // this statement will be removed
      if(entry->async_num == ACC_ASYNC_NOVAL) continue;
#endif
      
      int ret = _ACC_queue_test(entry->queue);
      if(ret == 0) return 0;
    }
  }
  return ~0;
}


void _ACC_mpool_get_async(void **ptr, int async_num)
{
  _ACC_queue_t *queue = _ACC_queue_map_get_queue(async_num);
  *ptr = _ACC_queue_get_mpool(queue);
}
void _ACC_mpool_get(void **ptr)
{
  _ACC_mpool_get_async(ptr, ACC_ASYNC_SYNC);
}
void _ACC_gpu_get_block_count_async(unsigned **count, int async_num)
{
  _ACC_queue_t *queue = _ACC_queue_map_get_queue(async_num);
  *count = _ACC_queue_get_block_count(queue);
}
void _ACC_gpu_get_block_count(unsigned **count)
{
  _ACC_gpu_get_block_count_async(count, ACC_ASYNC_SYNC);
}
