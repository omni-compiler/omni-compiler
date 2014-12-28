#include "acc_internal.h"
#include "acc_gpu_internal.h"

//static void* mpool_ptr = NULL;
//static char mpool_flags[_ACC_GPU_MPOOL_NUM_BLOCKS];
//static char* mpool_flags = NULL;

typedef struct mpool{
  void *ptr;
  char flags[_ACC_GPU_MPOOL_NUM_BLOCKS];
}mpool;

//static mpool *mpool_p;

//void _ACC_gpu_mpool_init();
//void _ACC_gpu_mpool_finalize();
void _ACC_gpu_mpool_alloc_block(void **);
void _ACC_gpu_mpool_free_block(void *);
#include<stdio.h>
void* _ACC_gpu_mpool_init()
{
  _ACC_DEBUG("mpool init\n")
  mpool *p = (mpool*)_ACC_alloc(sizeof(mpool));
  _ACC_gpu_alloc(&(p->ptr), _ACC_GPU_MPOOL_BLOCK_SIZE * _ACC_GPU_MPOOL_NUM_BLOCKS * sizeof(char));
  for(int i=0;i<_ACC_GPU_MPOOL_NUM_BLOCKS;i++){
    p->flags[i]=~0;
  }
  //mpool_p = p;
  return p;
}

void _ACC_gpu_mpool_finalize(void *p)
{
  mpool* mpool_ptr = (mpool*)p;
  if(mpool_ptr->ptr != NULL){
    _ACC_gpu_free(mpool_ptr->ptr);
  }
  mpool_ptr->ptr = NULL;
}

void _ACC_gpu_mpool_set(void* p)
{
  mpool* mpool_p = (mpool*)_ACC_gpu_get_current_mpool();
  mpool_p = (mpool*)p;
}

void _ACC_gpu_mpool_alloc_block(void **ptr)
{
  int i;
  mpool* mpool_p = (mpool*)_ACC_gpu_get_current_mpool();
  for(i=0;i<_ACC_GPU_MPOOL_NUM_BLOCKS;i++){
    if(mpool_p->flags[i]){
      mpool_p->flags[i] = 0;
      *ptr = ((char*)mpool_p->ptr) + _ACC_GPU_MPOOL_BLOCK_SIZE * i;
      return;
    }
  }
  _ACC_gpu_alloc(ptr, _ACC_GPU_MPOOL_BLOCK_SIZE*sizeof(char));
  return;
}

void _ACC_gpu_mpool_free_block(void *ptr)
{
  mpool* mpool_p = (mpool*)_ACC_gpu_get_current_mpool();
  long long i = ((long long)((char*)ptr - (char*)mpool_p->ptr)) / _ACC_GPU_MPOOL_BLOCK_SIZE;
  if(i>=0 && i<_ACC_GPU_MPOOL_NUM_BLOCKS){
    mpool_p->flags[i] = ~0;
  }else{
    _ACC_gpu_free(ptr);
  }
}

void _ACC_gpu_mpool_alloc(void **ptr, long long size, void *mpool, long long *pos){
  const int align = 8;
  long long aligned_size = ((size - 1) / align + 1) * align;
  if(*pos + aligned_size <= _ACC_GPU_MPOOL_BLOCK_SIZE){
    *ptr = ((char*)mpool) + *pos;
    *pos += aligned_size;
  }else{
    _ACC_gpu_alloc(ptr, size);
  }
}

void _ACC_gpu_mpool_free(void *ptr, void *mpool)
{
  long long pos = (long long)((char*)ptr - (char*)mpool);
  if(pos < 0 || pos >= _ACC_GPU_MPOOL_BLOCK_SIZE){
    _ACC_gpu_free(ptr);
  }
}
