#include "acc_gpu_internal.h"

static void* mpool_ptr = NULL;
static char mpool_flags[_ACC_GPU_MPOOL_NUM_BLOCKS];

void _ACC_gpu_mpool_init();
void _ACC_gpu_mpool_finalize();
void _ACC_gpu_mpool_alloc_block(void **);
void _ACC_gpu_mpool_free_block(void *);

void _ACC_gpu_mpool_init()
{
  _ACC_gpu_alloc(&mpool_ptr, _ACC_GPU_MPOOL_BLOCK_SIZE * _ACC_GPU_MPOOL_NUM_BLOCKS * sizeof(char));
  int i;
  for(i=0;i<_ACC_GPU_MPOOL_NUM_BLOCKS;i++){
    mpool_flags[i]=~0;
  }
}

void _ACC_gpu_mpool_finalize()
{
  if(mpool_ptr != NULL){
    _ACC_gpu_free(mpool_ptr);
  }
  mpool_ptr = NULL;
}

void _ACC_gpu_mpool_alloc_block(void **ptr)
{
  int i;
  for(i=0;i<_ACC_GPU_MPOOL_NUM_BLOCKS;i++){
    if(mpool_flags[i]){
      mpool_flags[i] = 0;
      *ptr = ((char*)mpool_ptr) + _ACC_GPU_MPOOL_BLOCK_SIZE * i;
      return;
    }
  }
  _ACC_gpu_alloc(ptr, _ACC_GPU_MPOOL_BLOCK_SIZE*sizeof(char));
  return;
}

void _ACC_gpu_mpool_free_block(void *ptr)
{
  long long i = ((long long)((char*)ptr - (char*)mpool_ptr)) / _ACC_GPU_MPOOL_BLOCK_SIZE;
  if(i>=0 && i<_ACC_GPU_MPOOL_NUM_BLOCKS){
    mpool_flags[i] = ~0;
  }else{
    _ACC_gpu_free(ptr);
  }
}
