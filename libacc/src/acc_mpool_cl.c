#include<stdio.h>
#include "acc_internal.h"
#include "acc_internal_cl.h"

#define _ACC_CL_MPOOL_NUM_BLOCKS 8
#define _ACC_CL_MPOOL_BLOCK_SIZE 1024

typedef struct _ACC_mpool_type mpool;
struct _ACC_mpool_type{
  cl_mem ptr[_ACC_CL_MPOOL_NUM_BLOCKS];
  bool flags[_ACC_CL_MPOOL_NUM_BLOCKS];
};


_ACC_mpool_t* _ACC_mpool_create()
{
  _ACC_DEBUG("mpool cl init\n")
  mpool *p = (mpool*)_ACC_alloc(sizeof(mpool));
  
  size_t size = _ACC_CL_MPOOL_BLOCK_SIZE * sizeof(char);
  for(int i=0;i<_ACC_CL_MPOOL_NUM_BLOCKS;i++){
    cl_int ret;
    p->ptr[i] = clCreateBuffer(_ACC_cl_current_context, CL_MEM_READ_WRITE, size, NULL, &ret);
    CL_CHECK(ret);

    p->flags[i]=true;
  }
  return p;
}

void _ACC_mpool_destroy(_ACC_mpool_t *mpool)
{
  for(int i = 0; i < _ACC_CL_MPOOL_NUM_BLOCKS; i++){
    CL_CHECK(clReleaseMemObject(mpool->ptr[i]));
  }
}

void _ACC_mpool_alloc_block(void **ptr)
{
  int i;
  mpool* mpool_p = _ACC_get_mpool();
  for(i=0;i<_ACC_CL_MPOOL_NUM_BLOCKS;i++){
    if(mpool_p->flags[i] == true){
      mpool_p->flags[i] = false;
      *ptr = mpool_p->ptr[i];
      return;
    }
  }

  cl_int ret;
  size_t size = _ACC_CL_MPOOL_BLOCK_SIZE*sizeof(char);
  *ptr = clCreateBuffer(_ACC_cl_current_context, CL_MEM_READ_WRITE, size, NULL, &ret);
  CL_CHECK(ret);
}

void _ACC_mpool_free_block(void *ptr)
{
  mpool* mpool_p = _ACC_get_mpool();

  for(int i = 0; i < _ACC_CL_MPOOL_NUM_BLOCKS; i++){
    if(mpool_p->ptr[i] == ptr){
      mpool_p->flags[i] = true;
      return;
    }
  }

  CL_CHECK(clReleaseMemObject(ptr));
}

void _ACC_mpool_alloc(void **ptr, long long size, void *mpool, long long *pos){
  const int align = 8;
  long long aligned_size = ((size - 1) / align + 1) * align;
  cl_int ret;
  if(*pos + aligned_size <= _ACC_CL_MPOOL_BLOCK_SIZE){
    cl_buffer_region info = {(size_t)(*pos), (size_t)size};
    *ptr = clCreateSubBuffer((cl_mem)mpool, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, &ret); //((char*)mpool) + *pos;
    CL_CHECK(ret);
    *pos += aligned_size;
  }else{
    *ptr = clCreateBuffer(_ACC_cl_current_context, CL_MEM_READ_WRITE, size, NULL, &ret);
    CL_CHECK(ret);
  }
}

void _ACC_mpool_free(void *ptr, void *mpool)
{
  CL_CHECK(clReleaseMemObject((cl_mem)ptr));
}
