#ifndef _ACC_GPU_FUNC
#define _ACC_GPU_FUNC

#include "acc_gpu_reduction.hpp"

// calculate floor(a/b)
#define _ACC_M_FLOORi(a_, b_) ((a_) / (b_))
#define _ACC_M_COUNT_TRIPLETi(l_, u_, s_) ( ((u_) >= (l_))? _ACC_M_FLOORi((u_) - (l_), s_) + 1 : 0)
#define _ACC_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _ACC_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _ACC_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

// --- cuda barrier functions
#define _ACC_GPU_M_BARRIER_THREADS() __syncthreads()
#define _ACC_GPU_M_BARRIER_KERNEL() cudaThreadSynchronize()

// --- cuda block and thread idx
#define _ACC_block_x_id blockIdx.x
#define _ACC_block_y_id blockIdx.y
#define _ACC_block_z_id blockIdx.z
#define _ACC_thread_x_id threadIdx.x
#define _ACC_thread_y_id threadIdx.y
#define _ACC_thread_z_id threadIdx.z
#define _ACC_grid_x_dim gridDim.x

template<typename T, typename T0>
__device__
static void _ACC_gpu_init_block_x_iter(T *gang_iter, T *gang_cond, T *gang_step, T0 totaliter){
  T0 gang_size = _ACC_M_CEILi(totaliter, gridDim.x);
  *gang_iter = gang_size * blockIdx.x;
  *gang_cond = _ACC_M_MIN(*gang_iter + gang_size, totaliter);
  *gang_step = 1;
}

template<typename T, typename T0>
__device__
static void _ACC_gpu_init_thread_x_iter(T *iter, T *cond, T *step, T0 totaliter){
  *iter = threadIdx.x;
  *cond = totaliter;
  *step = blockDim.x;
}

template<typename T, typename T0>
__device__
static void _ACC_gpu_init_block_thread_x_iter(T *bt_idx, T *bt_cond, T *bt_step, T0 totalIter){
  T gang_size = _ACC_M_CEILi(totalIter, gridDim.x);
  *bt_idx  = gang_size * blockIdx.x + threadIdx.x;
  *bt_cond = _ACC_M_MIN(gang_size * (blockIdx.x + 1), totalIter);
  *bt_step = blockDim.x;
}

__device__
static void _ACC_gpu_init_block_thread_y_iter(int *bt_idx, int *bt_cond, int *bt_step, int lower, int upper, int strid){
  int totalIter = _ACC_M_COUNT_TRIPLETi(lower, upper - 1, strid);
  int gang_size = _ACC_M_CEILi(totalIter, gridDim.y);
  *bt_idx  = gang_size * blockIdx.y + threadIdx.y;
  //  *bt_cond = _ACC_M_MIN(*bt_idx + gang_size, totalIter); //incorrect!
  *bt_cond = _ACC_M_MIN(gang_size * (blockIdx.y + 1), totalIter);
  *bt_step = blockDim.y;
}

template<typename T, typename T0, typename T1, typename T2, typename T3>
__device__
static void _ACC_gpu_calc_idx(T id, T0 *idx, T1 lower, T2 upper, T3 stride){
  *idx = lower + stride * id;
}

template<typename T>
__device__
static void _ACC_gpu_calc_gang_iter(unsigned long long gid, T lower, T upper, T stride, T *iter)
{
  *iter = lower + (gid * stride);
}
 
template<typename T>
__device__
static void _ACC_gpu_calc_worker_iter(unsigned long long wid, T lower, T upper, T stride, T *iter)
{
  *iter = lower + (wid * stride);
}


template<typename T>
static void _ACC_gpu_calc_block_params(unsigned long long *total_iter,
				int *block_x, int *block_y, int *block_z,
				T lower0, T upper0, T stride0) {
  *total_iter = _ACC_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  *block_x = *total_iter;
  *block_y = 1;
  *block_z = 1;
}

template<typename T>
static void _ACC_gpu_calc_thread_params(unsigned long long *total_iter,
				 int *thread_x, int *thread_y, int *thread_z,
				 T lower0, T upper0, T stride0) {
  *total_iter = _ACC_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  *thread_x = (*total_iter > 128)? 128 : *total_iter;
  *thread_y = 1;
  *thread_z = 1;
}

static void _ACC_GPU_ADJUST_GRID(int *gridX,int *gridY, int *gridZ, int limit){
  int total = *gridX * *gridY * *gridZ;
  if(total > limit){
    *gridZ = _ACC_M_MAX(1, *gridZ/_ACC_M_CEILi(total,limit));
    total = *gridX * *gridY * *gridZ;

    if(total > limit){
      *gridY = _ACC_M_MAX(1, *gridY/_ACC_M_CEILi(total,limit));
      total = *gridX * *gridY;
      
      if(total > limit){
	*gridX = _ACC_M_CEILi(*gridX, _ACC_M_CEILi(total,limit));
      }
    }
  }
  
  /*
  while(total > limit){
    if(*gridZ > 1){
    *gridZ /= 2;
    }else if(*gridY > 1){
    *gridY /= 2;
    }else{
    *gridX /= 2;
    }
    total = *gridX * *gridY * *gridZ;
    }*/
}

__device__
static void _ACC_calc_niter(int *niter, int init, int cond, int step){
  *niter = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
}

template<typename T, typename T0, typename T1, typename T2>
__device__
static void _ACC_calc_niter(T *niter, T0 init, T1 cond, T2 step){
  *niter = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
}

__device__
static int _ACC_calc_vidx(int *idx, int niter, int total_idx){
  *idx = total_idx % niter;
  return total_idx / niter;
}

template<typename T, typename T0, typename T1>
__device__
static int _ACC_calc_vidx(T *idx, T0 niter, T1 total_idx){
  *idx = total_idx % niter;
  return total_idx / niter;
}

template<typename T, typename T0, typename T1, typename T2>
__device__
static void _ACC_gpu_init_iter_cnt(T *cnt, T0 init, T1 cond, T2 step){
  *cnt = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
}

__device__
static void _ACC_init_private(void **p_prv, void *array, size_t size){
  *p_prv = (char *)array + size * blockIdx.x;
}

#endif //_ACC_GPU_FUNC
