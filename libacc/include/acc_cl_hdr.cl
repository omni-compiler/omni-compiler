// macro
#define _ACC_M_FLOORi(a_, b_) ((a_) / (b_))
#define _ACC_M_COUNT_TRIPLETi(l_, u_, s_) ( ((u_) >= (l_))? _ACC_M_FLOORi((u_) - (l_), s_) + 1 : 0)
#define _ACC_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
#define _ACC_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _ACC_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

#define _ACC_block_idx_x (get_group_id(0))   /* blockIdx.x */
#define _ACC_thread_idx_x (get_local_id(0))  /* threadIdx.x */
#define _ACC_grid_dim_x (get_num_groups(0))  /* gridDim.x */
#define _ACC_block_dim_x (get_local_size(0)) /* blockDim.x */

#define _ACC_block_x_id (get_group_id(0))   /* blockIdx.x */
#define _ACC_thread_x_id (get_local_id(0))  /* threadIdx.x */
#define _ACC_grid_x_dim (get_num_groups(0))  /* gridDim.x */

// template<typename T, typename T0>
// static inline 
// void _ACC_init_iter_block_thread_x(T *bt_idx, T *bt_cond, T *bt_step, T0 totalIter)
// {
//   T gang_size = _ACC_M_CEILi(totalIter, /*gridDim.x*/get_maxpid());
//   *bt_idx  = get_pid() * get_maxtid() + get_tid(); //gang_size * /*blockIdx.x*/get_pid() + /*threadIdx.x*/get_tid();
//   *bt_cond = totalIter; //_ACC_M_MIN(gang_size * (/*blockIdx.x*/get_pid() + 1), totalIter);
//   *bt_step = get_maxpid() * get_maxtid(); ///*blockDim.x*/get_maxtid();
// }

inline
void _ACC_init_iter_block_thread_x(unsigned *bt_idx, unsigned *bt_cond, unsigned *bt_step, unsigned totalIter)
{
  unsigned gang_size = _ACC_M_CEILi(totalIter, _ACC_grid_dim_x/*gridDim.x*/);
  *bt_idx  = gang_size * _ACC_block_idx_x /*blockIdx.x*/ + _ACC_thread_idx_x /*threadIdx.x*/;
  *bt_cond = _ACC_M_MIN(gang_size * (_ACC_block_idx_x /*blockIdx.x*/ + 1), totalIter);
  *bt_step = _ACC_block_dim_x /*blockDim.x*/;
}

// template<typename T, typename T0>
// static inline
// void _ACC_init_iter_block_x(T *gang_iter, T *gang_cond, T *gang_step, T0 totaliter){
//   T0 gang_size = _ACC_M_CEILi(totaliter, /*gridDim.x*/get_maxpid());
//   *gang_iter = get_pid(); //gang_size * /*blockIdx.x*/get_pid();
//   *gang_cond = totaliter; //_ACC_M_MIN(*gang_iter + gang_size, totaliter);
//   *gang_step = get_maxpid(); //1;
// }

// template<typename T, typename T0>
// static inline
// void _ACC_init_iter_thread_x(T *iter, T *cond, T *step, T0 totaliter){
//   *iter = /*threadIdx.x*/get_tid();
//   *cond = totaliter;
//   *step = /*blockDim.x*/get_maxtid();
// }

// template<typename T, typename T0, typename T1, typename T2>
// static inline
// void _ACC_calc_niter(T *niter, T0 init, T1 cond, T2 step)
// {
//   *niter = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
// }

inline
void _ACC_calc_niter(unsigned *niter, int init, int cond, int step)
{
   *niter = _ACC_M_COUNT_TRIPLETi(init, cond - 1, step);
}

// template<typename T, typename T0, typename T1, typename T2, typename T3>
// static inline
// void _ACC_calc_idx(T id, T0 *idx, T1 lower, T2 upper, T3 stride)
// {
//   *idx = lower + stride * id;
// }

inline
void _ACC_calc_idx(unsigned id, int *idx, int lower, int upper, int stride)
{
   *idx = lower + stride * id;
}

inline
void _ACC_sync_threads()
{
	//  wait_pe();
}

inline
void _ACC_flush()
{
    mem_fence(CLK_GLOBAL_MEM_FENCE);
	//  flush();
}

void _ACC_sync(int const n)
{
  // switch(n){
  // case 0:
  //   wait_pe();
  //   break;
  // case 1:
  //   wait_village();
  //   break;
  // case 2:
  //   wait_city();
  //   break;
  // case 3:
  //   wait_prefecture();
  //   break;
  // }
}

void _ACC_sync_gangs()
{
  //sync();
  //  flush();
}

void _ACC_sync_all()
{
	//  sync();
}

void _ACC_flush_all()
{
  _ACC_flush();
}

void _ACC_yield()
{
//  chgthread();
}

inline
void _ACC_stack_init(unsigned long *pos)
{
  *pos = 0;
}

inline
void *_ACC_stack_push(void *base, unsigned long *pos, size_t size)
{
  void *ret = (void*)((char*)base + *pos);
  *pos += size;
  return ret;
}

// template<typename T, typename T0, typename T1>
// static inline
// int _ACC_calc_vidx(T *idx, T0 niter, T1 total_idx){
//   *idx = total_idx % niter;
//   return total_idx / niter;
// }

// end: acc_pezy.hpp 

// reduction 
#define _ACC_REDUCTION_PLUS 0
#define _ACC_REDUCTION_MUL 1
#define _ACC_REDUCTION_MAX 2
#define _ACC_REDUCTION_MIN 3
#define _ACC_REDUCTION_BITAND 4
#define _ACC_REDUCTION_BITOR 5
#define _ACC_REDUCTION_BITXOR 6
#define _ACC_REDUCTION_LOGAND 7
#define _ACC_REDUCTION_LOGOR 8

// template<typename T>
// __device__ static inline
// T op(T a, T b, int kind){
//   switch(kind){
//   case _ACC_REDUCTION_PLUS: return a + b;
//   case _ACC_REDUCTION_MUL: return a * b;
//   case _ACC_REDUCTION_MAX: return (a > b)? a : b;
//   case _ACC_REDUCTION_MIN: return (a < b)? a : b;
//   default: return a;
//   }
// }

inline
double op_double(double a, double b, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: return a + b;
  case _ACC_REDUCTION_MUL: return a * b;
  case _ACC_REDUCTION_MAX: return (a > b)? a : b;
  case _ACC_REDUCTION_MIN: return (a < b)? a : b;
  default: return a;
  }
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
typedef ulong  uint_t;
#define ATOMIC_CMPXCHG(p,o,n)	atom_cmpxchg(p,o,n)

// void atom_add_double(double *val, double delta)
// void _ACC_reduce_threads(T *target, T input, int kind, bool do_acc)
inline
void _ACC_reduce_threads_double(volatile double *target, double input, int kind, bool do_acc)
{
    volatile union {
	real_t r;
	uint_t i;
    } next, expected, current;

    current.r = *target;

    do {
	expected.r = current.r;
	next.r     = do_acc? op_double(expected.r,input,kind):input;
	// next.r     = do_acc? (expected.r+input):input;
	current.i  = ATOMIC_CMPXCHG((volatile __global uint_t *) target, expected.i, next.i);
    } while (current.i != expected.i);

#if 0
  union {
  double f;
  ulong  i;
  } old, new;

  do
  {
     old.f = *target;
     new.f = do_acc? op_double(old.f,input,kind):input;
  } 
  while (atomic_cmpxchg((volatile __global ulong *)target, old.i, new.i) != old.i);
#endif
}


inline
void _ACC_gpu_init_reduction_var_double(double *var, int kind){
  switch(kind){
  case _ACC_REDUCTION_PLUS: *var = 0.0; return;
  case _ACC_REDUCTION_MUL: *var = 1.0; return;
  case _ACC_REDUCTION_MAX: *var = -DBL_MAX; return;
  case _ACC_REDUCTION_MIN: *var = DBL_MAX; return;
  }
}

// _ACC_gpu_reduction_t_double(&_ACC_gpu_reduction_tmp_multiplied_total,1,_ACC_reduction_bt_multiplied_total);
// _ACC_gpu_reduction_singleblock_double(multiplied_total,_ACC_gpu_reduction_tmp_multiplied_total,1);
// _ACC_gpu_reduction_tmp_double(_ACC_gpu_reduction_tmp_multiplied_total,_ACC_GPU_RED_TMP,0);
// _ACC_gpu_reduction_block_double(multiplied_total,1,_ACC_GPU_RED_TMP,0,_ACC_GPU_RED_NUM);


// template<typename T>
// __device__ static inline
// void _ACC_gpu_reduction_block(T* result, int kind, void* buf, size_t element_offset, int num_elements){
//  T *data = (T*)((char*)buf + (element_offset * num_elements));
//
//  T part_result;
//  _ACC_gpu_init_reduction_var(&part_result, kind);
//
//  for(int idx = threadIdx.x; idx < num_elements; idx += blockDim.x){
//    part_result = op(part_result, data[idx], kind);
//  }
//
//  _ACC_reduce_threads(result, part_result, kind, CL_TRUE);
// }

inline
void _ACC_gpu_reduction_block_double(double * result, int kind, void* buf, size_t element_offset, int num_elements){
  double *data = (double *)((char*)buf + (element_offset * num_elements));
  double part_result;
  _ACC_gpu_init_reduction_var_double(&part_result, kind);
  for(int idx = _ACC_thread_idx_x /*threadIdx.x*/; idx < num_elements; idx += _ACC_block_dim_x/*blockDim.x*/){
    part_result = op_double(part_result, data[idx], kind);
  }
  _ACC_reduce_threads_double(result, part_result, kind, true);
}

// template<typename T>
// __device__ static inline
// void _ACC_gpu_reduction_block(T* result, int kind, void* tmp, size_toffsetElementSize){
//  _ACC_gpu_reduction_block(result, kind, tmp, offsetElementSize, gridDim.x);
// }

// template<typename T>
// __device__ static inline
// void _ACC_gpu_reduction_singleblock(T* result, T resultInBlock, int kind){
//  if(threadIdx.x == 0){
//    *result = op(*result, resultInBlock, kind);
//  }
// }

inline
void _ACC_gpu_reduction_singleblock_double(double* result, double resultInBlock, int kind){
  if(_ACC_thread_idx_x /*threadIdx.x*/ == 0){
    *result = op_double(*result, resultInBlock, kind);
  }
}

// template<typename T>
// void _ACC_gpu_reduction_t(T *resultInBlock, int kind, T resultInThread){
//  _ACC_reduce_threads(resultInBlock, resultInThread, kind, true);
//}

inline
void _ACC_gpu_reduction_t_double(double *resultInBlock, int kind, double resultInThread){
  _ACC_reduce_threads_double(resultInBlock, resultInThread, kind, true);
}

// template<typename T>
// __device__ static inline
// void _ACC_gpu_reduction_tmp(T resultInBlock, void *tmp, size_t offsetElementSize){
//  if(threadIdx.x==0){
//    void *tmpAddr =  (char*)tmp + (gridDim.x * offsetElementSize);
//    ((T*)tmpAddr)[blockIdx.x] = resultInBlock;
//  }
//  __syncthreads();//is need?
//}

inline
void _ACC_gpu_reduction_tmp_double(double resultInBlock, void *tmp, size_t offsetElementSize){
  if(_ACC_thread_idx_x /*threadIdx.x*/ ==0){
    void *tmpAddr =  (char*) tmp + (_ACC_grid_dim_x /*gridDim.x*/ * offsetElementSize);
    ((double *)tmpAddr)[_ACC_block_idx_x /*blockIdx.x*/] = resultInBlock;
  }
  barrier(CLK_GLOBAL_MEM_FENCE); //  __syncthreads();//is need?
}

//----------------------------------------------------------------

#define _ACC_gpu_init_reduction_var_double_MUL(x)  _ACC_gpu_init_reduction_var_double(x, 1)

#define _ACC_gpu_reduction_loc_set_double_MUL(reduction_loc,result_loc, reduction_tmp) \
  if((_ACC_grid_dim_x)==(1)) reduction_loc = result_loc; \
  else reduction_loc = ((__global double *)reduction_tmp)+_ACC_block_idx_x;\
  if(_ACC_thread_idx_x == 0) _ACC_gpu_init_reduction_var_double(reduction_loc,1); \
  barrier(CLK_GLOBAL_MEM_FENCE); 

#define _ACC_gpu_reduction_loc_update_double_MUL(reduction_loc, reduction_value) \
    _ACC_reduce_threads_double(reduction_loc, reduction_value, 1, true)

#define _ACC_gpu_reduction_loc_block_double_MUL(result_loc, reduction_tmp, off, reduction_tmp_size) \
if((_ACC_thread_idx_x)==(0)) { \
  double part_result; \
  _ACC_gpu_init_reduction_var_double(&part_result, 1); \
  for(int i = 0; i < reduction_tmp_size; i++) \
      part_result = op_double(part_result, ((double *)reduction_tmp)[i], 1); \
  *multiplied_total = op_double(*multiplied_total, part_result, 1); \
  } 
