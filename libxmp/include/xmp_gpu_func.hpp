/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_GPU_RUNTIME_FUNC_DECL
#define _XMP_GPU_RUNTIME_FUNC_DECL

#include "xmp_constant.h"
#include "xmp_data_struct.h"
#include "xmp_index_macro.h"

// - index functions -----------------------------------------------------------------------------------------------
#define _XMP_GPU_M_GTOL(_desc, _dim) \
(((_XMP_gpu_array_t *)_desc)[_dim].gtol)
#define _XMP_GPU_M_ACC(_desc, _dim) \
(((_XMP_gpu_array_t *)_desc)[_dim].acc)

// --- integer functions
// calculate ceil(a/b)
#define _XMP_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
// calculate floor(a/b)
#define _XMP_M_FLOORi(a_, b_) ((a_) / (b_))
#define _XMP_M_COUNT_TRIPLETi(l_, u_, s_) (_XMP_M_FLOORi(((u_) - (l_)), s_) + 1)

// --- cuda barrier functions
#define _XMP_GPU_M_BARRIER_THREADS() __syncthreads()
#define _XMP_GPU_M_BARRIER_KERNEL() cudaThreadSynchronize()

// --- get array info functions
#define _XMP_GPU_M_GET_ARRAY_GTOL(_gtol, _desc, _dim) \
_gtol = _XMP_GPU_M_GTOL(_desc, _dim)
#define _XMP_GPU_M_GET_ARRAY_ACC(_acc, _desc, _dim) \
_acc = _XMP_GPU_M_ACC(_desc, _dim)

extern "C" void _XMP_fatal(char *msg);

extern int _XMP_gpu_max_thread;

extern int _XMP_gpu_max_block_dim_x;
extern int _XMP_gpu_max_block_dim_y;
extern int _XMP_gpu_max_block_dim_z;

template<typename T>
__device__ void _XMP_gpu_calc_thread_id(T *index) {
  *index = threadIdx.x +
          (threadIdx.y * blockDim.x) +
          (threadIdx.z * blockDim.x * blockDim.y) +
         ((blockIdx.x +
          (blockIdx.y * gridDim.x) +
          (blockIdx.z * gridDim.x * gridDim.y)) * (blockDim.x * blockDim.y * blockDim.z));
}

template<typename T>
__device__ void _XMP_gpu_calc_iter(unsigned long long tid,
                                   T lower0, T upper0, T stride0,
                                   T *iter0) {
  *iter0 = lower0 + (tid * stride0);
}

template<typename T>
__device__ void _XMP_gpu_calc_iter(unsigned long long tid,
                                   T lower0, T upper0, T stride0,
                                   T lower1, T upper1, T stride1,
                                   T *iter0,
                                   T *iter1) {
  T count0 = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);

  *iter0 = lower0 + ((tid % count0) * stride0);
  *iter1 = lower1 + ((tid / count0) * stride1);
}

template<typename T>
__device__ void _XMP_gpu_calc_iter(unsigned long long tid,
                                   T lower0, T upper0, T stride0,
                                   T lower1, T upper1, T stride1,
                                   T lower2, T upper2, T stride2,
                                   T *iter0,
                                   T *iter1,
                                   T *iter2) {
  T count0 = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  T count1 = _XMP_M_COUNT_TRIPLETi(lower1, (upper1 - 1), stride1);

  T temp1 = tid / count0;
  *iter0 = lower0 + ((tid % count0) * stride0);
  *iter1 = lower1 + ((temp1 % count1) * stride1);
  *iter2 = lower2 + ((temp1 / count1) * stride2);
}

#define _XMP_gpu_calc_iter_MAP_THREADS_1(_l0, _u0, _s0, _i0) \
{ \
  if ((blockIdx.x * blockDim.x + threadIdx.x) >= _XMP_M_COUNT_TRIPLETi(_l0, (_u0 - 1), _s0)) return; \
  \
  _i0 = _l0 + ((blockIdx.x * blockDim.x + threadIdx.x) * _s0); \
}

#define _XMP_gpu_calc_iter_MAP_THREADS_2(_l0, _u0, _s0, _l1, _u1, _s1, _i0, _i1) \
{ \
  if ((blockIdx.x * blockDim.x + threadIdx.x) >= _XMP_M_COUNT_TRIPLETi(_l0, (_u0 - 1), _s0)) return; \
  if ((blockIdx.y * blockDim.y + threadIdx.y) >= _XMP_M_COUNT_TRIPLETi(_l1, (_u1 - 1), _s1)) return; \
  \
  _i0 = _l0 + ((blockIdx.x * blockDim.x + threadIdx.x) * _s0); \
  _i1 = _l1 + ((blockIdx.y * blockDim.y + threadIdx.y) * _s1); \
}

#define _XMP_gpu_calc_iter_MAP_THREADS_3(_l0, _u0, _s0, _l1, _u1, _s1, _l2, _u2, _s2, _i0, _i1, _i2) \
{ \
  if ((blockIdx.x * blockDim.x + threadIdx.x) >= _XMP_M_COUNT_TRIPLETi(_l0, (_u0 - 1), _s0)) return; \
  if ((blockIdx.y * blockDim.y + threadIdx.y) >= _XMP_M_COUNT_TRIPLETi(_l1, (_u1 - 1), _s1)) return; \
  if ((blockIdx.z * blockDim.z + threadIdx.z) >= _XMP_M_COUNT_TRIPLETi(_l2, (_u2 - 1), _s2)) return; \
  \
  _i0 = _l0 + ((blockIdx.x * blockDim.x + threadIdx.x) * _s0); \
  _i1 = _l1 + ((blockIdx.y * blockDim.y + threadIdx.y) * _s1); \
  _i2 = _l2 + ((blockIdx.z * blockDim.z + threadIdx.z) * _s2); \
}

#define _XMP_GPU_M_CALC_CONFIG_PARAMS(_x, _y, _z) \
{ \
  unsigned long long num_threads = _x * _y * _z; \
\
  *total_iter = total_iter_v; \
\
  *thread_x = _x; \
  *thread_y = _y; \
  *thread_z = _z; \
\
  if (num_threads > _XMP_gpu_max_thread) { \
    _XMP_fatal("too many threads are requested for GPU"); \
  } \
\
  if (num_threads >= total_iter_v) { \
    *block_x = 1; \
    *block_y = 1; \
    *block_z = 1; \
    return; \
  } \
\
  total_iter_v = _XMP_M_CEILi(total_iter_v, num_threads); \
\
  if (total_iter_v > _XMP_gpu_max_block_dim_x) { \
    *block_x = _XMP_gpu_max_block_dim_x; \
\
    total_iter_v = _XMP_M_CEILi(total_iter_v, _XMP_gpu_max_block_dim_x); \
    if (total_iter_v > _XMP_gpu_max_block_dim_y) { \
      *block_y = _XMP_gpu_max_block_dim_y; \
\
      total_iter_v = _XMP_M_CEILi(total_iter_v, _XMP_gpu_max_block_dim_y); \
      if (total_iter_v > _XMP_gpu_max_block_dim_z) { \
        _XMP_fatal("data is too big for GPU"); \
      } else { \
        *block_z = total_iter_v; \
      } \
    } else { \
      *block_y = total_iter_v; \
      *block_z = 1; \
    } \
  } else { \
    *block_x = total_iter_v; \
    *block_y = 1; \
    *block_z = 1; \
  } \
}

template<typename T>
void _XMP_gpu_calc_config_params(unsigned long long *total_iter,
                                 int *block_x, int *block_y, int *block_z,
                                 int *thread_x, int *thread_y, int *thread_z,
                                 T lower0, T upper0, T stride0) {
  unsigned long long total_iter_v = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  _XMP_GPU_M_CALC_CONFIG_PARAMS(16, 16, 1);
}

template<typename T>
void _XMP_gpu_calc_config_params(unsigned long long *total_iter,
                                 int *block_x, int *block_y, int *block_z,
                                 int *thread_x, int *thread_y, int *thread_z,
                                 T lower0, T upper0, T stride0,
                                 T lower1, T upper1, T stride1) {
  unsigned long long total_iter_v = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0)
                                  * _XMP_M_COUNT_TRIPLETi(lower1, (upper1 - 1), stride1);
  _XMP_GPU_M_CALC_CONFIG_PARAMS(16, 16, 1);
}

template<typename T>
void _XMP_gpu_calc_config_params(unsigned long long *total_iter,
                                 int *block_x, int *block_y, int *block_z,
                                 int *thread_x, int *thread_y, int *thread_z,
                                 T lower0, T upper0, T stride0,
                                 T lower1, T upper1, T stride1,
                                 T lower2, T upper2, T stride2) {
  unsigned long long total_iter_v = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0)
                                  * _XMP_M_COUNT_TRIPLETi(lower1, (upper1 - 1), stride1)
                                  * _XMP_M_COUNT_TRIPLETi(lower2, (upper2 - 1), stride2);
  _XMP_GPU_M_CALC_CONFIG_PARAMS(16, 16, 1);
}

template<typename T>
void _XMP_gpu_calc_config_params_MAP_THREADS(int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v,
                                             T lower0, T upper0, T stride0) {
  T iter_x = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);

  *thread_x = thread_x_v;
  *thread_y = 1;
  *thread_z = 1;

  *block_x = _XMP_M_CEILi(iter_x, thread_x_v);
  *block_y = 1;
  *block_z = 1;
}

template<typename T>
void _XMP_gpu_calc_config_params_MAP_THREADS(int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v, int thread_y_v,
                                             T lower0, T upper0, T stride0,
                                             T lower1, T upper1, T stride1) {
  T iter_x = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  T iter_y = _XMP_M_COUNT_TRIPLETi(lower1, (upper1 - 1), stride1);

  *thread_x = thread_x_v;
  *thread_y = thread_y_v;
  *thread_z = 1;

  *block_x = _XMP_M_CEILi(iter_x, thread_x_v);
  *block_y = _XMP_M_CEILi(iter_y, thread_y_v);
  *block_z = 1;
}

template<typename T>
void _XMP_gpu_calc_config_params_MAP_THREADS(int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v, int thread_y_v, int thread_z_v,
                                             T lower0, T upper0, T stride0,
                                             T lower1, T upper1, T stride1,
                                             T lower2, T upper2, T stride2) {
  T iter_x = _XMP_M_COUNT_TRIPLETi(lower0, (upper0 - 1), stride0);
  T iter_y = _XMP_M_COUNT_TRIPLETi(lower1, (upper1 - 1), stride1);
  T iter_z = _XMP_M_COUNT_TRIPLETi(lower2, (upper2 - 1), stride2);

  *thread_x = thread_x_v;
  *thread_y = thread_y_v;
  *thread_z = thread_z_v;

  *block_x = _XMP_M_CEILi(iter_x, thread_x_v);
  *block_y = _XMP_M_CEILi(iter_y, thread_y_v);
  *block_z = _XMP_M_CEILi(iter_z, thread_z_v);
}

#endif // _XMP_GPU_RUNTIME_FUNC_DECL
