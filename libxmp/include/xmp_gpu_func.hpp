/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_GPU_RUNTIME_FUNC_DECL
#define _XMP_GPU_RUNTIME_FUNC_DECL

// --- integer functions
// calculate ceil(a/b)
#define _XMP_M_CEILi(a_, b_) (((a_) % (b_)) == 0 ? ((a_) / (b_)) : ((a_) / (b_)) + 1)
// calculate floor(a/b)
#define _XMP_M_FLOORi(a_, b_) ((a_) / (b_))
#define _XMP_M_COUNTi(a_, b_) ((b_) - (a_) + 1)
#define _XMP_M_COUNT_TRIPLETi(l_, u_, s_) (_XMP_M_FLOORi(((u_) - (l_)), s_) + 1)

// --- generic functions
#define _XMP_M_MAX(a_, b_) ((a_) > (b_) ? (a_) : (b_))
#define _XMP_M_MIN(a_, b_) ((a_) > (b_) ? (b_) : (a_))

// --- cuda barrier func
#define _XMP_GPU_M_BARRIER_THREADS() __syncthreads()
#define _XMP_GPU_M_BARRIER_KERNEL() cudaThreadSynchronize()

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
                                 T lower, T upper, T stride) {
  unsigned long long total_iter_v = _XMP_M_COUNT_TRIPLETi(lower, (upper - 1), stride);
  _XMP_GPU_M_CALC_CONFIG_PARAMS(16, 16, 1);
}

template<typename T>
void _XMP_gpu_calc_config_params_NUM_THREADS(unsigned long long *total_iter,
                                             int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v, int thread_y_v, int thread_z_v,
                                             T lower, T upper, T stride) {
  unsigned long long total_iter_v = _XMP_M_COUNT_TRIPLETi(lower, (upper - 1), stride);
  _XMP_GPU_M_CALC_CONFIG_PARAMS(thread_x_v, thread_y_v, thread_z_v);
}

#endif // _XMP_GPU_RUNTIME_FUNC_DECL
