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
          (blockIdx.z * gridDim.x * gridDim.y)) * (blockDim.z * blockDim.y * blockDim.x));
}

template<typename T>
void _XMP_gpu_calc_config_params(int *block_x, int *block_y, int *block_z,
                                 int *thread_x, int *thread_y, int *thread_z,
                                 T lower, T upper, T stride) {
  unsigned long long num_elmts = _XMP_M_COUNT_TRIPLETi(lower, upper, stride);

  *thread_x = 16;
  *thread_y = 16;
  *thread_z = 1;

  if ((16 * 16) >= num_elmts) {
    *block_x = 1;
    *block_y = 1;
    *block_z = 1;
    return;
  }

  num_elmts = _XMP_M_CEILi(num_elmts, (16 * 16));

  // calc block_x, block_y, block_z
  if (num_elmts > _XMP_gpu_max_block_dim_x) {
    *block_x = _XMP_gpu_max_block_dim_x;

    num_elmts = _XMP_M_CEILi(num_elmts, _XMP_gpu_max_block_dim_x);
    if (num_elmts > _XMP_gpu_max_block_dim_y) {
      *block_y = _XMP_gpu_max_block_dim_y;

      num_elmts = _XMP_M_CEILi(num_elmts, _XMP_gpu_max_block_dim_y);
      if (num_elmts > _XMP_gpu_max_block_dim_z) {
        _XMP_fatal("data is too big for GPU");
      } else {
        *block_z = num_elmts;
      }
    } else {
      *block_y = num_elmts;
      *block_z = 1;
    }
  } else {
    *block_x = num_elmts;
    *block_y = 1;
    *block_z = 1;
  }
}

template<typename T>
void _XMP_gpu_calc_config_params_NUM_THREADS(int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v, int thread_y_v, int thread_z_v,
                                             T lower, T upper, T stride) {
  unsigned long long num_elmts = _XMP_M_COUNT_TRIPLETi(lower, upper, stride);
  unsigned long long num_threads = thread_x_v * thread_y_v * thread_z_v;

  *thread_x = thread_x_v;
  *thread_y = thread_y_v;
  *thread_z = thread_z_v;

  if (num_threads > _XMP_gpu_max_thread) {
    _XMP_fatal("too many threads are requested for GPU");
  }

  if (num_threads >= num_elmts) {
    *block_x = 1;
    *block_y = 1;
    *block_z = 1;
    return;
  }

  num_elmts = _XMP_M_CEILi(num_elmts, num_threads);

  // calc block_x, block_y, block_z
  if (num_elmts > _XMP_gpu_max_block_dim_x) {
    *block_x = _XMP_gpu_max_block_dim_x;
    
    num_elmts = _XMP_M_CEILi(num_elmts, _XMP_gpu_max_block_dim_x);
    if (num_elmts > _XMP_gpu_max_block_dim_y) {
      *block_y = _XMP_gpu_max_block_dim_y;

      num_elmts = _XMP_M_CEILi(num_elmts, _XMP_gpu_max_block_dim_y);
      if (num_elmts > _XMP_gpu_max_block_dim_z) {
        _XMP_fatal("data is too big for GPU");
      } else {
        *block_z = num_elmts;
      }
    } else {
      *block_y = num_elmts;
      *block_z = 1;
    }
  } else {
    *block_x = num_elmts;
    *block_y = 1;
    *block_z = 1;
  }
}

#endif // _XMP_GPU_RUNTIME_FUNC_DECL
