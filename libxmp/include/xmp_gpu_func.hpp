/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_GPU_RUNTIME_FUNC_DECL
#define _XMP_GPU_RUNTIME_FUNC_DECL

// FIXME include header
#include <stdbool.h>
// aligned array descriptor
typedef struct _XMP_array_info_type {
  _Bool is_shadow_comm_member;
  _Bool is_regular_chunk;
  int align_manner;

  int ser_lower;
  int ser_upper;
  int ser_size;

  // enable when is_allocated is true
  int par_lower;
  int par_upper;
  int par_stride;
  int par_size;

  int local_lower;
  int local_upper;
  int local_stride;
  int alloc_size;

  int *temp0;
  int temp0_v;

  unsigned long long dim_acc;
  unsigned long long dim_elmts;
  // --------------------------------

  long long align_subscript;

  int shadow_type;
  int shadow_size_lo;
  int shadow_size_hi;

  // enable when is_shadow_comm_member is true
  void *shadow_comm;
  int shadow_comm_size;
  int shadow_comm_rank;
  // -----------------------------------------

  // align_manner is not _XMP_N_ALIGN_NOT_ALIGNED
  int align_template_index;
  void *align_template_info;
  void *align_template_chunk;
  // --------------------------------------------
} _XMP_array_info_t;

typedef struct _XMP_array_type {
  _Bool is_allocated;
  _Bool is_align_comm_member;
  int dim;
  int type;
  size_t type_size;

  // enable when is_allocated is true
  unsigned long long total_elmts;
  // --------------------------------

  // enable when is_align_comm_member is true
  void *align_comm;
  int align_comm_size;
  int align_comm_rank;
  // ----------------------------------------

  void *align_template;
  _XMP_array_info_t info[1];
} _XMP_array_t;

typedef struct _XMP_gpu_data_type {
  _Bool is_aligned_array;

  void *host_addr;
  void *device_addr;

  struct _XMP_gpu_data_type *device_gpu_data_desc;
  _XMP_array_t *device_array_desc;

  size_t size;
} _XMP_gpu_data_t;

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
__device__ void _XMP_gpu_calc_index(unsigned long long *index, T t, void *gpu_data_desc) {
  _XMP_array_t *array_desc = ((_XMP_gpu_data_t *)gpu_data_desc)->device_array_desc;
  *index = t - array_desc->info[0].temp0_v;
}

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
                                   T lower, T upper, T stride,
                                   T *iter) {
  *iter = lower + (((T)tid) * stride);
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
