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

template<typename T>
__device__ void _XMP_gpu_calc_thread_id(T *index) {
  *index = threadIdx.x;
}

template<typename T>
void _XMP_gpu_calc_config_params(int *block_x, int *block_y, int *block_z,
                                 int *thread_x, int *thread_y, int *thread_z,
                                 T lower, T upper, T stride) {
  *block_x = 1;
  *block_y = 1;
  *block_z = 1;
  *thread_x = _XMP_M_COUNT_TRIPLETi(lower, upper, stride);
  *thread_y = 1;
  *thread_z = 1;
}

template<typename T>
void _XMP_gpu_calc_config_params_NUM_THREADS(int *block_x, int *block_y, int *block_z,
                                             int *thread_x, int *thread_y, int *thread_z,
                                             int thread_x_v, int thread_y_v, int thread_z_v,
                                             T lower, T upper, T stride) {
  *block_x = 1;
  *block_y = 1;
  *block_z = 1;
  *thread_x = thread_x_v;
  *thread_y = thread_y_v;
  *thread_z = thread_z_v;
}

#endif // _XMP_GPU_RUNTIME_FUNC_DECL
