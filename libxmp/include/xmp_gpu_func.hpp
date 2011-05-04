/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_GPU_RUNTIME_FUNC_DECL
#define _XMP_GPU_RUNTIME_FUNC_DECL

template<typename T>
__device__ void _XMP_gpu_calc_thread_id(T *index) {
  *index = threadIdx.x;
}

#endif // _XMP_GPU_RUNTIME_FUNC_DECL
