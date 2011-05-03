// FIXME incomplete
template<typename T>
__device__ void _XMP_gpu_calc_thread_id(T *index) {
  *index = threadIdx.x;
}
