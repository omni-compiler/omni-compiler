int _XMP_gpu_device_count;

void _XMP_gpu_init(void) {
  cudaGetDeviceCount(&_XMP_gpu_device_count);
}

void _XMP_gpu_finalize(void) {
  return;
}
