int _XMP_gpu_device_count;

extern "C" void _XMP_gpu_init(void) {
  cudaGetDeviceCount(&_XMP_gpu_device_count);
}

extern "C" void _XMP_gpu_finalize(void) {
  return;
}
