#include "xmp_gpu_internal.h"

int _XMP_gpu_device_count;

extern "C" void _XMP_gpu_init(void) {
  cudaGetDeviceCount(&_XMP_gpu_device_count);

  if (_XMP_gpu_device_count == 0) {
    _XMP_fatal("no GPU device found");
  }
}

extern "C" void _XMP_gpu_finalize(void) {
  return;
}
