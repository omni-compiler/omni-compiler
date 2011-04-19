#include <stdio.h>

int _XMP_gpu_device_count;

extern "C" void _XMP_gpu_init(void) {
  cudaGetDeviceCount(&_XMP_gpu_device_count);

  if (_XMP_gpu_device_count == 0) {
    printf("no device\n");
    return;
  }

  for (int i = 0; i < _XMP_gpu_device_count; i++) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, i);

    printf("device[%d]\n", i);
    printf(" name = %s\n", dev_prop.name);
    printf(" CC = %d.%d \n", dev_prop.major, dev_prop.minor);
  }
}

extern "C" void _XMP_gpu_finalize(void) {
  return;
}
