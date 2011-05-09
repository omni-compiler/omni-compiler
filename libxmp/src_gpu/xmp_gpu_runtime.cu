#include "xmp_gpu_internal.h"

int _XMP_gpu_device_count;

int _XMP_gpu_max_thread;

int _XMP_gpu_max_block_dim_x;
int _XMP_gpu_max_block_dim_y;
int _XMP_gpu_max_block_dim_z;

extern "C" void _XMP_gpu_init(void) {
  // FIXME consider multi-GPU

  cudaGetDeviceCount(&_XMP_gpu_device_count);

  if (_XMP_gpu_device_count == 0) {
    _XMP_fatal("no GPU device found");
  }

  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties(&dev_prop, 0);

  _XMP_gpu_max_thread = dev_prop.maxThreadsPerBlock;

  _XMP_gpu_max_block_dim_x = dev_prop.maxGridSize[0];
  _XMP_gpu_max_block_dim_y = dev_prop.maxGridSize[1];
  _XMP_gpu_max_block_dim_z = dev_prop.maxGridSize[2];
}

extern "C" void _XMP_gpu_finalize(void) {
  return;
}
