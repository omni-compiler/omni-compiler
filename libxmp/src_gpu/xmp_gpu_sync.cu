#include "xmp_constant.h"
#include "xmp_gpu_internal.h"

extern "C" void _XMP_gpu_sync(_XMP_gpu_data_t *desc, int direction) {
  void *host_addr = desc->host_addr;
  void *device_addr = desc->device_addr;
  size_t size = desc->size;

  if (direction == _XMP_N_GPUSYNC_IN) {
    cudaMemcpy(device_addr, host_addr, size, cudaMemcpyHostToDevice);
  } else if (direction == _XMP_N_GPUSYNC_OUT) {
    cudaMemcpy(host_addr, device_addr, size, cudaMemcpyDeviceToHost);
  } else {
    _XMP_fatal("unknown clause in 'gpu sync' directive");
  }
}
