#include "xmp_internal.h"
#include "xmp_gpu_internal.h"

extern "C" void _XMP_gpu_init_gpudata_NOT_ALIGNED(void **desc, void *addr, size_t size) {
  _XMP_gpudata_t *d = (_XMP_gpudata_t *)_XMP_alloc(sizeof(_XMP_gpudata_t));

  d->host_addr = addr;
  _XMP_gpu_alloc(&(d->device_addr), size);
  d->size = size;

  *desc = d;
}

extern "C" void _XMP_gpu_finalize_gpudata(void *desc) {
  _XMP_gpudata_t *d = (_XMP_gpudata_t *)desc;

  _XMP_gpu_free(d->device_addr);
  _XMP_free(desc);
}
