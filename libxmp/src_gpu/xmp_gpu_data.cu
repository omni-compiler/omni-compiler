#include "xmp_gpu_internal.h"

extern "C" void _XMP_gpu_init_gpudata_NOT_ALIGNED(_XMP_gpudata_t **host_desc, _XMP_gpudata_t **device_desc, void **device_addr, void *addr, size_t size) {
  _XMP_gpudata_t *host_d = NULL;
  _XMP_gpudata_t *device_d = NULL;

  // alloc desciptors
  host_d = (_XMP_gpudata_t *)_XMP_alloc(sizeof(_XMP_gpudata_t));
  _XMP_gpu_alloc((void **)&device_d, sizeof(_XMP_gpudata_t));

  // init host descriptor
  host_d->is_aligned_array = false;

  host_d->host_addr = addr;
  _XMP_gpu_alloc(&(host_d->device_addr), size);

  host_d->device_gpudata_desc = device_d;
  host_d->device_array_desc = NULL;

  host_d->size = size;

  *host_desc = host_d;
  *device_desc = device_d;
  *device_addr = host_d->device_addr;

  // init device descriptor
  cudaMemcpy(device_d, host_d, sizeof(_XMP_gpudata_t), cudaMemcpyHostToDevice);
}

extern "C" void _XMP_gpu_init_gpudata_ALIGNED(_XMP_gpudata_t **host_gpudata_desc, _XMP_gpudata_t **device_gpudata_desc, void **device_addr, void *addr, _XMP_array_t *array_desc) {
  _XMP_gpudata_t *host_d = NULL;
  _XMP_gpudata_t *device_d = NULL;
  _XMP_array_t *device_a = NULL;

  size_t array_size = (array_desc->total_elmts) * (array_desc->type_size);
  size_t array_desc_size = sizeof(_XMP_array_t) + sizeof(_XMP_array_info_t) * (array_desc->dim - 1);

  // alloc desciptors
  host_d = (_XMP_gpudata_t *)_XMP_alloc(sizeof(_XMP_gpudata_t));
  _XMP_gpu_alloc((void **)&device_d, sizeof(_XMP_gpudata_t));
  _XMP_gpu_alloc((void **)&(device_a), array_desc_size);

  // init host descriptor
  host_d->is_aligned_array = true;

  host_d->host_addr = addr;
  _XMP_gpu_alloc(&(host_d->device_addr), array_size);

  host_d->device_gpudata_desc = device_d;
  host_d->device_array_desc = device_a;

  host_d->size = array_size;

  *host_gpudata_desc = host_d;
  *device_gpudata_desc = device_d;
  *device_addr = host_d->device_addr;

  // init device descriptor
  cudaMemcpy(device_d, host_d, sizeof(_XMP_gpudata_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_a, array_desc, array_desc_size, cudaMemcpyHostToDevice);
}

extern "C" void _XMP_gpu_finalize_gpudata(_XMP_gpudata_t *desc) {
  _XMP_gpu_free(desc->device_addr);
  _XMP_gpu_free(desc->device_gpudata_desc);

  if (desc->is_aligned_array) {
    _XMP_gpu_free(desc->device_array_desc);
  }

  _XMP_free(desc);
}
