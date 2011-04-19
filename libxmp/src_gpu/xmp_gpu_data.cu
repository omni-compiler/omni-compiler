// FIXME for debug
#include <stdio.h>

#include "xmp_internal.h"
#include "xmp_gpu_internal.h"

extern "C" void _XMP_gpu_init_gpudata_NOT_ALIGNED(_XMP_gpudata_t **desc, void *addr, size_t size) {
  _XMP_gpudata_t *d = (_XMP_gpudata_t *)_XMP_alloc(sizeof(_XMP_gpudata_t));

  d->host_addr = addr;
  _XMP_gpu_alloc(&(d->device_addr), size);
  d->size = size;

  *desc = d;

// FIXME for debug
  printf("[%d] gpu alloc = %lu\n", _XMP_world_rank, size);
}

extern "C" void _XMP_gpu_init_gpudata_ALIGNED(_XMP_gpudata_t **gpudata_desc, void *addr, _XMP_array_t *array_desc) {
  _XMP_gpudata_t *d = (_XMP_gpudata_t *)_XMP_alloc(sizeof(_XMP_gpudata_t));

  size_t size = (array_desc->total_elmts) * (array_desc->type_size);

  d->host_addr = addr;
  _XMP_gpu_alloc(&(d->device_addr), size);
  d->size = size;

  *gpudata_desc = d;

// FIXME for debug
  printf("[%d] gpu alloc = %lu\n", _XMP_world_rank, size);
}

extern "C" void _XMP_gpu_finalize_gpudata(_XMP_gpudata_t *desc) {
  _XMP_gpu_free(desc->device_addr);
  _XMP_free(desc);
}
