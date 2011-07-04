#include "xmp_gpu_internal.h"

extern "C" void _XMP_gpu_init_data_NOT_ALIGNED(_XMP_gpu_data_t **host_data_desc,
                                               void **device_addr, void *addr, size_t size) {
  _XMP_gpu_data_t *host_data_d = NULL;

  // alloc desciptors
  host_data_d = (_XMP_gpu_data_t *)_XMP_alloc(sizeof(_XMP_gpu_data_t));

  // init host descriptor
  host_data_d->is_aligned_array = false;
  host_data_d->host_addr = addr;
  _XMP_gpu_alloc(&(host_data_d->device_addr), size);
  host_data_d->host_array_desc = NULL;
  host_data_d->device_array_desc = NULL;
  host_data_d->size = size;

  // init params
  *host_data_desc = host_data_d;
  *device_addr = host_data_d->device_addr;
}

extern "C" void _XMP_gpu_init_data_ALIGNED(_XMP_gpu_data_t **host_data_desc, _XMP_gpu_array_t **device_array_desc,
                                           void **device_addr, void *addr, _XMP_array_t *array_desc) {
  _XMP_gpu_data_t *host_data_d = NULL;
  _XMP_gpu_array_t *device_array_d = NULL;

  int array_dim = array_desc->dim;
  size_t array_size = (array_desc->total_elmts) * (array_desc->type_size);
  size_t device_array_desc_size = sizeof(_XMP_gpu_array_t) * (array_dim);

  // alloc desciptors
  host_data_d = (_XMP_gpu_data_t *)_XMP_alloc(sizeof(_XMP_gpu_data_t));
  _XMP_gpu_alloc((void **)&(device_array_d), device_array_desc_size);

  // init host descriptor
  host_data_d->is_aligned_array = true;
  host_data_d->host_addr = addr;
  _XMP_gpu_alloc(&(host_data_d->device_addr), array_size);
  host_data_d->host_array_desc = array_desc;
  host_data_d->device_array_desc = device_array_d;
  host_data_d->size = array_size;

  // init device descriptor
  _XMP_gpu_array_t *host_array_d = (_XMP_gpu_array_t *)_XMP_alloc(device_array_desc_size);
  for (int i = 0; i < array_dim; i++) {
    _XMP_array_info_t *ai = &(array_desc->info[i]);
    host_array_d[i].gtol = ai->temp0_v;
    host_array_d[i].acc = ai->dim_acc;
  }

  cudaMemcpy(device_array_d, host_array_d, device_array_desc_size, cudaMemcpyHostToDevice);
  _XMP_free(host_array_d);

  // init params
  *host_data_desc = host_data_d;
  *device_array_desc = device_array_d;
  *device_addr = host_data_d->device_addr;
}

extern "C" void _XMP_gpu_finalize_data(_XMP_gpu_data_t *desc) {
  _XMP_gpu_free(desc->device_addr);

  if (desc->is_aligned_array) {
    _XMP_gpu_free(desc->device_array_desc);
  }

  _XMP_free(desc);
}
