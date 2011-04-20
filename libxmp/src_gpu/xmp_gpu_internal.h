#ifndef _XMP_GPU_INTERNAL

#include <stddef.h>
#include "xmp_internal.h"

typedef struct _XMP_gpudata_type {
  _Bool is_aligned_array;

  void *host_addr;
  void *device_addr;

  struct _XMP_gpudata_type *device_gpudata_desc;
  _XMP_array_t *device_array_desc;

  size_t size;
} _XMP_gpudata_t;

// xmp_gpu_runtime.cu
extern int _XMP_gpu_device_count;

// xmp_gpu_util.cu
extern void _XMP_gpu_alloc(void **addr, size_t size);
extern void _XMP_gpu_free(void *addr);

#endif // _XMP_GPU_INTERNAL
