#ifndef _XMP_GPU_INTERNAL

#include <stddef.h>

typedef struct _XMP_gpudata_type {
  void *host_addr;
  void *device_addr;
  size_t size;
} _XMP_gpudata_t;

// xmp_gpu_runtime.cu
extern int _XMP_gpu_device_count;

// xmp_gpu_util.cu
extern void _XMP_gpu_alloc(void **addr, size_t size);
extern void _XMP_gpu_free(void *addr);

#endif // _XMP_GPU_INTERNAL
