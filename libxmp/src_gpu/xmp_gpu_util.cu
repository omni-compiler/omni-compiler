#include "xmp_gpu_internal.h"

void _XMP_gpu_alloc(void **addr, size_t size) {
  if (cudaMalloc(addr, size) != cudaSuccess) {
    _XMP_fatal("failed to allocate data on GPU");
  }
}

void _XMP_gpu_free(void *addr) {
  if (cudaFree(addr) != cudaSuccess) {
    _XMP_fatal("failed to free data on GPU");
  }
}
