#ifndef _XMP_GPU_INTERNAL

#include <stddef.h>
#include "xmp_internal.h"

// xmp_gpu_runtime.cu
extern int _XMP_gpu_device_count;
extern int _XMP_gpu_max_thread;
extern int _XMP_gpu_max_block_dim_x;
extern int _XMP_gpu_max_block_dim_y;
extern int _XMP_gpu_max_block_dim_z;

// xmp_gpu_util.cu
extern void _XMP_gpu_alloc(void **addr, size_t size);
extern void _XMP_gpu_free(void *addr);

// xmp_gpu_array_section.cu
extern void _XMP_gpu_pack_array(void *host_buffer, void *array_addr, size_t type_size, size_t alloc_size,
                                int array_dim, int *lower, int *upper, int *stride, unsigned long long *dim_acc);

#endif // _XMP_GPU_INTERNAL
