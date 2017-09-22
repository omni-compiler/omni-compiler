#ifndef _XACC_INTERNAL_H
#define _XACC_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//--CUDA------------------------------------------------
#if defined(_XMP_XACC_CUDA)
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(call)						\
  do {                                                                  \
    cudaError_t err = call;						\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString(err) );		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

typedef cudaStream_t _XACC_queue_t;
typedef void* _XACC_memory_t;
#define _XACC_QUEUE_NULL 0

#endif

//--OpenCL------------------------------------------------
#if defined(_XMP_XACC_OPENCL)
#include <CL/cl.h>

#define CL_CHECK(ret)					\
  do{							\
    if(ret != CL_SUCCESS){				\
      fprintf(stderr, "%s:%d,rank=%d,  OpenCL error code %d\n", __FILE__, __LINE__, _XMP_world_rank, ret); \
      exit(1);						\
    }							\
  }while(0)


typedef cl_command_queue _XACC_queue_t;
typedef cl_mem _XACC_memory_t;
#define _XACC_QUEUE_NULL NULL

#endif

//--Common------------------------------------------------

void _XACC_init(void);
void _XACC_util_init(void);

void _XACC_queue_create(_XACC_queue_t *queue);
void _XACC_queue_destroy(_XACC_queue_t *queue);
void _XACC_queue_wait(_XACC_queue_t queue);

void _XACC_memory_alloc(_XACC_memory_t *memory, size_t size);
void _XACC_memory_free(_XACC_memory_t *memory);
void _XACC_memory_read(void *addr, _XACC_memory_t memory, size_t memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking);
void _XACC_memory_write(_XACC_memory_t memory, size_t memory_offset, void *addr, size_t size, _XACC_queue_t queue, bool is_blocking);
void _XACC_memory_copy(_XACC_memory_t dst_memory, size_t dst_memory_offset, _XACC_memory_t src_memory, size_t src_memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking);
void* _XACC_memory_get_address(_XACC_memory_t memory);

void _XACC_host_malloc(void **ptr, size_t size);
void _XACC_host_free(void **ptr);

void _XACC_memory_pack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
			      _XACC_memory_t src_mem, size_t src_offset,
			      size_t blocklength, size_t stride, size_t count,
			      size_t typesize,
			      _XACC_queue_t queue, bool is_blocking);
void _XACC_memory_unpack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
				_XACC_memory_t src_mem, size_t src_offset,
				size_t blocklength, size_t stride, size_t count,
				size_t typesize,
				_XACC_queue_t queue, bool is_blocking);
void _XACC_memory_pack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
			       _XACC_memory_t src0_mem, size_t src0_offset,
			       size_t blocklength0, size_t stride0, size_t count0,
			       _XACC_memory_t dst1_mem, size_t dst1_offset,
			       _XACC_memory_t src1_mem, size_t src1_offset,
			       size_t blocklength1, size_t stride1, size_t count1,
			       size_t typesize,
			       _XACC_queue_t queue, bool is_blocking);
void _XACC_memory_unpack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
				 _XACC_memory_t src0_mem, size_t src0_offset,
				 size_t blocklength0, size_t stride0, size_t count0,
				 _XACC_memory_t dst1_mem, size_t dst1_offset,
				 _XACC_memory_t src1_mem, size_t src1_offset,
				 size_t blocklength1, size_t stride1, size_t count1,
				 size_t typesize,
				 _XACC_queue_t queue, bool is_blocking);


#define _XACC_fatal _XMP_fatal

#endif //_XACC_INTERNAL_H

