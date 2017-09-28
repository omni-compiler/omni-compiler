#include "xmp_internal.h"
#include "xacc_internal.h"

void _XACC_util_init(void){}

void _XACC_queue_create(_XACC_queue_t *queue)
{
    CUDA_SAFE_CALL(cudaStreamCreate(queue));
}

void _XACC_queue_destroy(_XACC_queue_t *queue)
{
    if(*queue != _XACC_QUEUE_NULL){
	CUDA_SAFE_CALL(cudaStreamDestroy(*queue));
	*queue = _XACC_QUEUE_NULL;
    }
}

void _XACC_queue_wait(_XACC_queue_t queue)
{
    CUDA_SAFE_CALL(cudaStreamSynchronize(queue));
}



void _XACC_memory_alloc(_XACC_memory_t *memory, size_t size)
{
    CUDA_SAFE_CALL(cudaMalloc(memory, size));
}

void _XACC_memory_free(_XACC_memory_t *memory)
{
    CUDA_SAFE_CALL(cudaFree(*memory));
    *memory = NULL;
}

static
void memory_copy(void *dst, void *src, size_t size, _XACC_queue_t queue, bool is_blocking){
    CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, queue));

    if(is_blocking){
	_XACC_queue_wait(queue);
    }
}

void _XACC_memory_read(void *addr, _XACC_memory_t memory, size_t memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    void *src_addr = (char*)memory + memory_offset;
    memory_copy(addr, src_addr, size, queue, is_blocking);
}

void _XACC_memory_write(_XACC_memory_t memory, size_t memory_offset, void *addr, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    void *dst_addr = (char*)memory + memory_offset;
    memory_copy(dst_addr, addr, size, queue, is_blocking);
}

void _XACC_memory_copy(_XACC_memory_t dst_memory, size_t dst_memory_offset, _XACC_memory_t src_memory, size_t src_memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    void *dst_addr = (char*)dst_memory + dst_memory_offset;
    void *src_addr = (char*)src_memory + src_memory_offset;
    memory_copy(dst_addr, src_addr, size, queue, is_blocking);
}

void _XACC_host_malloc(void **ptr, size_t size)
{
    *ptr = _XMP_alloc(size);
    CUDA_SAFE_CALL(cudaHostRegister(*ptr, size, cudaHostRegisterDefault));

    //CUDA_SAFE_CALL(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
}

void _XACC_host_free(void **ptr)
{
    if(*ptr != NULL) CUDA_SAFE_CALL(cudaHostUnregister(*ptr));
    _XMP_free(*ptr);

    //CUDA_SAFE_CALL(cudaFreeHost(*ptr));

    *ptr = NULL;
}

void* _XACC_memory_get_address(_XACC_memory_t memory)
{
    return (void*)memory;
}


// pack/unpack functions
void _XACC_memory_pack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
			      _XACC_memory_t src_mem, size_t src_offset,
			      size_t blocklength, size_t stride, size_t count,
			      size_t typesize,
			      _XACC_queue_t queue, bool is_blocking)
{
    void _XMP_gpu_pack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);

    _XMP_gpu_pack_vector_async((char * restrict)dst_mem + dst_offset,
			       (char * restrict)src_mem + src_offset,
			       count, blocklength, stride,
			       typesize,
			       &queue);
}

void _XACC_memory_unpack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
				_XACC_memory_t src_mem, size_t src_offset,
				size_t blocklength, size_t stride, size_t count,
				size_t typesize,
				_XACC_queue_t queue, bool is_blocking)
{
    void _XMP_gpu_unpack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);
    _XMP_gpu_unpack_vector_async((char * restrict)dst_mem + dst_offset,
				 (char * restrict)src_mem + src_offset,
				 count, blocklength, stride,
				 typesize,
				 &queue);
}

void _XACC_memory_pack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
			       _XACC_memory_t src0_mem, size_t src0_offset,
			       size_t blocklength0, size_t stride0, size_t count0,
			       _XACC_memory_t dst1_mem, size_t dst1_offset,
			       _XACC_memory_t src1_mem, size_t src1_offset,
			       size_t blocklength1, size_t stride1, size_t count1,
			       size_t typesize,
			       _XACC_queue_t queue, bool is_blocking)
{
    void _XMP_gpu_pack_vector2_async(char * restrict dst0, char * restrict src0, int blocklength0, long stride0,
				     char * restrict dst1, char * restrict src1, int blocklength1, long stride1,
				     int count, size_t typesize, cudaStream_t st);
    if(count0 != count1){
	_XACC_fatal("two counts of vectors must be same");
    }

    char * restrict dst0 = (dst0_mem != NULL)? (char * restrict)dst0_mem + dst0_offset : NULL;
    char * restrict src0 = (src0_mem != NULL)? (char * restrict)src0_mem + src0_offset : NULL;
    char * restrict dst1 = (dst1_mem != NULL)? (char * restrict)dst1_mem + dst1_offset : NULL;
    char * restrict src1 = (src1_mem != NULL)? (char * restrict)src1_mem + src1_offset : NULL;

    _XMP_gpu_pack_vector2_async(dst0,
				src0,
				blocklength0, stride0,
				dst1,
				src1,
				blocklength1, stride1,
				count0,
				typesize,
				queue);
}

void _XACC_memory_unpack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
				 _XACC_memory_t src0_mem, size_t src0_offset,
				 size_t blocklength0, size_t stride0, size_t count0,
				 _XACC_memory_t dst1_mem, size_t dst1_offset,
				 _XACC_memory_t src1_mem, size_t src1_offset,
				 size_t blocklength1, size_t stride1, size_t count1,
				 size_t typesize,
				 _XACC_queue_t queue, bool is_blocking)
{
    void _XMP_gpu_unpack_vector2_async(char * restrict dst0, char * restrict src0, int blocklength0, long stride0,
				       char * restrict dst1, char * restrict src1, int blocklength1, long stride1,
				       int count, size_t typesize, cudaStream_t st);
    if(count0 != count1){
	_XACC_fatal("two counts of vectors must be same");
    }

    char * restrict dst0 = (dst0_mem != NULL)? (char * restrict)dst0_mem + dst0_offset : NULL;
    char * restrict src0 = (src0_mem != NULL)? (char * restrict)src0_mem + src0_offset : NULL;
    char * restrict dst1 = (dst1_mem != NULL)? (char * restrict)dst1_mem + dst1_offset : NULL;
    char * restrict src1 = (src1_mem != NULL)? (char * restrict)src1_mem + src1_offset : NULL;

    _XMP_gpu_unpack_vector2_async(dst0,
				  src0,
				  blocklength0, stride0,
				  dst1,
				  src1,
				  blocklength1, stride1,
				  count0,
				  typesize,
				  queue);
}
