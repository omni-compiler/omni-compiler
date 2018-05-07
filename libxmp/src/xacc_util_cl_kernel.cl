#define SIZE_T unsigned long

#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long

__kernel
void _XACC_pack_vector(__global char *dst, SIZE_T dst_off, __global const char *src, SIZE_T src_off, SIZE_T blocklength, SIZE_T stride, SIZE_T count)
{
    const size_t c_init = get_global_id(1);
    const size_t c_step = get_global_size(1);
    const size_t b_init = get_global_id(0);
    const size_t b_step = get_global_size(0);
    size_t c, b;
    for(c = c_init; c < count; c += c_step){
	for(b = b_init; b < blocklength; b += b_step){
	    dst[dst_off + blocklength * c + b] = src[src_off + stride * c + b];
	}
    }
}

#define PACK_VECTOR(N) \
__kernel \
void _XACC_pack_vector_##N (__global int##N##_t *dst, SIZE_T dst_off_e, __global const int##N##_t *src, SIZE_T src_off_e, SIZE_T blocklength_e, SIZE_T stride_e, SIZE_T count)\
{\
    const size_t c_init = get_global_id(1);\
    const size_t c_step = get_global_size(1);\
    const size_t b_init = get_global_id(0);\
    const size_t b_step = get_global_size(0);\
    size_t c,b;\
    for(c = c_init; c < count; c += c_step){\
	for(b = b_init; b < blocklength_e; b += b_step){\
	    dst[dst_off_e + blocklength_e * c + b] = src[src_off_e + stride_e * c + b];\
	}\
    }\
}

PACK_VECTOR(8)
PACK_VECTOR(16)
PACK_VECTOR(32)
PACK_VECTOR(64)

__kernel
void _XACC_unpack_vector(__global char *dst, SIZE_T dst_off, __global const char *src, SIZE_T src_off, SIZE_T blocklength, SIZE_T stride, SIZE_T count)
{
    const size_t c_init = get_global_id(1);
    const size_t c_step = get_global_size(1);
    const size_t b_init = get_global_id(0);
    const size_t b_step = get_global_size(0);

    size_t c, b;
    for(c = c_init; c < count; c += c_step){
	for(b = b_init; b < blocklength; b += b_step){
	    dst[dst_off + stride * c + b] = src[src_off + blocklength * c + b];
	}
    }
}

#define UNPACK_VECTOR(N) \
__kernel \
void _XACC_unpack_vector_##N (__global int##N##_t *dst, SIZE_T dst_off_e, __global const int##N##_t *src, SIZE_T src_off_e, SIZE_T blocklength_e, SIZE_T stride_e, SIZE_T count)\
{\
    const size_t c_init = get_global_id(1);\
    const size_t c_step = get_global_size(1);\
    const size_t b_init = get_global_id(0);\
    const size_t b_step = get_global_size(0);\
    size_t c,b;\
    for(c = c_init; c < count; c += c_step){\
	for(b = b_init; b < blocklength_e; b += b_step){\
	    dst[dst_off_e + stride_e * c + b] = src[src_off_e + blocklength_e * c + b];\
	}\
    }\
}

UNPACK_VECTOR(8)
UNPACK_VECTOR(16)
UNPACK_VECTOR(32)
UNPACK_VECTOR(64)

