extern "C"
{
  void _XMP_gpu_pack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride);
  void _XMP_gpu_unpack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride);
  void _XMP_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void* async_id);
  void _XMP_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void* async_id);
  void _XMP_gpu_pack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
				    char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				    int count, size_t typesize, cudaStream_t st);
  void _XMP_gpu_unpack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
     				      char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				      int count, size_t typesize, cudaStream_t st);
}

static const int numThreads = 128;

template <typename T>
__global__ static
void _XMP_gpu_pack_vector_kernel(T * __restrict__ dst, const T * __restrict__ src, int count, int blocklength_c, long stride_c)
{
  long i_init = blockIdx.y * blockDim.y + threadIdx.y;
  long i_step = gridDim.y * blockDim.y;
  int j_init = blockIdx.x * blockDim.x + threadIdx.x;
  int j_step = gridDim.x * blockDim.x;

  for(int i = i_init; i < count; i += i_step){
    for(int j = j_init; j < blocklength_c; j += j_step){
      *(dst + i * blocklength_c + j) = *(src + i * stride_c + j);
    }
  }
}

void _XMP_gpu_pack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride)
{
  int numBlocks = (count - 1) / numThreads + 1;
  _XMP_gpu_pack_vector_kernel<<<numBlocks, numThreads>>>(dst, src, count, blocklength, stride);
}
#include<stdio.h>
void _XMP_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void *async_id)
{
  const int numThreads = 128; //must be 2^n
  cudaStream_t st = *((cudaStream_t*)async_id);

  int blocklength_c = blocklength / typesize;
  int stride_c = stride / typesize;
  int bx = 1, by;
  int tx = 1, ty;
  int tmp = blocklength_c;
  if(tmp >= numThreads){
    tx = numThreads;
  }else{
    while(tx < tmp){
      tx <<= 1;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridSize(bx,by);
  dim3 blockSize(tx, ty);

  //printf("blocklen=%d, count=%d, grid(%d,%d), block(%d,%d)\n", blocklength_c, count, bx,by,tx,ty);
  switch(typesize){
  case 1:
    _XMP_gpu_pack_vector_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst, (char *)src, count, blocklength_c, stride_c);
    break;
  case 2:
    _XMP_gpu_pack_vector_kernel<short><<<gridSize, blockSize, 0, st>>>((short *)dst, (short *)src, count, blocklength_c, stride_c);
    break;
  case 4:
    _XMP_gpu_pack_vector_kernel<int><<<gridSize, blockSize, 0, st>>>((int *)dst, (int *)src, count, blocklength_c, stride_c);
    break;
  case 8:
    _XMP_gpu_pack_vector_kernel<long long><<<gridSize, blockSize, 0, st>>>((long long *)dst, (long long *)src, count, blocklength_c, stride_c);
    break;
  default:
    _XMP_gpu_pack_vector_kernel<char><<<gridSize, blockSize, 0, st>>>(dst, src, count, blocklength, stride);
  }
}

template <typename T>
__global__ static
void _XMP_gpu_unpack_vector_kernel(T * __restrict__ dst, const T * __restrict__ src, int count, int blocklength_c, long stride_c)
{
  long i_init = blockIdx.y * blockDim.y + threadIdx.y;
  long i_step = gridDim.y * blockDim.y;
  int j_init = blockIdx.x * blockDim.x + threadIdx.x;
  int j_step = gridDim.x * blockDim.x;

  for(int i = i_init; i < count; i += i_step){
    for(int j = j_init; j < blocklength_c; j += j_step){
      *(dst + i * stride_c + j) = *(src + i * blocklength_c + j);
    }
  }
}

void _XMP_gpu_unpack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride)
{
  int numBlocks = (count - 1) / numThreads + 1;
  _XMP_gpu_unpack_vector_kernel<<<numBlocks, numThreads>>>(dst, src, count, blocklength, stride);
}

void _XMP_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void *async_id)
{
  const int numThreads = 128; //must be 2^n
  cudaStream_t st = *((cudaStream_t*)async_id);

  int blocklength_c = blocklength / typesize;
  int stride_c = stride / typesize;
  int bx = 1, by;
  int tx = 1, ty;
  int tmp = blocklength_c;
  if(tmp >= numThreads){
    tx = numThreads;
  }else{
    while(tx < tmp){
      tx <<= 1;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridSize(bx,by);
  dim3 blockSize(tx, ty);

  //printf("blocklen=%d, count=%d, grid(%d,%d), block(%d,%d)\n", blocklength_c, count, bx,by,tx,ty);
  switch(typesize){
  case 1:
    _XMP_gpu_unpack_vector_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst, (char *)src, count, blocklength_c, stride_c);
    break;
  case 2:
    _XMP_gpu_unpack_vector_kernel<short><<<gridSize, blockSize, 0, st>>>((short *)dst, (short *)src, count, blocklength_c, stride_c);
    break;
  case 4:
    _XMP_gpu_unpack_vector_kernel<int><<<gridSize, blockSize, 0, st>>>((int *)dst, (int *)src, count, blocklength_c, stride_c);
    break;
  case 8:
    _XMP_gpu_unpack_vector_kernel<long long><<<gridSize, blockSize, 0, st>>>((long long *)dst, (long long *)src, count, blocklength_c, stride_c);
    break;
  default:
    _XMP_gpu_unpack_vector_kernel<char><<<gridSize, blockSize, 0, st>>>(dst, src, count, blocklength, stride);
  }
}

//////////////

template <typename T>
__global__ static
void memcpy2D2_kernel(T * __restrict__ dst0, const T * __restrict__ src0, int blocklength0, long dst_stride0, long src_stride0,
                      T * __restrict__ dst1, const T * __restrict__ src1, int blocklength1, long dst_stride1, long src_stride1,
					  int count)
{
  long i_init = blockIdx.y * blockDim.y + threadIdx.y;
  long i_step = gridDim.y * blockDim.y;
  int j_init = blockIdx.x * blockDim.x + threadIdx.x;
  int j_step = gridDim.x * blockDim.x;

  if(blockIdx.z == 0){
    for(int i = i_init; i < count; i += i_step){
      for(int j = j_init; j < blocklength0; j += j_step){
        *(dst0 + i * dst_stride0 + j) = *(src0 + i * src_stride0 + j);
      }
    }
  }else{
    for(int i = i_init; i < count; i += i_step){
      for(int j = j_init; j < blocklength1; j += j_step){
        *(dst1 + i * dst_stride1 + j) = *(src1 + i * src_stride1 + j);
      }
    }
  }
}

static void memcpy2D2_async(char * __restrict__ dst0, long dst_stride0, char * __restrict__ src0, long src_stride0, int blocklength0, 
                            char * __restrict__ dst1, long dst_stride1, char * __restrict__ src1, long src_stride1, int blocklength1,
							int count, size_t typesize, cudaStream_t st)
{
  char *dst[2], *src[2];
  dst[0] = dst[1] = NULL;
  src[0] = src[1] = NULL;
  int blocklength_c[2];
  long src_stride_c[2], dst_stride_c[2];

  int numVector = 0;
  if(dst0 != NULL && src0 != NULL){
    dst[numVector] = dst0;
    src[numVector] = src0;
    blocklength_c[numVector] = blocklength0 / typesize;
    src_stride_c[numVector] = src_stride0 / typesize;
    dst_stride_c[numVector] = dst_stride0 / typesize;
    numVector++;
  }
  if(dst1 != NULL && src1 != NULL){
    dst[numVector] = dst1;
    src[numVector] = src1;
    blocklength_c[numVector] = blocklength1 / typesize;
    src_stride_c[numVector] = src_stride1 / typesize;
    dst_stride_c[numVector] = dst_stride1 / typesize;
    numVector++;
  }

  if(numVector == 0) return;

  int bx = 1, by;
  int tx = 1, ty;
  int tmp;
  if(numVector == 1){
    tmp = blocklength_c[0];
  }else{
    tmp = (blocklength_c[0] < blocklength_c[1])? blocklength_c[0] : blocklength_c[1];
  }
  if(tmp >= numThreads){
    tx = numThreads;
  }else{
    while(tx < tmp){
      tx <<= 1;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridSize(bx,by,numVector);
  dim3 blockSize(tx, ty);

  switch(typesize){
  case 1:
    memcpy2D2_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst[0], (char *)src[0], blocklength_c[0], dst_stride_c[0], src_stride_c[0],
														   (char *)dst[1], (char *)src[1], blocklength_c[1], dst_stride_c[1], src_stride_c[1], count);
    break;
  case 2:
    memcpy2D2_kernel<short><<<gridSize, blockSize, 0, st>>>((short *)dst[0], (short *)src[0], blocklength_c[0], dst_stride_c[0], src_stride_c[0],
															(short *)dst[1], (short *)src[1], blocklength_c[1], dst_stride_c[1], src_stride_c[1], count);
    break;
  case 4:
    memcpy2D2_kernel<float><<<gridSize, blockSize, 0, st>>>((float *)dst[0], (float *)src[0], blocklength_c[0], dst_stride_c[0], src_stride_c[0],
															(float *)dst[1], (float *)src[1], blocklength_c[1], dst_stride_c[1], src_stride_c[1], count);
    break;
  case 8:
    memcpy2D2_kernel<double><<<gridSize, blockSize, 0, st>>>((double *)dst[0], (double *)src[0], blocklength_c[0], dst_stride_c[0], src_stride_c[0],
															 (double *)dst[1], (double *)src[1], blocklength_c[1], dst_stride_c[1], src_stride_c[1], count);
    break;
  default:
    memcpy2D2_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst[0], (char *)src[0], blocklength_c[0] * typesize, dst_stride_c[0] * typesize, src_stride_c[0] * typesize,
														   (char *)dst[1], (char *)src[1], blocklength_c[1] * typesize, dst_stride_c[1] * typesize, src_stride_c[1] * typesize, count);
  }
}

void _XMP_gpu_pack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
								 char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
								 int count, size_t typesize, cudaStream_t st)
{
  memcpy2D2_async(dst0, blocklength0, src0, stride0, blocklength0,
				  dst1, blocklength1, src1, stride1, blocklength1,
				  count, typesize, st);
}

void _XMP_gpu_unpack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
								   char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
								   int count, size_t typesize, cudaStream_t st)
{
  memcpy2D2_async(dst0, stride0, src0, blocklength0, blocklength0,
				  dst1, stride1, src1, blocklength1, blocklength1,
				  count, typesize, st);
}
