extern "C"
{
  void _XMP_gpu_pack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride);
  void _XMP_gpu_unpack_vector(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride);
  void _XMP_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void* async_id);
  void _XMP_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, void* async_id);
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
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
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
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
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

