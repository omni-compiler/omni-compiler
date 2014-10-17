#include <cuda_runtime.h>
#include <stdio.h>

static const int numThreads = 128;

extern "C"
{
  void _XACC_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
  void _XACC_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
}

template <typename T>
__global__ static
void memcpy2D_kernel(T * __restrict__ dst, const T * __restrict__ src, int count, int blocklength_c, long dst_stride_c, long src_stride_c)
{
  long i_init = blockIdx.y * blockDim.y + threadIdx.y;
  long i_step = gridDim.y * blockDim.y;
  int j_init = blockIdx.x * blockDim.x + threadIdx.x;
  int j_step = gridDim.x * blockDim.x;

  for(int i = i_init; i < count; i += i_step){
    for(int j = j_init; j < blocklength_c; j += j_step){
      *(dst + i * dst_stride_c + j) = *(src + i * src_stride_c + j);
    }
  }
}

static void memcpy2D_async(char * __restrict__ dst, long dst_stride, char * __restrict__ src, long src_stride, int blocklength, int count, size_t typesize, cudaStream_t st)
{
  int blocklength_c = blocklength / typesize;
  int src_stride_c = src_stride / typesize;
  int dst_stride_c = dst_stride / typesize;
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
    memcpy2D_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst, (char *)src, count, blocklength_c, dst_stride_c, src_stride_c);
    break;
  case 2:
    memcpy2D_kernel<short><<<gridSize, blockSize, 0, st>>>((short *)dst, (short *)src, count, blocklength_c, dst_stride_c, src_stride_c);
    break;
  case 4:
    memcpy2D_kernel<int><<<gridSize, blockSize, 0, st>>>((int *)dst, (int *)src, count, blocklength_c, dst_stride_c, src_stride_c);
    break;
  case 8:
    memcpy2D_kernel<long long><<<gridSize, blockSize, 0, st>>>((long long *)dst, (long long *)src, count, blocklength_c, dst_stride_c, src_stride_c);
    break;
  default:
    memcpy2D_kernel<char><<<gridSize, blockSize, 0, st>>>(dst, src, count, blocklength, dst_stride, src_stride);
  }
}

void _XACC_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t st)
{
  memcpy2D_async(dst, blocklength, src, stride, blocklength, count, typesize, st);
}

void _XACC_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t st)
{
  memcpy2D_async(dst, stride, src, blocklength, blocklength, count, typesize, st);
}