#include <cuda_runtime.h>
#include <stdio.h>

static const int numThreads = 128;

extern "C"
{
  void _XACC_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
  void _XACC_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
  void _XACC_gpu_pack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
				    char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				    int count, size_t typesize, cudaStream_t st);
  void _XACC_gpu_unpack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
     				      char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				      int count, size_t typesize, cudaStream_t st);

}

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()						\
  do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */			\
    cudaError_t err = cudaGetLastError();				\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString(err) );		\
      exit(EXIT_FAILURE);						\
    }									\
    /* Check asynchronous errors, i.e. kernel failed (ULF) */		\
    err = cudaThreadSynchronize();					\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString( err) );		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)


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
  //CHECK_LAUNCH_ERROR();
}

void _XACC_gpu_pack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t st)
{
  memcpy2D_async(dst, blocklength, src, stride, blocklength, count, typesize, st);
}

void _XACC_gpu_unpack_vector_async(char * __restrict__ dst, char * __restrict__ src, int count, int blocklength, long stride, size_t typesize, cudaStream_t st)
{
  memcpy2D_async(dst, stride, src, blocklength, blocklength, count, typesize, st);
}



///////

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
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
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

  //CHECK_LAUNCH_ERROR();
}

void _XACC_gpu_pack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
     				  char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				  int count, size_t typesize, cudaStream_t st)
{
  memcpy2D2_async(dst0, blocklength0, src0, stride0, blocklength0,
		  dst1, blocklength1, src1, stride1, blocklength1,
		  count, typesize, st);
}

void _XACC_gpu_unpack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
     				    char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				    int count, size_t typesize, cudaStream_t st)
{
  memcpy2D2_async(dst0, stride0, src0, blocklength0, blocklength0,
  		  dst1, stride1, src1, blocklength1, blocklength1,
		  count, typesize, st);
}

/*
static void memcpy2D2_async(char * __restrict__ dst0, long dst_stride0, char * __restrict__ src0, long src_stride0, int blocklength0, 
                            char * __restrict__ dst1, long dst_stride1, char * __restrict__ src1, long src_stride1, int blocklength1,
			    int count, size_t typesize, cudaStream_t st)
{
  int blocklength0_c = blocklength0 / typesize;
  int src_stride0_c = src_stride0 / typesize;
  int dst_stride0_c = dst_stride0 / typesize;
  int blocklength1_c = blocklength1 / typesize;
  int src_stride1_c = src_stride1 / typesize;
  int dst_stride1_c = dst_stride1 / typesize;
  int bx = 1, by;
  int tx = 1, ty;
  int tmp = (blocklength0_c > blocklength1_c)? blocklength0_c : blocklength1_c;
  while(tmp > 1){
    tmp = (tmp - 1)/2 + 1;
    tx *= 2;
    if(tx >= numThreads){
      break;
    }
  }
  ty = numThreads / tx;
  by = (count-1)/ty + 1;
  dim3 gridSize(bx,by,2);
  dim3 blockSize(tx, ty);

  switch(typesize){
  case 1:
    memcpy2D2_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst0, (char *)src0, blocklength0_c, dst_stride0_c, src_stride0_c,
    				       		           (char *)dst1, (char *)src1, blocklength1_c, dst_stride1_c, src_stride1_c, count);
    break;
  case 2:
    memcpy2D2_kernel<short><<<gridSize, blockSize, 0, st>>>((short *)dst0, (short *)src0, blocklength0_c, dst_stride0_c, src_stride0_c,
							    (short *)dst1, (short *)src1, blocklength1_c, dst_stride1_c, src_stride1_c, count);
    break;
  case 4:
    memcpy2D2_kernel<int><<<gridSize, blockSize, 0, st>>>((int *)dst0, (int *)src0, blocklength0_c, dst_stride0_c, src_stride0_c,
							  (int *)dst1, (int *)src1, blocklength1_c, dst_stride1_c, src_stride1_c, count);
    break;
  case 8:
    memcpy2D2_kernel<long long><<<gridSize, blockSize, 0, st>>>((long long *)dst0, (long long *)src0, blocklength0_c, dst_stride0_c, src_stride0_c,
								(long long *)dst1, (long long *)src1, blocklength1_c, dst_stride1_c, src_stride1_c, count);
    break;
  default:
    memcpy2D2_kernel<char><<<gridSize, blockSize, 0, st>>>((char *)dst0, (char *)src0, blocklength0, dst_stride0, src_stride0,
    				       		           (char *)dst1, (char *)src1, blocklength1, dst_stride1, src_stride1, count);
  }
}
*/
