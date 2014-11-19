//#include<openacc.h>
/* typedef  int acc_device_t; */
/* //_XACC_device_t *current_device; */

/* typedef struct _XACC_device_type { */
/*   acc_device_t acc_device; */
/*   int lb; */
/*   int ub; */
/*   int step; */
/*   int size; */
/* } _XACC_device_t; */

/* void _XACC_init_device(void* desc, acc_device_t device, int lower, int upper, int step); */
/* int _XACC_get_num_current_devices(); */
/* acc_device_t _XACC_get_current_device(); */
/* void _XACC_get_current_device_info(int* lower, int* upper, int* step); */
/* void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step); */

/* void _XMP_init_device(void* desc, acc_device_t device, int lower, int upper, int step); */
/* void _XMP_get_device_info(void *desc, int* lower, int* upper, int* step); */

#include <stdlib.h>
#include "include/cuda_runtime.h"
#include "xacc_data_struct.h"

void _XACC_init_device(_XACC_device_t** desc, acc_device_t device, int lower, int upper, int step);
int _XACC_get_num_current_devices();
acc_device_t _XACC_get_current_device();
void _XACC_get_current_device_info(int* lower, int* upper, int* step);
void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step);

//void _XMP_init_device(void* desc, acc_device_t device, int lower, int upper, int step);
//void _XMP_get_device_info(void *desc, int* lower, int* upper, int* step);

void _XACC_init_layouted_array(_XACC_arrays_t **array, _XMP_array_t *alignedArray, _XACC_device_t* device);
void _XACC_split_layouted_array_BLOCK(_XACC_arrays_t* array, int dim);
void _XACC_split_layouted_array_DUPLICATION(_XACC_arrays_t* array, int dim);
void _XACC_calc_size(_XACC_arrays_t* array);

void _XACC_get_size(_XACC_arrays_t* array, unsigned long long* offset,
               unsigned long long* size, int deviceNum);
void _XACC_get_copy_size(_XACC_arrays_t* array, unsigned long long* offset,
               unsigned long long* size, int deviceNum);
void _XACC_sched_loop_layout_BLOCK(int init,
                                   int cond,
                                   int step,
                                   int* sched_init,
                                   int* sched_cond,
                                   int* sched_step,
                                   _XACC_arrays_t* array_desc,
                                   int dim,
                                   int deviceNum);
void _XACC_set_shadow_NORMAL(_XACC_arrays_t* array_desc, int dim , int lo, int hi);


//xacc_util.c
/* void _XACC_gpu_alloc(void **addr, size_t size); */
/* void _XACC_gpu_free(void *addr); */
/* void* _XACC_gpu_host_alloc(size_t size); */
/* void _XACC_gpu_host_free(void *addr); */

//xacc_pack.cu
void _XACC_gpu_pack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
void _XACC_gpu_unpack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, cudaStream_t stream);
void _XACC_gpu_pack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
				  char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				  int count, size_t typesize, cudaStream_t st);
void _XACC_gpu_unpack_vector2_async(char * __restrict__ dst0, char * __restrict__ src0, int blocklength0, long stride0,
				    char * __restrict__ dst1, char * __restrict__ src1, int blocklength1, long stride1,
				    int count, size_t typesize, cudaStream_t st);

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)						\
  do {                                                                  \
    cudaError_t err = call;						\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString(err) );		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

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

#include <omp.h>

//#define _TLOG

#ifdef _TLOG
#include "tlog.h"
#ifdef _OPENMP
#define TLOG_LOG(log) do{if(omp_get_thread_num()==0)tlog_log((log));}while(0)
#else
#define TLOG_LOG(log) do{tlog_log((log));}while(0)
#endif
#else
#define TLOG_LOG(log) do{}while(0)
#endif
