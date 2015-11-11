//#define DEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "xmp_internal.h"
#ifdef _XMP_XACC
#include <cuda_runtime.h>
#endif

size_t _xmp_mpi_onesided_heap_size; //host and device
char *_xmp_mpi_onesided_buf;
MPI_Win _xmp_mpi_onesided_win;
MPI_Win _xmp_mpi_distarray_win;
//size_t _xmp_mpi_onesided_coarray_shift = 0;

char *_xmp_mpi_onesided_buf_acc;
MPI_Win _xmp_mpi_onesided_win_acc;
MPI_Win _xmp_mpi_distarray_win_acc;
//size_t _xmp_mpi_onesided_coarray_shift_acc = 0;

////int _xmp_mpi_onesided_enable_host_device_comm = 0; //if 0 then use MPI_Win_create else MPI_Win_dynamic

#define CUDA_SAFE_CALL(call)						\
  do {                                                                  \
    cudaError_t err = call;						\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString(err) );		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)


void _XMP_mpi_onesided_initialize(int argc, char **argv, const size_t heap_size)
{
  XACC_DEBUG("_XMP_mpi_onesided_initialize start");
  _xmp_mpi_onesided_heap_size = heap_size;

  XACC_DEBUG("alloc memory size=%zd\n", heap_size);
  XACC_DEBUG("alloced _xmp_mpi_onesided_buf(%p)\n", _xmp_mpi_onesided_buf);
  MPI_Win_allocate(heap_size, //window size
		   sizeof(char), //gap size
		   MPI_INFO_NULL,
		   MPI_COMM_WORLD,
		   &_xmp_mpi_onesided_buf, //window address
		   &_xmp_mpi_onesided_win);

  _XMP_mpi_build_shift_queue(false);
  MPI_Win_lock_all(0, _xmp_mpi_onesided_win);
  //MPI_Win_fence(MPI_MODE_NOPRECEDE, _xmp_mpi_onesided_win);

  MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &_xmp_mpi_distarray_win);
  MPI_Win_lock_all(0, _xmp_mpi_distarray_win);
  
#ifdef _XMP_XACC
  CUDA_SAFE_CALL(cudaMalloc((void**)&_xmp_mpi_onesided_buf_acc, heap_size));
  XACC_DEBUG("alloced gpu addr =%p\n", _xmp_mpi_onesided_buf_acc);
  MPI_Win_create((void*)_xmp_mpi_onesided_buf_acc, //window address
		 heap_size, //window size
		 sizeof(char), //gap size
		 MPI_INFO_NULL,
		 MPI_COMM_WORLD,
		 &_xmp_mpi_onesided_win_acc);

  _XMP_mpi_build_shift_queue(true);
  MPI_Win_lock_all(0, _xmp_mpi_onesided_win_acc);
  //MPI_Win_fence(MPI_MODE_NOPRECEDE, _xmp_mpi_onesided_win_acc);

  MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &_xmp_mpi_distarray_win_acc);
  MPI_Win_lock_all(0, _xmp_mpi_distarray_win_acc);
#endif
}

void _XMP_mpi_onesided_finalize(){
  XACC_DEBUG("_XMP_mpi_onesided_finalize()");

  MPI_Win_unlock_all(_xmp_mpi_onesided_win);
  //MPI_Win_fence(MPI_MODE_NOSUCCEED, _xmp_mpi_onesided_win);
  _XMP_mpi_destroy_shift_queue(false);
  MPI_Win_free(&_xmp_mpi_onesided_win);
  XACC_DEBUG("free _xmp_mpi_onesided_buf(%p)\n", _xmp_mpi_onesided_buf);

  MPI_Win_unlock_all(_xmp_mpi_distarray_win);
  MPI_Win_free(&_xmp_mpi_distarray_win);
  
#ifdef _XMP_XACC
  MPI_Win_unlock_all(_xmp_mpi_onesided_win_acc);
  //MPI_Win_fence(MPI_MODE_NOSUCCEED, _xmp_mpi_onesided_win_acc);
  _XMP_mpi_destroy_shift_queue(true);
  MPI_Win_free(&_xmp_mpi_onesided_win_acc);
  CUDA_SAFE_CALL(cudaFree(_xmp_mpi_onesided_buf_acc));

  MPI_Win_unlock_all(_xmp_mpi_distarray_win_acc);
  MPI_Win_free(&_xmp_mpi_distarray_win_acc);
#endif
}
