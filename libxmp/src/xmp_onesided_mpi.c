//#define DEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "xmp_internal.h"
#ifdef _XMP_XACC
#include "xacc_internal.h"
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

int _XMP_flag_multi_win = 0;

void _XMP_mpi_onesided_initialize(const size_t heap_size)
{
  XACC_DEBUG("_XMP_mpi_onesided_initialize start");
  {
    char *p = getenv("XMP_ONESIDED_MPI_MULTI_WIN");
    if(p != NULL){
      _XMP_flag_multi_win = atoi(p);
    }
  }

  if(_XMP_flag_multi_win) return;

  _xmp_mpi_onesided_heap_size = heap_size;

  XACC_DEBUG("alloc memory size=%zd\n", heap_size);
  XACC_DEBUG("alloced _xmp_mpi_onesided_buf(%p)\n", _xmp_mpi_onesided_buf);
  int a = 0;
  MPI_Initialized(&a);
  _XMP_mpi_onesided_alloc_win(&_xmp_mpi_onesided_win, (void**)&_xmp_mpi_onesided_buf, heap_size, MPI_COMM_WORLD, false);

  _XMP_mpi_build_shift_queue(false);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, _xmp_mpi_onesided_win);
  //MPI_Win_fence(MPI_MODE_NOPRECEDE, _xmp_mpi_onesided_win);

  MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &_xmp_mpi_distarray_win);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, _xmp_mpi_distarray_win);
  
#if defined(_XMP_XACC) && defined(_XMP_XACC_CUDA)
  _XMP_mpi_onesided_alloc_win(&_xmp_mpi_onesided_win_acc, (void**)&_xmp_mpi_onesided_buf_acc, heap_size, MPI_COMM_WORLD, true);

  _XMP_mpi_build_shift_queue(true);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, _xmp_mpi_onesided_win_acc);
  //MPI_Win_fence(MPI_MODE_NOPRECEDE, _xmp_mpi_onesided_win_acc);

  MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &_xmp_mpi_distarray_win_acc);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, _xmp_mpi_distarray_win_acc);
#endif
}

void _XMP_mpi_onesided_finalize(){
  XACC_DEBUG("_XMP_mpi_onesided_finalize()");

  if(_XMP_flag_multi_win) return;

  MPI_Win_unlock_all(_xmp_mpi_onesided_win);
  _XMP_mpi_destroy_shift_queue(false);
  MPI_Win_unlock_all(_xmp_mpi_distarray_win);

  MPI_Barrier(MPI_COMM_WORLD);
  _XMP_mpi_onesided_dealloc_win(&_xmp_mpi_onesided_win, (void **)&_xmp_mpi_onesided_buf, false);
  XACC_DEBUG("free _xmp_mpi_onesided_buf(%p)\n", _xmp_mpi_onesided_buf);
  MPI_Win_free(&_xmp_mpi_distarray_win);
  
#if defined(_XMP_XACC) && defined(_XMP_XACC_CUDA)
  MPI_Win_unlock_all(_xmp_mpi_onesided_win_acc);
  _XMP_mpi_destroy_shift_queue(true);
  MPI_Win_unlock_all(_xmp_mpi_distarray_win_acc);

  MPI_Barrier(MPI_COMM_WORLD);
  _XMP_mpi_onesided_dealloc_win(&_xmp_mpi_onesided_win_acc, (void **)&_xmp_mpi_onesided_buf_acc, true);
  MPI_Win_free(&_xmp_mpi_distarray_win_acc);
#endif
}

void _XMP_mpi_onesided_create_win(MPI_Win *win, void *addr, size_t size, MPI_Comm comm)
{
  MPI_Win_create(addr,     //window address
		 size,     //window size
		 1,        //gap size
		 MPI_INFO_NULL,
		 comm,
		 win);
}

void _XMP_mpi_onesided_alloc_win(MPI_Win *win, void **addr, size_t size, MPI_Comm comm, bool is_acc)
{
  MPI_Barrier(comm);
  if(is_acc){
#ifdef _XMP_XACC
#if defined(_XMP_XACC_CUDA)
    _XACC_memory_alloc(addr, size);
    _XMP_mpi_onesided_create_win(win, *addr, size, comm);
#elif defined(_XMP_XACC_OPENCL)
    _XMP_fatal("_XMP_mpi_onesided_alloc_win: XACC/OpenCL does not support onesided communication");
#endif
#else
    _XMP_fatal("_XMP_mpi_onesided_alloc_win: XACC is not enabled");
#endif
  }else{
    MPI_Win_allocate(size,   //window size
		     1,      //gap size
		     MPI_INFO_NULL,
		     comm,
		     addr,   //window address
		     win);
  }
  MPI_Barrier(comm);
}

void _XMP_mpi_onesided_destroy_win(MPI_Win *win)
{
  MPI_Win_free(win);
}

void _XMP_mpi_onesided_dealloc_win(MPI_Win *win, void **addr, bool is_acc)
{
  void *win_base = NULL;
  if(is_acc){
    int flag;
    MPI_Win_get_attr(*win, MPI_WIN_BASE, &win_base, &flag);
    if(! flag){
      _XMP_fatal("cannot get win_base");
    }
    if(*addr != win_base){
      _XMP_fatal("addr differ from win_base");
    }
  }

  _XMP_mpi_onesided_destroy_win(win);

  if(is_acc){
#ifdef _XMP_XACC
#if defined(_XMP_XACC_CUDA)
    _XACC_memory_free(addr);
#elif defined(_XMP_XACC_OPENCL)
    _XMP_fatal("_XMP_mpi_onesided_dealloc_win: XACC/OpenCL does not support onesided communication");
#endif
#else
    _XMP_fatal("_XMP_mpi_onesided_dealloc_win: XACC is not enabled");
#endif
  }

  *addr = NULL;
}
