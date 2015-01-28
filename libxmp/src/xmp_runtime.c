#include "xmp_internal.h"
#include "mpi.h"
#ifdef _XMP_COARRAY_FJRDMA
#include "mpi-ext.h"
#endif
#include <stdio.h>

static __thread int _XMP_runtime_working = _XMP_N_INT_FALSE;
__thread int _XMPC_running = 1;
__thread int _XMPF_running = 0;

__thread int _XMP_num_threads;
__thread int _XMP_thread_num;

void _XMP_init(int argc, char** argv)
{
  if (!_XMP_runtime_working) {
#if defined(_XMP_COARRAY_GASNET) || defined(_XMP_COARRAY_FJRDMA)
    _XMP_coarray_initialize(argc, argv);
    _XMP_post_wait_initialize();
#endif

#ifdef _XMP_TCA
    _XMP_init_tca();
#endif
  }
  _XMP_init_world(NULL, NULL);
  _XMP_runtime_working = _XMP_N_INT_TRUE;
  _XMP_check_reflect_type();
}

void _XMP_init_thread(int argc, char **argv, int num_threads, int thread_num)
{
  _XMP_num_threads = num_threads;
  _XMP_thread_num = thread_num;
  _XMP_init(argc, argv);
}

void _XMP_finalize(int return_val)
{
  if (_XMP_runtime_working) {
#ifdef _XMP_COARRAY_GASNET
    _XMP_coarray_finalize(return_val);
#elif _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_finalize();
#endif
    _XMP_finalize_world();
    _XMP_runtime_working = _XMP_N_INT_FALSE;
  }
}

char *_XMP_desc_of(void *p)
{
  return (char *)p;
}

void xmpc_init_all(int argc, char** argv)
{
  _XMP_init(argc, argv);
}

void xmpc_init_thread_all(int argc, char **argv, int num_threads, int thread_num)
{
  _XMP_init_thread(argc, argv, num_threads, thread_num);
}

void xmpc_finalize_all(int return_val)
{
  _XMP_finalize(return_val);
}
