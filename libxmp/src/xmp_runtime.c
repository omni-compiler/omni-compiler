#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"
#include "mpi.h"

#ifdef _XMP_FJRDMA
#include "mpi-ext.h"
#endif

static int _XMP_runtime_working = _XMP_N_INT_FALSE;
int _XMPC_running = 1;
int _XMPF_running = 0;

void _XMP_init(int argc, char** argv)
{
  if (!_XMP_runtime_working) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_XMP_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_XMP_world_size);

#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA) || defined(_XMP_TCA) || defined(_XMP_MPI3_ONESIDED)
    _XMP_onesided_initialize(argc, argv);
#endif
  }
  _XMP_init_world(NULL, NULL);
  _XMP_runtime_working = _XMP_N_INT_TRUE;
  _XMP_check_reflect_type();
}

void _XMP_finalize(int return_val)
{
  if (_XMP_runtime_working) {
#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA) || defined(_XMP_TCA) || defined(_XMP_MPI3_ONESIDED)
    _XMP_onesided_finalize(return_val);
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

void xmpc_finalize_all(int return_val)
{
  _XMP_finalize(return_val);
}

