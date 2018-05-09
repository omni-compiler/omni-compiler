#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"
#ifdef _XMP_XACC
#include "xacc_internal.h"
#endif
#include "mpi.h"

#ifdef _XMP_FJRDMA
#include "mpi-ext.h"
#endif

static int _XMP_runtime_working = _XMP_N_INT_FALSE;
int _XMPC_running = 1;
int _XMPF_running = 0;
extern void xmpc_traverse_init();
extern void xmpc_traverse_finalize();

void (*_xmp_pack_array)(void *buffer, void *src, int array_type, size_t array_type_size,
			int array_dim, int *l, int *u, int *s, unsigned long long *d) = _XMPC_pack_array;
void (*_xmp_unpack_array)(void *dst, void *buffer, int array_type, size_t array_type_size,
			  int array_dim, int *l, int *u, int *s, unsigned long long *d) = _XMPC_unpack_array;

#ifdef _XMPT
int xmpt_initialize();
int xmpt_enabled = 0;
#endif

int xmp_get_ruuning()
{
  return _XMP_runtime_working;
}

void _XMP_init(int argc, char** argv, MPI_Comm comm)
{
  if (!_XMP_runtime_working) {
    int flag = 0;
    MPI_Initialized(&flag);

    if(!flag)
      MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_XMP_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_XMP_world_size);

    //    int result = 0;
    //    MPI_Comm_compare(MPI_COMM_WORLD, comm, &result);
    //    if(result != MPI_IDENT)
    //      _XMP_fatal("Now implementation does not support subcommunicator");
    
#ifdef _XMP_XACC
    _XACC_init();
#endif

#ifdef _XMP_TCA
    _XMP_init_tca();
#endif

#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA) || defined(_XMP_TCA) || defined(_XMP_MPI3_ONESIDED)
    _XMP_initialize_onesided_functions();
#endif
#ifdef _XMP_MPI3
    _XMP_initialize_async_comm_tab();
#endif
    xmp_reduce_initialize();
#ifdef _XMPT
    xmpt_enabled = xmpt_initialize();
#endif
  }


  _XMP_init_world(NULL, NULL);
  _XMP_check_reflect_type();
  
  if (!_XMP_runtime_working) {
    xmpc_traverse_init();
  }
  _XMP_runtime_working = _XMP_N_INT_TRUE;
}

void _XMP_finalize(bool isFinalize)
{
  if (_XMP_runtime_working) {
    xmpc_traverse_finalize();
    
#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA) || defined(_XMP_TCA) || defined(_XMP_MPI3_ONESIDED)
    _XMP_finalize_onesided_functions();
#endif
    _XMP_finalize_world(isFinalize);
    _XMP_runtime_working = _XMP_N_INT_FALSE;
  }
}

char *_XMP_desc_of(void *p)
{
  return (char *)p;
}

void xmp_init_all(int argc, char** argv)
{
  _XMP_init(argc, argv, MPI_COMM_WORLD);
}

void xmp_finalize_all()
{
  _XMP_finalize(true);
}

#include "config.h"

size_t _XMP_get_datatype_size(int datatype)
{
  size_t size;

  // size of each type is obtained from config.h.
  // Note: need to fix when building a cross compiler.
  switch (datatype){

  case _XMP_N_TYPE_BOOL:
    size = _XMPF_running ? SIZEOF_UNSIGNED_INT : SIZEOF__BOOL;
    break;

  case _XMP_N_TYPE_CHAR:
  case _XMP_N_TYPE_UNSIGNED_CHAR:
    size = SIZEOF_UNSIGNED_CHAR; break;

  case _XMP_N_TYPE_SHORT:
  case _XMP_N_TYPE_UNSIGNED_SHORT:
    size = SIZEOF_UNSIGNED_SHORT; break;

  case _XMP_N_TYPE_INT:
  case _XMP_N_TYPE_UNSIGNED_INT:
    size = SIZEOF_UNSIGNED_INT; break;

  case _XMP_N_TYPE_LONG:
  case _XMP_N_TYPE_UNSIGNED_LONG:
    size = SIZEOF_UNSIGNED_LONG; break;

  case _XMP_N_TYPE_LONGLONG:
  case _XMP_N_TYPE_UNSIGNED_LONGLONG:
    size = SIZEOF_UNSIGNED_LONG_LONG; break;

  case _XMP_N_TYPE_FLOAT:
#ifdef __STD_IEC_559_COMPLEX__
  case _XMP_N_TYPE_FLOAT_IMAGINARY:
#endif
    size = SIZEOF_FLOAT; break;

  case _XMP_N_TYPE_DOUBLE:
#ifdef __STD_IEC_559_COMPLEX__
  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
#endif
    size = SIZEOF_DOUBLE; break;

  case _XMP_N_TYPE_LONG_DOUBLE:
#ifdef __STD_IEC_559_COMPLEX__
  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
#endif
    size = SIZEOF_LONG_DOUBLE; break;

  case _XMP_N_TYPE_FLOAT_COMPLEX:
    size = SIZEOF_FLOAT * 2; break;

  case _XMP_N_TYPE_DOUBLE_COMPLEX:
    size = SIZEOF_DOUBLE * 2; break;

  case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
    size = SIZEOF_LONG_DOUBLE * 2; break;

  case _XMP_N_TYPE_NONBASIC: // should be fixed for structures.
  default:
    size = 0; break;
  }

  return size;
}
