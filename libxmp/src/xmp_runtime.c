#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"
#include "mpi.h"

#ifdef _XMP_FJRDMA
#include "mpi-ext.h"
#define XMP_FJRDMA_MAX_NODES 16384
static int compare_char(const void *x, const void *y);
static int within_limit_of_physical_nodes();
static int _num_of_physical_nodes = 1;
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

#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA)
    _XMP_onesided_initialize(argc, argv);
#endif
#ifdef _XMP_TCA
    _XMP_init_tca();
#endif
  }
  _XMP_init_world(NULL, NULL);
  _XMP_runtime_working = _XMP_N_INT_TRUE;
  _XMP_check_reflect_type();
}

void _XMP_finalize(int return_val)
{
  if (_XMP_runtime_working) {
#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA)
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

#ifdef _XMP_FJRDMA
static int compare_char(const void *x, const void *y)
{
  return strcmp((char *)x, (char *)y);
}

static int within_limit_of_physical_nodes()
{
  if(_XMP_world_size < XMP_FJRDMA_MAX_NODES)
    return _XMP_N_INT_TRUE;

  int  namelen, max_namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int tag = 0;
  MPI_Status s;

  MPI_Get_processor_name(processor_name, &namelen);
  MPI_Allreduce(&namelen, &max_namelen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  char all_processor_name[_XMP_world_size][max_namelen];

  if(_XMP_world_rank == 0){
    for(int i=1;i<_XMP_world_size;i++){
      MPI_Recv(all_processor_name[i], sizeof(char)*max_namelen, MPI_CHAR, i, tag, MPI_COMM_WORLD, &s);
    }
    memcpy(all_processor_name[0], processor_name, sizeof(char)*max_namelen);
  }
  else{
    MPI_Send(processor_name, sizeof(char)*max_namelen, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
  }

  if(_XMP_world_rank == 0){
    qsort(all_processor_name, _XMP_world_size, sizeof(char)*max_namelen, compare_char);
    for(int i=0;i<_XMP_world_size-1;i++)
      if(strncmp(all_processor_name[i], all_processor_name[i+1], max_namelen))
        _num_of_physical_nodes++;
  }

  MPI_Bcast(&_num_of_physical_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(_num_of_physical_nodes > XMP_FJRDMA_MAX_NODES)
    return _XMP_N_INT_FALSE;
  else
    return _XMP_N_INT_TRUE;
}
#endif
