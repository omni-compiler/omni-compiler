#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "xmp_internal.h"

extern void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
extern void _XMP_reduce_gpu_CLAUSE(void *data_addr, int count, int datatype, int op);

static char comm_mode = -1;

static void set_comm_mode()
{
  if(comm_mode < 0){
    char *mode_str = getenv("XACC_COMM_MODE");
    if(mode_str !=  NULL){
      comm_mode = atoi(mode_str);
    }else{
      comm_mode = 0;
    }
  }
}

void _XMP_reduce_acc_NODES_ENTIRE(_XMP_nodes_t *nodes, void *data_addr, int count, int datatype, int op)
{
  set_comm_mode();

  if(comm_mode >= 1){
    _XMP_reduce_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
  }else{
    _XMP_reduce_gpu_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
  }
}

void _XMP_reduce_acc_FLMM_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op, int num_locs, ...)
{
  _XMP_fatal("_XMP_reduce_acc_FLMM_NODES_ENTIRE is unimplemented");
}

void _XMP_reduce_acc_CLAUSE(void *data_addr, int count, int datatype, int op)
{
  set_comm_mode();

  if(comm_mode >= 1){
    _XMP_reduce_CLAUSE(data_addr, count, datatype, op);
  }else{
    _XMP_reduce_gpu_CLAUSE(data_addr, count, datatype, op);
  }
}

void _XMP_reduce_acc_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...)
{
  _XMP_fatal("_XMP_reduce_acc_FLMM_CLAUSE is unimplemented");
}
