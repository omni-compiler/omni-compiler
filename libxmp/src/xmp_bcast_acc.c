#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

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

void _XMP_bcast_acc_NODES_ENTIRE_OMITTED(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size) {
  set_comm_mode();

  if(comm_mode >= 1){
    _XMP_bcast_NODES_ENTIRE_OMITTED(bcast_nodes, addr, count, datatype_size);
  }else{
    _XMP_fatal("uninplemented");
  }
}
void _XMP_bcast_acc_NODES_ENTIRE_NODES(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
				       _XMP_nodes_t *from_nodes, ...) {
  set_comm_mode();

  if(comm_mode >= 1){
    va_list args;
    va_start(args,from_nodes);
    _XMP_bcast_NODES_ENTIRE_NODES_V(bcast_nodes, addr, count, datatype_size, from_nodes, args);
    va_end(args);
  }else{
    _XMP_fatal("uninplemented");
  }
}

