/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */
#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include "mpi.h"
#include "xmp_internal.h"

void _XMP_barrier_NODES_ENTIRE(_XMP_nodes_t *nodes) {
  _XMP_RETURN_IF_SINGLE;

  if (nodes->is_member) {
    MPI_Barrier(*((MPI_Comm *)nodes->comm));
  }
}

void _XMP_barrier_EXEC(void) {
  _XMP_RETURN_IF_SINGLE;

  MPI_Barrier(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm));
}
