/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "mpi.h"
#include "xmp_internal.h"

void _XMP_barrier_NODES_ENTIRE(_XMP_nodes_t *nodes) {
  if (nodes->is_member) {
    MPI_Barrier(*((MPI_Comm *)nodes->comm));
  }
}

void _XMP_barrier_EXEC(void) {
  MPI_Barrier(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm));
}
