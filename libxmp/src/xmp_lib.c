/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "mpi.h"
#include "xmp_internal.h"

// FIXME utility functions
void xmp_get_comm(void **comm) {
  *comm = _XMP_get_execution_nodes()->comm;
}

int xmp_get_size(void) {
  return _XMP_get_execution_nodes()->comm_size;
}

int xmp_get_rank(void) {
  return _XMP_get_execution_nodes()->comm_rank;
}

void xmp_barrier(void) {
  _XMP_barrier_EXEC();
}

int xmp_get_world_size(void) {
  return _XMP_world_size;
}

int xmp_get_world_rank(void) {
  return _XMP_world_rank;
}

double xmp_wtime(void) {
  return MPI_Wtime();
}
