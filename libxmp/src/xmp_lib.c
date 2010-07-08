#include "xmp_internal.h"

// FIXME utility functions
void xmp_get_comm(void **comm) {
  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes();
  *comm = exec_nodes;
}

int xmp_get_size(void) {
  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes();
  return exec_nodes->comm_size;
}

int xmp_get_rank(void) {
  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes();
  return exec_nodes->comm_rank;
}

void xmp_barrier(void) {
  _XCALABLEMP_barrier_EXEC();
}

int xmp_get_world_size(void) {
  return _XCALABLEMP_world_size;
}

int xmp_get_world_rank(void) {
  return _XCALABLEMP_world_rank;
}

double xmp_wtime(void) {
  return MPI_Wtime();
}
