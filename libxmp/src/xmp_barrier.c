#include "xmp_internal.h"

void _XCALABLEMP_barrier_NODES_ENTIRE(_XCALABLEMP_nodes_t *nodes) {
  if (nodes == NULL) return;
  else MPI_Barrier(*(nodes->comm));
}

void _XCALABLEMP_barrier_EXEC(void) {
  MPI_Barrier(*((_XCALABLEMP_get_execution_nodes())->comm));
}
