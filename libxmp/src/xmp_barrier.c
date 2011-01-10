/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

void _XMP_barrier_NODES_ENTIRE(_XMP_nodes_t *nodes) {
  _XMP_ASSERT(nodes != NULL);

  if (nodes->is_member) {
    MPI_Barrier(*(nodes->comm));
  }
}

void _XMP_barrier_EXEC(void) {
  MPI_Barrier(*((_XMP_get_execution_nodes())->comm));
}
