/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

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

  _XMP_thread_barrier(&_XMP_thread_barrier_key, _XMP_num_threads);
}

void _XMP_thread_barrier(volatile _XMP_thread_barrier_t *barrier, int nthreads) {
  _XMP_RETURN_IF_SINGLE;
  if (barrier == NULL) return;
  
  _Bool sense = barrier->sense;
  int count = __sync_fetch_and_add(&barrier->count, 1);
  if (count == nthreads - 1) {
    barrier->count = 0;
    barrier->sense = !sense;
  } else {
    while (barrier->sense == sense) ;
  }
}

_XMP_thread_barrier_t _XMP_thread_barrier_key;
