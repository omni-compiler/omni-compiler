/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_internal.h"

int _XCALABLEMP_world_size;
int _XCALABLEMP_world_rank;
void *_XCALABLEMP_world_nodes;

void _XCALABLEMP_init_world(int *argc, char ***argv) {
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(argc, argv);

    MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
    MPI_Comm_dup(MPI_COMM_WORLD, comm);

    _XCALABLEMP_nodes_t *n = _XCALABLEMP_create_nodes_by_comm(comm);

    _XCALABLEMP_world_size = n->comm_size;
    _XCALABLEMP_world_rank = n->comm_rank;
    _XCALABLEMP_world_nodes = n;

    _XCALABLEMP_push_nodes(n);
  }
}

void _XCALABLEMP_init_world_NULL(void) {
  _XCALABLEMP_init_world(NULL, NULL);
}

void _XCALABLEMP_barrier_WORLD(void) {
  MPI_Barrier(MPI_COMM_WORLD);
}

int _XCALABLEMP_finalize_world(int ret) {
  MPI_Finalize();
  return ret;
}
