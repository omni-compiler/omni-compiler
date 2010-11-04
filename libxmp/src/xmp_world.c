#include "xmp_internal.h"

int _XCALABLEMP_world_rank;
int _XCALABLEMP_world_size;
void *_XCALABLEMP_world_nodes;

void _XCALABLEMP_init_world(int *argc, char ***argv) {
  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(argc, argv);

    MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
    MPI_Comm_dup(MPI_COMM_WORLD, comm);
    _XCALABLEMP_push_comm(comm);

    _XCALABLEMP_world_nodes = _XCALABLEMP_get_execution_nodes();
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
