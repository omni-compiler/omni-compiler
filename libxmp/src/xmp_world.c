#include "xmp_internal.h"

int _XCALABLEMP_world_rank;
int _XCALABLEMP_world_size;
void *_XCALABLEMP_world_nodes;

void _XCALABLEMP_init_world(int *argc, char ***argv) {
  int flag = 0;

  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_XCALABLEMP_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_XCALABLEMP_world_size);

    // init global communicator
    _XCALABLEMP_nodes_t *world_nodes = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t));
    world_nodes->comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
    MPI_Comm_dup(MPI_COMM_WORLD, world_nodes->comm);
    world_nodes->comm_size = _XCALABLEMP_world_size;
    world_nodes->comm_rank = _XCALABLEMP_world_rank;
    world_nodes->dim = 0;

    // push global communicator
    _XCALABLEMP_push_nodes(world_nodes);

    _XCALABLEMP_world_nodes = world_nodes;
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
