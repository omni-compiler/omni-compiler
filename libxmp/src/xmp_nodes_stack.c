#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include "mpi.h"
#include "xmp_internal.h"

typedef struct _XMP_nodes_dish_type {
  _XMP_nodes_t *nodes;
  struct _XMP_nodes_dish_type *prev;
} _XMP_nodes_dish_t;

static _XMP_nodes_dish_t *_XMP_nodes_stack_top = NULL;

void _XMP_push_nodes(_XMP_nodes_t *nodes)
{
  _XMP_nodes_dish_t *new_dish = _XMP_alloc(sizeof(_XMP_nodes_dish_t));
  new_dish->nodes             = nodes;
  new_dish->prev              = _XMP_nodes_stack_top;
  _XMP_nodes_stack_top        = new_dish;
}

void _XMP_pop_nodes(void)
{
  _XMP_nodes_dish_t *freed_dish = _XMP_nodes_stack_top;
  _XMP_nodes_stack_top          = freed_dish->prev;
  _XMP_free(freed_dish);
}

void _XMP_pop_n_free_nodes(void)
{
  _XMP_nodes_dish_t *freed_dish = _XMP_nodes_stack_top;
  _XMP_nodes_stack_top = freed_dish->prev;
  _XMP_finalize_nodes(freed_dish->nodes);
  _XMP_free(freed_dish);
}

void _XMP_pop_n_free_nodes_wo_finalize_comm(void)
{
  _XMP_nodes_dish_t *freed_dish = _XMP_nodes_stack_top;
  _XMP_nodes_stack_top          = freed_dish->prev;
  _XMP_free(freed_dish->nodes);
  _XMP_free(freed_dish);
}

_XMP_nodes_t *_XMP_get_execution_nodes(void)
{
  return _XMP_nodes_stack_top->nodes;
}

int _XMP_get_execution_nodes_rank(void)
{
  return _XMP_get_execution_nodes()->comm_rank;
}

void _XMP_push_comm(_XMP_comm_t *comm)
{
  _XMP_push_nodes(_XMP_create_nodes_by_comm(_XMP_N_INT_TRUE, comm));
}

void _XMP_finalize_comm(_XMP_comm_t *comm)
{
  MPI_Comm_free(comm);
  _XMP_free(comm);
}
