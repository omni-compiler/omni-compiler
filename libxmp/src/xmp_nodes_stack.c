#include "xmp_internal.h"

typedef struct _XCALABLEMP_nodes_dish_type {
  _XCALABLEMP_nodes_t *nodes;
  struct _XCALABLEMP_nodes_dish_type *prev;
} _XCALABLEMP_nodes_dish_t;

static _XCALABLEMP_nodes_dish_t *_XCALABLEMP_nodes_stack_top = NULL;

void _XCALABLEMP_push_nodes(_XCALABLEMP_nodes_t *nodes) {
  assert(nodes != NULL);

  _XCALABLEMP_nodes_dish_t *new_dish = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_dish_t));
  new_dish->nodes = nodes;
  new_dish->prev = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = new_dish;
}

void _XCALABLEMP_pop_nodes(void) {
  assert(_XCALABLEMP_nodes_stack_top != NULL);

  _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = freed_dish->prev;
  _XCALABLEMP_free(freed_dish);
}

void _XCALABLEMP_pop_n_free_nodes(void) {
  assert(_XCALABLEMP_nodes_stack_top != NULL);

  _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = freed_dish->prev;
  _XCALABLEMP_finalize_nodes(freed_dish->nodes);
  _XCALABLEMP_free(freed_dish);
}

void _XCALABLEMP_pop_n_free_nodes_wo_finalize_comm(void) {
  assert(_XCALABLEMP_nodes_stack_top != NULL);

  _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = freed_dish->prev;
  _XCALABLEMP_free(freed_dish->nodes);
  _XCALABLEMP_free(freed_dish);
}

_XCALABLEMP_nodes_t *_XCALABLEMP_get_execution_nodes(void) {
  assert(_XCALABLEMP_nodes_stack_top != NULL);

  return _XCALABLEMP_nodes_stack_top->nodes;
}

int _XCALABLEMP_get_execution_nodes_rank(void) {
  return _XCALABLEMP_get_execution_nodes()->comm_rank;
}

void _XCALABLEMP_push_comm(MPI_Comm *comm) {
  assert(comm != NULL);

  _XCALABLEMP_push_nodes(_XCALABLEMP_create_nodes_by_comm(comm));
}

void _XCALABLEMP_finalize_comm(MPI_Comm *comm) {
  assert(comm != NULL);

  MPI_Comm_free(comm);
  _XCALABLEMP_free(comm);
}
