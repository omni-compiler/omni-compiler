#include "xmp_internal.h"

typedef struct _XCALABLEMP_nodes_dish_type {
  _XCALABLEMP_nodes_t *nodes;
  struct _XCALABLEMP_nodes_dish_type *prev;
} _XCALABLEMP_nodes_dish_t;

static _XCALABLEMP_nodes_dish_t *_XCALABLEMP_nodes_stack_top = NULL;

void _XCALABLEMP_push_nodes(_XCALABLEMP_nodes_t *nodes) {
  _XCALABLEMP_nodes_dish_t *new_dish = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_dish_t));
  new_dish->nodes = nodes;
  new_dish->prev = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = new_dish;
}

void _XCALABLEMP_pop_nodes(void) {
  _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = freed_dish->prev;
  _XCALABLEMP_free(freed_dish);
}

void _XCALABLEMP_pop_n_free_nodes(void) {
  _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = freed_dish->prev;
  _XCALABLEMP_free(freed_dish->nodes);
  _XCALABLEMP_free(freed_dish);
}

_XCALABLEMP_nodes_t *_XCALABLEMP_get_execution_nodes(void) {
  return _XCALABLEMP_nodes_stack_top->nodes;
}

int _XCALABLEMP_get_execution_nodes_rank(void) {
  // is_member is always true
  return (_XCALABLEMP_get_execution_nodes())->comm_rank;
}

void _XCALABLEMP_push_comm(MPI_Comm *comm) {
  int size, rank;
  MPI_Comm_size(*comm, &size);
  MPI_Comm_rank(*comm, &rank);

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t));

  n->is_member = true;
  n->dim = 1;

  n->comm = comm;
  n->comm_size = size;
  n->comm_rank = rank;

  n->info[0].size = size;
  n->info[0].rank = rank;

  _XCALABLEMP_push_nodes(n);
}
