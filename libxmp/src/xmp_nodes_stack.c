#include "xmp_internal.h"

typedef struct _XCALABLEMP_nodes_dish_type {
  _XCALABLEMP_nodes_t *nodes;
  struct _XCALABLEMP_nodes_dish_type *prev;
} _XCALABLEMP_nodes_dish_t;

static _XCALABLEMP_nodes_dish_t *_XCALABLEMP_nodes_stack_top = NULL;

void _XCALABLEMP_push_nodes(_XCALABLEMP_nodes_t *nodes) {
  if (nodes == NULL)
    _XCALABLEMP_fatal("null nodes descriptor detected");

  _XCALABLEMP_nodes_dish_t *new_dish = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_dish_t));
  new_dish->nodes = nodes;
  new_dish->prev = _XCALABLEMP_nodes_stack_top;
  _XCALABLEMP_nodes_stack_top = new_dish;
}

void _XCALABLEMP_pop_nodes(void) {
  if (_XCALABLEMP_nodes_stack_top == NULL)
    _XCALABLEMP_fatal("global communicator has removed");
  else {
    _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
    _XCALABLEMP_nodes_stack_top = freed_dish->prev;
    _XCALABLEMP_free(freed_dish);
  }
}

void _XCALABLEMP_pop_n_free_nodes(void) {
  if (_XCALABLEMP_nodes_stack_top == NULL)
    _XCALABLEMP_fatal("global communicator has removed");
  else {
    _XCALABLEMP_nodes_dish_t *freed_dish = _XCALABLEMP_nodes_stack_top;
    _XCALABLEMP_nodes_stack_top = freed_dish->prev;
    _XCALABLEMP_free(freed_dish->nodes);
    _XCALABLEMP_free(freed_dish);
  }
}

_XCALABLEMP_nodes_t *_XCALABLEMP_get_execution_nodes(void) {
  if (_XCALABLEMP_nodes_stack_top == NULL) _XCALABLEMP_fatal("cannot get the execution nodes");
  else {
    _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_nodes_stack_top->nodes;
    if (exec_nodes == NULL) _XCALABLEMP_fatal("cannot get the execution nodes");
    else return exec_nodes;
  }
}
