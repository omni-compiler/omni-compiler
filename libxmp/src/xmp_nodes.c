/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <limits.h>
#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

// XXX <nodes-ref> is { 1-ogigin in language | 0-origin in runtime }, needs converting

static unsigned long long _XMP_nodes_id_counter = 0;

static _XMP_nodes_t *_XMP_create_new_nodes(int is_member, int dim, int comm_size, _XMP_comm *comm) {
  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  if (_XMP_nodes_id_counter == ULLONG_MAX) {
    _XMP_fatal("cannot create a new nodes descriptor: too many nodes");
  } else {
    n->nodes_id = _XMP_nodes_id_counter;
    _XMP_nodes_id_counter++;
  }
  
  n->is_member = is_member;
  n->dim = dim;
  n->comm_size = comm_size;
  n->comm = comm;

  if (is_member) {
    int size, rank;
    MPI_Comm_size(*((MPI_Comm *)comm), &size);
    MPI_Comm_rank(*((MPI_Comm *)comm), &rank);

    n->comm_rank = rank;

    if (size != comm_size) {
      _XMP_fatal("cannot create a new nodes descriptor: communicator size is not correct");
    }
  } else {
    n->comm_rank = _XMP_N_INVALID_RANK;
  }

  return n;
}

// XXX args are 1-origin
static void _XMP_validate_nodes_ref(int *lower, int *upper, int *stride, int size) {
  // setup temporary variables
  int l, u, s = *(stride);
  if (s > 0) {
    l = *lower;
    u = *upper;
  }
  else if (s < 0) {
    l = *upper;
    u = *lower;
  }
  else {
    _XMP_fatal("the stride of <nodes-ref> is 0");
    l = 0; u = 0; // XXX dummy
  }

  // check boundary
  if (1 > l) {
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-lower> is less than 1");
  }

  if (l > u) {
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is less than <ref-lower>");
  }

  if (u > size) {
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is greater than the node size");
  }

  // validate values
  if (s > 0) {
    u = u - ((u - l) % s);
    *upper = u;
  }
  else {
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }

  // XXX convert 1-origin to 0-origin
  (*lower)--;
  (*upper)--;
}

static int _XMP_check_nodes_ref_inclusion(int lower, int upper, int stride, int size, int rank) {
  if (rank < lower) {
    return _XMP_N_INT_FALSE;
  }

  if (rank > upper) {
    return _XMP_N_INT_FALSE;
  }

  if (((rank - lower) % stride) == 0) {
    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

static _XMP_nodes_inherit_info_t *_XMP_calc_inherit_info(_XMP_nodes_t *n) {
  int dim = n->dim;
  _XMP_nodes_inherit_info_t *inherit_info = _XMP_alloc(sizeof(_XMP_nodes_inherit_info_t) * dim);

  for (int i = 0; i < dim; i++) {
    int size = n->info[i].size;

    inherit_info[i].is_enable = _XMP_N_INT_TRUE;
    inherit_info[i].lower = 0;
    inherit_info[i].upper = size - 1;
    inherit_info[i].stride = 1;

    inherit_info[i].size = size;
  }

  return inherit_info;
}

static _XMP_nodes_inherit_info_t *_XMP_calc_inherit_info_by_ref(_XMP_nodes_t *n,
                                                                int *shrink, int *lower, int *upper, int *stride) {
  int dim = n->dim;
  _XMP_nodes_inherit_info_t *inherit_info = _XMP_alloc(sizeof(_XMP_nodes_inherit_info_t) * dim);

  for (int i = 0; i < dim; i++) {
    if (shrink[i]) {
      int size = n->info[i].size;

      inherit_info[i].is_enable = _XMP_N_INT_FALSE;
      inherit_info[i].lower = 0;
      inherit_info[i].upper = size - 1;;
      inherit_info[i].stride = 1;
      inherit_info[i].size = size;
    } else {
      inherit_info[i].is_enable = _XMP_N_INT_TRUE;
      inherit_info[i].lower = lower[i];
      inherit_info[i].upper = upper[i];
      inherit_info[i].stride = stride[i];
      inherit_info[i].size = _XMP_M_COUNT_TRIPLETi(lower[i], upper[i], stride[i]);
    }
  }

  return inherit_info;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_GLOBAL(int dim) {
  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(MPI_COMM_WORLD, comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(_XMP_N_INT_TRUE, dim, _XMP_world_size, (_XMP_comm *)comm);

  // calc inherit info
  n->inherit_nodes = NULL;
  n->inherit_info = NULL;

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_EXEC(int dim) {
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int size = exec_nodes->comm_size;

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(*((MPI_Comm *)exec_nodes->comm), comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(_XMP_N_INT_TRUE, dim, size, (_XMP_comm *)comm);

  // calc inherit info
  _XMP_nodes_t *inherit_nodes = _XMP_get_execution_nodes();
  n->inherit_nodes = inherit_nodes;
  n->inherit_info = _XMP_calc_inherit_info(inherit_nodes);

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper, int ref_stride) {
  _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XMP_world_size);
  int is_member = _XMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, _XMP_world_size, _XMP_world_rank);

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), is_member, _XMP_world_rank, comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(is_member, dim, _XMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride),
                                          (_XMP_comm *)comm);

  // calc inherit info
  int shrink[1] = {_XMP_N_INT_FALSE};
  int l[1] = {ref_lower};
  int u[1] = {ref_upper};
  int s[1] = {ref_stride};
  n->inherit_nodes = _XMP_world_nodes;
  n->inherit_info = _XMP_calc_inherit_info_by_ref(_XMP_world_nodes, shrink, l, u, s);

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NAMED(int dim, _XMP_nodes_t *ref_nodes,
                                                        int *shrink, int *ref_lower, int *ref_upper, int *ref_stride) {
  int ref_dim = ref_nodes->dim;
  int is_ref_nodes_member = ref_nodes->is_member;

  int color = 1;
  int is_member = _XMP_N_INT_TRUE;
  if (is_ref_nodes_member) {
    int acc_nodes_size = 1;
    for (int i = 0; i < ref_dim; i++) {
      int size = ref_nodes->info[i].size;
      int rank = ref_nodes->info[i].rank;

      if (shrink[i]) {
        color += (acc_nodes_size * rank);
      } else {
        _XMP_validate_nodes_ref(&(ref_lower[i]), &(ref_upper[i]), &(ref_stride[i]), size);
        is_member = is_member && _XMP_check_nodes_ref_inclusion(ref_lower[i], ref_upper[i], ref_stride[i],
                                                                size, rank);
      }

      acc_nodes_size *= size;
    }

    if (!is_member) {
      color = 0;
    }
  } else {
    is_member = _XMP_N_INT_FALSE;
    color = 0;
  }

  int comm_size = 1;
  for (int i = 0; i < ref_dim; i++) {
    if (!shrink[i]) {
      comm_size *= _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);
    }
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  return _XMP_create_new_nodes(is_member, dim, comm_size, (_XMP_comm *)comm);
}

static void _XMP_calc_nodes_rank(_XMP_nodes_t *n, int linear_rank) {
  _XMP_ASSERT(n->is_member);

  int acc_size = 1;
  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    int dim_size = n->info[i].size;
    n->info[i].rank = (linear_rank / acc_size) % dim_size;
    acc_size *= dim_size;
  }
}

static void _XMP_disable_nodes_rank(_XMP_nodes_t *n) {
  _XMP_ASSERT(!n->is_member);

  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    n->info[i].rank = _XMP_N_INVALID_RANK;
  }
}

static void _XMP_check_nodes_size_STATIC(_XMP_nodes_t *n) {
  int acc_size = 1;
  int dim = n->dim;
  int linear_size = n->comm_size;
  for (int i = 0; i < dim; i++) {
    acc_size *= n->info[i].size;
  }

  if (acc_size != linear_size) {
    _XMP_fatal("incorrect communicator size");
  }
}

static void _XMP_check_nodes_size_DYNAMIC(_XMP_nodes_t *n) {
  int acc_size = 1;
  int dim = n->dim;
  int linear_size = n->comm_size;
  int linear_rank = n->comm_rank;
  for (int i = 0; i < dim - 1; i++) {
    acc_size *= n->info[i].size;
  }

  if (acc_size > linear_size) {
    _XMP_fatal("indicated communicator size is bigger than the actual communicator size");
  }

  if ((linear_size % acc_size) != 0) {
    _XMP_fatal("cannot determine communicator size dynamically");
  }

  int end_size = linear_size / acc_size;
  n->info[dim-1].size = end_size;

  if (n->is_member) {
    n->info[dim-1].rank = (linear_rank / acc_size) % end_size;
  }
}

static void _XMP_set_task_desc(_XMP_task_desc_t *desc, int execute, _XMP_nodes_t *n,
                               int dim, int *lower, int *upper, int *stride) {
  desc->inherit_nodes_id = _XMP_get_execution_nodes()->nodes_id;
  desc->execute = execute;
  desc->nodes = n;
  desc->dim = dim;
  for (int i = 0; i < dim; i++) {
    desc->lower[i] = lower[i];
    desc->upper[i] = upper[i];
    desc->stride[i] = stride[i];
  }
}

static int _XMP_compare_task_exec_cond(_XMP_task_desc_t *task_desc, int *lower, int *upper, int *stride) {
  int dim = task_desc->dim;

  if ((_XMP_get_execution_nodes()->nodes_id) != (task_desc->inherit_nodes_id)) {
    return _XMP_N_INT_FALSE;
  }

  for (int i = 0; i < dim; i++) {
    if (((int)(task_desc->lower[i]) != lower[i]) || (int)((task_desc->upper[i]) != upper[i]) ||
        ((int)(task_desc->stride[i]) != stride[i])) {
      return _XMP_N_INT_FALSE;
    }
  }

  return _XMP_N_INT_TRUE;
}

static void _XMP_init_nodes_STATIC_NODES_NAMED_MAIN(_XMP_nodes_t **nodes, int dim,
                                                    _XMP_nodes_t *ref_nodes,
                                                    int *shrink, int *ref_lower, int *ref_upper, int *ref_stride,
                                                    int *dim_size) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper, ref_stride);

  for (int i = 0; i < dim; i++) {
    n->info[i].size = dim_size[i];
  }

  _XMP_check_nodes_size_STATIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  } else {
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;

  // calc inherit info
  n->inherit_nodes = ref_nodes;
  n->inherit_info = _XMP_calc_inherit_info_by_ref(ref_nodes, shrink, ref_lower, ref_upper, ref_stride);
}

void _XMP_init_nodes_STATIC_GLOBAL(_XMP_nodes_t **nodes, int dim, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_GLOBAL(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }

  // is_member is always true
  _XMP_check_nodes_size_STATIC(n);
  _XMP_calc_nodes_rank(n, _XMP_world_rank);

  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = (void *)n;
}

void _XMP_init_nodes_DYNAMIC_GLOBAL(_XMP_nodes_t **nodes, int dim, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_GLOBAL(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  int *last_dim_size_p = va_arg(args, int *);

  // is_member is always true
  _XMP_check_nodes_size_DYNAMIC(n);
  _XMP_calc_nodes_rank(n, _XMP_world_rank);

  *last_dim_size_p = n->info[dim - 1].size;
  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_init_nodes_STATIC_EXEC(_XMP_nodes_t **nodes, int dim, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_EXEC(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }

  _XMP_check_nodes_size_STATIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_init_nodes_DYNAMIC_EXEC(_XMP_nodes_t **nodes, int dim, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_EXEC(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  int *last_dim_size_p = va_arg(args, int *);

  _XMP_check_nodes_size_DYNAMIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  *last_dim_size_p = n->info[dim - 1].size;
  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_init_nodes_STATIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                         int ref_lower, int ref_upper, int ref_stride, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride);

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }

  _XMP_check_nodes_size_STATIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_init_nodes_DYNAMIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                          int ref_lower, int ref_upper, int ref_stride, ...) {
  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride);

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  int *last_dim_size_p = va_arg(args, int *);

  _XMP_check_nodes_size_DYNAMIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  *last_dim_size_p = n->info[dim - 1].size;
  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_init_nodes_STATIC_NODES_NAMED(_XMP_nodes_t **nodes, int dim,
                                        _XMP_nodes_t *ref_nodes, ...) {
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim];
  int ref_upper[ref_dim];
  int ref_stride[ref_dim];
  int dim_size[dim];

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      ref_lower[i] = va_arg(args, int);
      ref_upper[i] = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  for (int i = 0; i < dim; i++) {
    int dim_size_temp = va_arg(args, int);
    if (dim_size_temp <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    } else {
      dim_size[i] = dim_size_temp;
    }
  }

  _XMP_init_nodes_STATIC_NODES_NAMED_MAIN(nodes, dim,
                                          ref_nodes,
                                          shrink, ref_lower, ref_upper, ref_stride,
                                          dim_size);

  _XMP_nodes_t *n = *nodes;
  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_DYNAMIC_NODES_NAMED(_XMP_nodes_t **nodes, int dim,
                                         _XMP_nodes_t *ref_nodes, ...) {
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim];
  int ref_upper[ref_dim];
  int ref_stride[ref_dim];

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      ref_lower[i] = va_arg(args, int);
      ref_upper[i] = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper, ref_stride);

  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) _XMP_fatal("<nodes-size> should be less or equal to zero");

    n->info[i].size = dim_size;
  }
  int *last_dim_size_p = va_arg(args, int *);

  _XMP_check_nodes_size_DYNAMIC(n);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  *last_dim_size_p = n->info[dim - 1].size;
  for (int i = 0; i < dim; i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);

  *nodes = n;
}

void _XMP_finalize_nodes(_XMP_nodes_t *nodes) {
  _XMP_finalize_comm(nodes->comm);
  _XMP_free(nodes->inherit_info);
  _XMP_free(nodes);
}

int _XMP_exec_task_GLOBAL_PART(_XMP_task_desc_t **task_desc, int ref_lower, int ref_upper, int ref_stride) {
  int lower[1], upper[1], stride[1];
  lower[0] = ref_lower;
  upper[0] = ref_upper;
  stride[0] = ref_stride;

  _XMP_task_desc_t *desc = NULL;
  if (*task_desc == NULL) {
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
  } else {
    desc = *task_desc;
    if (_XMP_compare_task_exec_cond(desc, lower, upper, stride)) {
      if (desc->execute) {
        _XMP_push_nodes(desc->nodes);
        return _XMP_N_INT_TRUE;
      } else {
        return _XMP_N_INT_FALSE;
      }
    } else {
      if (desc->nodes != NULL) {
        _XMP_finalize_nodes(desc->nodes);
      }
    }
  }

  _XMP_nodes_t *n = NULL;
  _XMP_init_nodes_STATIC_NODES_NUMBER(&n, 1, ref_lower, ref_upper, ref_stride, _XMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride));
  _XMP_set_task_desc(desc, n->is_member, n, 1, lower, upper, stride);
  if (n->is_member) {
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

int _XMP_exec_task_NODES_ENTIRE(_XMP_task_desc_t **task_desc, _XMP_nodes_t *ref_nodes) {
  if (ref_nodes->is_member) {
    _XMP_push_nodes(ref_nodes);
    return true;
  }
  else {
    return false;
  }
}

int _XMP_exec_task_NODES_PART(_XMP_task_desc_t **task_desc, _XMP_nodes_t *ref_nodes, ...) {
  va_list args;
  va_start(args, ref_nodes);
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim], lower[ref_dim], upper[ref_dim], stride[ref_dim];
  int acc_dim_size = 1;
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (!shrink[i]) {
      lower[i] = va_arg(args, int);
      upper[i] = va_arg(args, int);
      stride[i] = va_arg(args, int);

      acc_dim_size *= _XMP_M_COUNT_TRIPLETi(lower[i], upper[i], stride[i]);
    }
  }
  va_end(args);

  _XMP_task_desc_t *desc = NULL;
  if (*task_desc == NULL) {
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
  } else {
    desc = *task_desc;
    if (_XMP_compare_task_exec_cond(desc, lower, upper, stride)) {
      if (desc->execute) {
        _XMP_push_nodes(desc->nodes);
        return _XMP_N_INT_TRUE;
      } else {
        return _XMP_N_INT_FALSE;
      }
    } else {
      if (desc->nodes != NULL) {
        _XMP_finalize_nodes(desc->nodes);
      }
    }
  }

  _XMP_nodes_t *n = NULL;
  _XMP_init_nodes_STATIC_NODES_NAMED_MAIN(&n, 1,
                                          ref_nodes,
                                          shrink, lower, upper, stride,
                                          &acc_dim_size);
  _XMP_set_task_desc(desc, n->is_member, n, ref_dim, lower, upper, stride);
  if (n->is_member) {
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    return _XMP_N_INT_FALSE;
  }
}

// FIXME do not use this function
_XMP_nodes_t *_XMP_create_nodes_by_comm(int is_member, _XMP_comm *comm) {
  int size;
  MPI_Comm_size(*((MPI_Comm *)comm), &size);

  _XMP_nodes_t *n = _XMP_create_new_nodes(is_member, 1, size, comm);

  n->info[0].size = size;
  if (is_member) {
    MPI_Comm_rank(*((MPI_Comm *)comm), &(n->info[0].rank));
  }

  // calc inherit info
  n->inherit_nodes = NULL;
  n->inherit_info = NULL;

  return n;
}

int _XMP_calc_linear_rank(_XMP_nodes_t *n, int *ranks) {
  _XMP_nodes_t *inherit_nodes = n->inherit_nodes;
  if (inherit_nodes == NULL) {
    int acc_rank = 0;
    int acc_nodes_size = 1;

    int nodes_dim = n->dim;
    for (int i = 0; i < nodes_dim; i++) {
      acc_rank += (ranks[i]) * acc_nodes_size;
      acc_nodes_size *= n->info[i].size;
    }

    return acc_rank;
  } else {
    int inherit_nodes_dim = inherit_nodes->dim;
    int *new_ranks = _XMP_alloc(sizeof(int) * inherit_nodes_dim);
    _XMP_nodes_inherit_info_t *inherit_info = n->inherit_info;

    int j = 0;
    for (int i = 0; i < inherit_nodes_dim; i++) {
      if (inherit_info[i].is_enable) {
        new_ranks[i] = ((inherit_info[i].stride) * ranks[j]) + (inherit_info[i].lower);
        j++;
      } else {
        // FIXME how implement ???
        new_ranks[i] = 0;
      }
    }

    int ret = _XMP_calc_linear_rank(inherit_nodes, new_ranks);
    _XMP_free(new_ranks);
    return ret;
  }
}
