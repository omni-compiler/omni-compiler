/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <limits.h>
#include <stdarg.h>
#include "mpi.h"
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

// XXX <nodes-ref> is { 1-ogigin in language | 0-origin in runtime }, needs converting

static unsigned long long _XMP_nodes_id_counter = 0;

static unsigned long long _XMP_get_new_nodes_id(void) {
  unsigned long long ret = _XMP_nodes_id_counter;
  if (ret == ULLONG_MAX) {
    _XMP_fatal("cannot create a new nodes descriptor: too many nodes");
  }

  _XMP_nodes_id_counter++;
  return ret;
}

static _Bool _XMP_check_nodes_ref_inclusion(int lower, int upper, int stride, int rank) {
  if (rank < lower) {
    return false;
  }

  if (rank > upper) {
    return false;
  }

  if (((rank - lower) % stride) == 0) {
    return true;
  }
  else {
    return false;
  }
}

static _XMP_nodes_t *_XMP_init_nodes_struct_GLOBAL(int dim) {
  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  n->nodes_id = _XMP_get_new_nodes_id();
  n->is_member = true;
  n->dim = dim;
  n->comm_size = _XMP_world_size;

  n->comm_rank = _XMP_world_rank;
  n->comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(MPI_COMM_WORLD, n->comm);

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_EXEC(int dim) {
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int size = exec_nodes->comm_size;
  int rank = exec_nodes->comm_rank;

  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  n->nodes_id = _XMP_get_new_nodes_id();
  n->is_member = true;
  n->dim = dim;
  n->comm_size = size;

  n->comm_rank = rank;
  n->comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(*((MPI_Comm *)exec_nodes->comm), n->comm);

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper, int ref_stride) {
  _Bool is_member = _XMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, _XMP_world_rank);

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  int color;
  if (is_member) {
    color = 1;
  }
  else {
    color = 0;
  }
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, _XMP_world_rank, comm);

  _XMP_nodes_t *n = n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  n->nodes_id = _XMP_get_new_nodes_id();
  n->is_member = is_member;
  n->dim = dim;
  n->comm_size = _XMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride);

  if (is_member) {
    MPI_Comm_rank(*comm, &(n->comm_rank));
    n->comm = comm;

    int split_comm_size;
    MPI_Comm_size(*comm, &split_comm_size);
    if (split_comm_size != n->comm_size) {
      _XMP_fatal("incorrect communicator size");
    }
  }
  else {
    _XMP_finalize_comm(comm);

    n->comm_rank = _XMP_N_INVALID_RANK;
    n->comm = NULL;
  }

  return n;
}

static _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NAMED(int dim, _XMP_nodes_t *ref_nodes,
                                                        int *ref_lower, int *ref_upper, int *ref_stride) {
  _XMP_ASSERT(ref_nodes->is_member);

  int comm_size = 1;
  int ref_dim = ref_nodes->dim;
  _Bool is_member = true;
  for (int i = 0; i < ref_dim; i++) {
    comm_size *= _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);
    is_member = is_member &&
                _XMP_check_nodes_ref_inclusion(ref_lower[i], ref_upper[i], ref_stride[i], ref_nodes->info[i].rank);
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  int color;
  if (is_member) {
    color = 1;
  }
  else {
    color = 0;
  }
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, ref_nodes->comm_rank, comm);

  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  n->nodes_id = _XMP_get_new_nodes_id();
  n->is_member = is_member;
  n->dim = dim;
  n->comm_size = comm_size;

  if (is_member) {
    MPI_Comm_rank(*comm, &(n->comm_rank));
    n->comm = comm;

    int split_comm_size;
    MPI_Comm_size(*comm, &split_comm_size);
    if (split_comm_size != n->comm_size) {
      _XMP_fatal("incorrect communicator size");
    }
  }
  else {
    _XMP_finalize_comm(comm);

    n->comm_rank = _XMP_N_INVALID_RANK;
    n->comm = NULL;
  }

  return n;
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

static void _XMP_check_nodes_size_STATIC(_XMP_nodes_t *n, int linear_size) {
  int acc_size = 1;
  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    acc_size *= n->info[i].size;
  }

  if (acc_size != linear_size) {
    _XMP_fatal("incorrect communicator size");
  }
}

static void _XMP_check_nodes_size_DYNAMIC(_XMP_nodes_t *n, int linear_size, int linear_rank) {
  int acc_size = 1;
  int dim = n->dim;
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

// XXX args are 1-origin
void _XMP_validate_nodes_ref(int *lower, int *upper, int *stride, int size) {
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
  va_end(args);

  // is_member is always true
  _XMP_check_nodes_size_STATIC(n, _XMP_world_size);
  _XMP_calc_nodes_rank(n, _XMP_world_rank);

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
  va_end(args);

  // is_member is always true
  _XMP_check_nodes_size_DYNAMIC(n, _XMP_world_size, _XMP_world_rank);
  _XMP_calc_nodes_rank(n, _XMP_world_rank);

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
  va_end(args);

  _XMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

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
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XMP_N_INVALID_RANK);
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XMP_init_nodes_STATIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                         int ref_lower, int ref_upper, int ref_stride, ...) {
  _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XMP_world_size);

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
  va_end(args);

  _XMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XMP_init_nodes_DYNAMIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                          int ref_lower, int ref_upper, int ref_stride, ...) {
  _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XMP_world_size);

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
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XMP_N_INVALID_RANK);
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XMP_init_nodes_STATIC_NODES_NAMED(int get_upper, _XMP_nodes_t **nodes, int dim,
                                        _XMP_nodes_t *ref_nodes, ...) {
  if (!ref_nodes->is_member) {
    _XMP_fatal("cannot create a new nodes descriptor");
  }

  int ref_dim = ref_nodes->dim;
  int *ref_lower = _XMP_alloc(sizeof(int) * ref_dim);
  int *ref_upper = _XMP_alloc(sizeof(int) * ref_dim);
  int *ref_stride = _XMP_alloc(sizeof(int) * ref_dim);

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    ref_lower[i] = va_arg(args, int);
    if ((i == (ref_dim - 1)) && (get_upper == 1)) {
      ref_upper[i] = ref_nodes->info[i].size;
    }
    else {
      ref_upper[i] = va_arg(args, int);
    }
    ref_stride[i] = va_arg(args, int);

    _XMP_validate_nodes_ref(&(ref_lower[i]), &(ref_upper[i]), &(ref_stride[i]), ref_nodes->info[i].size);
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, ref_lower, ref_upper, ref_stride);

  _XMP_free(ref_lower);
  _XMP_free(ref_upper);
  _XMP_free(ref_stride);

  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  _XMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XMP_init_nodes_DYNAMIC_NODES_NAMED(int get_upper, _XMP_nodes_t **nodes, int dim,
                                         _XMP_nodes_t *ref_nodes, ...) {
  if (!ref_nodes->is_member) {
    _XMP_fatal("cannot create a new nodes descriptor");
  }

  int ref_dim = ref_nodes->dim;
  int *ref_lower = _XMP_alloc(sizeof(int) * ref_dim);
  int *ref_upper = _XMP_alloc(sizeof(int) * ref_dim);
  int *ref_stride = _XMP_alloc(sizeof(int) * ref_dim);

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    ref_lower[i] = va_arg(args, int);
    if ((i == (ref_dim - 1)) && (get_upper == 1)) {
      ref_upper[i] = ref_nodes->info[i].size;
    }
    else {
      ref_upper[i] = va_arg(args, int);
    }
    ref_stride[i] = va_arg(args, int);

    _XMP_validate_nodes_ref(&(ref_lower[i]), &(ref_upper[i]), &(ref_stride[i]), ref_nodes->info[i].size);
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, ref_lower, ref_upper, ref_stride);

  _XMP_free(ref_lower);
  _XMP_free(ref_upper);
  _XMP_free(ref_stride);

  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) _XMP_fatal("<nodes-size> should be less or equal to zero");

    n->info[i].size = dim_size;
  }
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XMP_N_INVALID_RANK);
    _XMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XMP_finalize_nodes(_XMP_nodes_t *nodes) {
  if (nodes->is_member) {
    _XMP_finalize_comm(nodes->comm);
  }

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

  _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XMP_world_size);

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NUMBER(0, ref_lower, ref_upper, ref_stride);
  if (n->is_member) {
    _XMP_set_task_desc(desc, _XMP_N_INT_TRUE, n, 1, lower, upper, stride);
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    _XMP_set_task_desc(desc, _XMP_N_INT_FALSE, NULL, 1, lower, upper, stride);
    _XMP_finalize_nodes(n);
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

int _XMP_exec_task_NODES_PART(_XMP_task_desc_t **task_desc, int get_upper, _XMP_nodes_t *ref_nodes, ...) {
  int shrink[_XMP_N_MAX_DIM], lower[_XMP_N_MAX_DIM], upper[_XMP_N_MAX_DIM], stride[_XMP_N_MAX_DIM];
  va_list args;
  va_start(args, ref_nodes);
  int ref_dim = ref_nodes->dim;
  for (int i = 0; i < ref_dim; i++) {
    shrink[i] = va_arg(args, int);
    if (shrink[i] == 1) {
      lower[i] = 1;
      upper[i] = ref_nodes->info[i].size;
      stride[i] = 1;
    } else {
      lower[i] = va_arg(args, int);
      upper[i] = va_arg(args, int);
      stride[i] = va_arg(args, int);
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

  if (!ref_nodes->is_member) {
    _XMP_set_task_desc(desc, _XMP_N_INT_FALSE, NULL, ref_dim, lower, upper, stride);
    return _XMP_N_INT_FALSE;
  }

  int color = 1;
  _Bool is_member = true;
  int acc_nodes_size = 1;
  int ref_lower, ref_upper, ref_stride;
  for (int i = 0; i < ref_dim; i++) {
    int size = ref_nodes->info[i].size;
    int rank = ref_nodes->info[i].rank;

    if (shrink[i] == 1) {
      color += (acc_nodes_size * rank);
    } else {
      ref_lower = lower[i];
      if ((i == (ref_dim - 1)) && (get_upper == 1)) {
        ref_upper = size;
      } else {
        ref_upper = upper[i];
      }
      ref_stride = stride[i];

      _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, size);

      is_member = is_member && _XMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, rank);
    }

    acc_nodes_size *= size;
  }

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  if (!is_member) {
    color = 0;
  }

  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), color, ref_nodes->comm_rank, comm);

  if (is_member) {
    _XMP_nodes_t *n = _XMP_create_nodes_by_comm(comm);
    _XMP_set_task_desc(desc, _XMP_N_INT_TRUE, n, ref_dim, lower, upper, stride);
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  } else {
    _XMP_set_task_desc(desc, _XMP_N_INT_FALSE, NULL, ref_dim, lower, upper, stride);
    _XMP_finalize_comm(comm);
    return _XMP_N_INT_FALSE;
  }
}

_XMP_nodes_t *_XMP_create_nodes_by_comm(_XMP_comm *comm) {
  int size, rank;
  MPI_Comm_size(*((MPI_Comm *)comm), &size);
  MPI_Comm_rank(*((MPI_Comm *)comm), &rank);

  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t));

  n->nodes_id = _XMP_get_new_nodes_id();
  n->is_member = true;
  n->dim = 1;

  n->comm = comm;
  n->comm_size = size;
  n->comm_rank = rank;

  n->info[0].size = size;
  n->info[0].rank = rank;

  return n;
}
