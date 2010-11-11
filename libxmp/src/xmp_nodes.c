/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

// XXX <nodes-ref> is { 1-ogigin in language | 0-origin in runtime }, needs converting

static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_GLOBAL(int dim);
static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_EXEC(int dim);
static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper, int ref_stride);
static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_NODES_NAMED(int dim, _XCALABLEMP_nodes_t *ref_nodes,
                                                                      int *ref_lower, int *ref_upper, int *ref_stride);
static void _XCALABLEMP_calc_nodes_rank(_XCALABLEMP_nodes_t *n, int linear_rank);
static void _XCALABLEMP_disable_nodes_rank(_XCALABLEMP_nodes_t *n);
static void _XCALABLEMP_check_nodes_size_STATIC(_XCALABLEMP_nodes_t *n, int linear_size);
static void _XCALABLEMP_check_nodes_size_DYNAMIC(_XCALABLEMP_nodes_t *n, int linear_size, int linear_rank);
static _Bool _XCALABLEMP_check_nodes_ref_inclusion(int lower, int upper, int stride, int rank);

static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_GLOBAL(int dim) {
  _XCALABLEMP_nodes_t *n = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t) +
                                             sizeof(_XCALABLEMP_nodes_info_t) * (dim - 1));

  n->is_member = true;
  n->dim = dim;
  n->comm_size = _XCALABLEMP_world_size;

  n->comm_rank = _XCALABLEMP_world_rank;
  n->comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(MPI_COMM_WORLD, n->comm);

  return n;
}

static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_EXEC(int dim) {
  _XCALABLEMP_nodes_t *exec_nodes = _XCALABLEMP_get_execution_nodes();
  int size = exec_nodes->comm_size;
  int rank = exec_nodes->comm_rank;

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t) +
                                             sizeof(_XCALABLEMP_nodes_info_t) * (dim - 1));

  n->is_member = true;
  n->dim = dim;
  n->comm_size = size;

  n->comm_rank = rank;
  n->comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(*(exec_nodes->comm), n->comm);

  return n;
}

static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper, int ref_stride) {
  _Bool is_member = _XCALABLEMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, _XCALABLEMP_world_rank);

  MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  int color;
  if (is_member) {
    color = 1;
  }
  else {
    color = 0;
  }
  MPI_Comm_split(MPI_COMM_WORLD, color, _XCALABLEMP_world_rank, comm);

  _XCALABLEMP_nodes_t *n = n = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t) +
                                                 sizeof(_XCALABLEMP_nodes_info_t) * (dim - 1));

  n->is_member = is_member;
  n->dim = dim;
  n->comm_size = _XCALABLEMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride);

  if (is_member) {
    MPI_Comm_rank(*comm, &(n->comm_rank));
    n->comm = comm;

    int split_comm_size;
    MPI_Comm_size(*comm, &split_comm_size);
    if (split_comm_size != n->comm_size) {
      _XCALABLEMP_fatal("incorrect communicator size");
    }
  }
  else {
    _XCALABLEMP_finalize_comm(comm);

    n->comm_rank = _XCALABLEMP_N_INVALID_RANK;
    n->comm = NULL;
  }

  return n;
}

static _XCALABLEMP_nodes_t *_XCALABLEMP_init_nodes_struct_NODES_NAMED(int dim, _XCALABLEMP_nodes_t *ref_nodes,
                                                                      int *ref_lower, int *ref_upper, int *ref_stride) {
  _XCALABLEMP_ASSERT(ref_nodes != NULL);
  _XCALABLEMP_ASSERT(ref_lower != NULL);
  _XCALABLEMP_ASSERT(ref_upper != NULL);
  _XCALABLEMP_ASSERT(ref_stride != NULL);
  _XCALABLEMP_ASSERT(ref_nodes->is_member);

  int comm_size = 1;
  int ref_dim = ref_nodes->dim;
  _Bool is_member = true;
  for (int i = 0; i < ref_dim; i++) {
    comm_size *= _XCALABLEMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);
    is_member = is_member &&
                _XCALABLEMP_check_nodes_ref_inclusion(ref_lower[i], ref_upper[i], ref_stride[i], ref_nodes->info[i].rank);
  }

  MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  int color;
  if (is_member) {
    color = 1;
  }
  else {
    color = 0;
  }
  MPI_Comm_split(*(ref_nodes->comm), color, ref_nodes->comm_rank, comm);

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_alloc(sizeof(_XCALABLEMP_nodes_t) +
                                             sizeof(_XCALABLEMP_nodes_info_t) * (dim - 1));

  n->is_member = is_member;
  n->dim = dim;
  n->comm_size = comm_size;

  if (is_member) {
    MPI_Comm_rank(*comm, &(n->comm_rank));
    n->comm = comm;

    int split_comm_size;
    MPI_Comm_size(*comm, &split_comm_size);
    if (split_comm_size != n->comm_size) {
      _XCALABLEMP_fatal("incorrect communicator size");
    }
  }
  else {
    _XCALABLEMP_finalize_comm(comm);

    n->comm_rank = _XCALABLEMP_N_INVALID_RANK;
    n->comm = NULL;
  }

  return n;
}

static void _XCALABLEMP_calc_nodes_rank(_XCALABLEMP_nodes_t *n, int linear_rank) {
  _XCALABLEMP_ASSERT(n != NULL);
  _XCALABLEMP_ASSERT(n->is_member);

  int acc_size = 1;
  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    int dim_size = n->info[i].size;
    n->info[i].rank = (linear_rank / acc_size) % dim_size;
    acc_size *= dim_size;
  }
}

static void _XCALABLEMP_disable_nodes_rank(_XCALABLEMP_nodes_t *n) {
  _XCALABLEMP_ASSERT(n != NULL);
  _XCALABLEMP_ASSERT(!n->is_member);

  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    n->info[i].rank = _XCALABLEMP_N_INVALID_RANK;
  }
}

static void _XCALABLEMP_check_nodes_size_STATIC(_XCALABLEMP_nodes_t *n, int linear_size) {
  _XCALABLEMP_ASSERT(n != NULL);

  int acc_size = 1;
  int dim = n->dim;
  for (int i = 0; i < dim; i++) {
    acc_size *= n->info[i].size;
  }

  if (acc_size != linear_size) {
    _XCALABLEMP_fatal("incorrect communicator size");
  }
}

static void _XCALABLEMP_check_nodes_size_DYNAMIC(_XCALABLEMP_nodes_t *n, int linear_size, int linear_rank) {
  _XCALABLEMP_ASSERT(n != NULL);

  int acc_size = 1;
  int dim = n->dim;
  for (int i = 0; i < dim - 1; i++) {
    acc_size *= n->info[i].size;
  }

  if (acc_size > linear_size) {
    _XCALABLEMP_fatal("indicated communicator size is bigger than the actual communicator size");
  }

  if ((linear_size % acc_size) != 0) {
    _XCALABLEMP_fatal("cannot determine communicator size dynamically");
  }

  int end_size = linear_size / acc_size;
  n->info[dim-1].size = end_size;

  if (n->is_member) {
    n->info[dim-1].rank = (linear_rank / acc_size) % end_size;
  }
}

static _Bool _XCALABLEMP_check_nodes_ref_inclusion(int lower, int upper, int stride, int rank) {
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

// XXX args are 1-origin
void _XCALABLEMP_validate_nodes_ref(int *lower, int *upper, int *stride, int size) {
  _XCALABLEMP_ASSERT(lower != NULL);
  _XCALABLEMP_ASSERT(upper != NULL);
  _XCALABLEMP_ASSERT(stride != NULL);

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
    _XCALABLEMP_fatal("the stride of <nodes-ref> is 0");
    l = 0; u = 0; // XXX dummy
  }

  // check boundary
  if (1 > l) {
    _XCALABLEMP_fatal("<nodes-ref> is out of bounds, <ref-lower> is less than 1");
  }

  if (l > u) {
    _XCALABLEMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is less than <ref-lower>");
  }

  if (u > size) {
    _XCALABLEMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is greater than the node size");
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

void _XCALABLEMP_init_nodes_STATIC_GLOBAL(int map_type, _XCALABLEMP_nodes_t **nodes, int dim, ...) {
  // FIXME <map-type> is ignored

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_GLOBAL(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  // is_member is always true
  _XCALABLEMP_check_nodes_size_STATIC(n, _XCALABLEMP_world_size);
  _XCALABLEMP_calc_nodes_rank(n, _XCALABLEMP_world_rank);

  *nodes = (void *)n;
}

void _XCALABLEMP_init_nodes_DYNAMIC_GLOBAL(int map_type, _XCALABLEMP_nodes_t **nodes, int dim, ...) {
  // FIXME <map-type> is ignored

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_GLOBAL(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  // is_member is always true
  _XCALABLEMP_check_nodes_size_DYNAMIC(n, _XCALABLEMP_world_size, _XCALABLEMP_world_rank);
  _XCALABLEMP_calc_nodes_rank(n, _XCALABLEMP_world_rank);

  *nodes = n;
}

void _XCALABLEMP_init_nodes_STATIC_EXEC(int map_type, _XCALABLEMP_nodes_t **nodes, int dim, ...) {
  // FIXME <map-type> is ignored

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_EXEC(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  _XCALABLEMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XCALABLEMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_init_nodes_DYNAMIC_EXEC(int map_type, _XCALABLEMP_nodes_t **nodes, int dim, ...) {
  // FIXME <map-type> is ignored

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_EXEC(dim);

  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XCALABLEMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XCALABLEMP_N_INVALID_RANK);
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_init_nodes_STATIC_NODES_NUMBER(int map_type, _XCALABLEMP_nodes_t **nodes, int dim,
                                                int ref_lower, int ref_upper, int ref_stride, ...) {
  _XCALABLEMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XCALABLEMP_world_size);

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride);

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  _XCALABLEMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XCALABLEMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_init_nodes_DYNAMIC_NODES_NUMBER(int map_type, _XCALABLEMP_nodes_t **nodes, int dim,
                                                 int ref_lower, int ref_upper, int ref_stride, ...) {
  _XCALABLEMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XCALABLEMP_world_size);

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, ref_stride);

  va_list args;
  va_start(args, ref_stride);
  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XCALABLEMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XCALABLEMP_N_INVALID_RANK);
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_init_nodes_STATIC_NODES_NAMED(int get_upper, int map_type, _XCALABLEMP_nodes_t **nodes, int dim,
                                               _XCALABLEMP_nodes_t *ref_nodes, ...) {
  _XCALABLEMP_ASSERT(ref_nodes != NULL);

  if (!ref_nodes->is_member) {
    _XCALABLEMP_fatal("cannot create a new nodes descriptor");
  }

  int ref_dim = ref_nodes->dim;
  int *ref_lower = _XCALABLEMP_alloc(sizeof(int) * ref_dim);
  int *ref_upper = _XCALABLEMP_alloc(sizeof(int) * ref_dim);
  int *ref_stride = _XCALABLEMP_alloc(sizeof(int) * ref_dim);

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

    _XCALABLEMP_validate_nodes_ref(&(ref_lower[i]), &(ref_upper[i]), &(ref_stride[i]), ref_nodes->info[i].size);
  }

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, ref_lower, ref_upper, ref_stride);

  _XCALABLEMP_free(ref_lower);
  _XCALABLEMP_free(ref_upper);
  _XCALABLEMP_free(ref_stride);

  for (int i = 0; i < dim; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) {
      _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");
    }

    n->info[i].size = dim_size;
  }
  va_end(args);

  _XCALABLEMP_check_nodes_size_STATIC(n, n->comm_size);
  if (n->is_member) {
    _XCALABLEMP_calc_nodes_rank(n, n->comm_rank);
  }
  else {
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_init_nodes_DYNAMIC_NODES_NAMED(int get_upper, int map_type, _XCALABLEMP_nodes_t **nodes, int dim,
                                               _XCALABLEMP_nodes_t *ref_nodes, ...) {
  _XCALABLEMP_ASSERT(ref_nodes != NULL);

  if (!ref_nodes->is_member) {
    _XCALABLEMP_fatal("cannot create a new nodes descriptor");
  }

  int ref_dim = ref_nodes->dim;
  int *ref_lower = _XCALABLEMP_alloc(sizeof(int) * ref_dim);
  int *ref_upper = _XCALABLEMP_alloc(sizeof(int) * ref_dim);
  int *ref_stride = _XCALABLEMP_alloc(sizeof(int) * ref_dim);

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

    _XCALABLEMP_validate_nodes_ref(&(ref_lower[i]), &(ref_upper[i]), &(ref_stride[i]), ref_nodes->info[i].size);
  }

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, ref_lower, ref_upper, ref_stride);

  _XCALABLEMP_free(ref_lower);
  _XCALABLEMP_free(ref_upper);
  _XCALABLEMP_free(ref_stride);

  for (int i = 0; i < dim - 1; i++) {
    int dim_size = va_arg(args, int);
    if (dim_size <= 0) _XCALABLEMP_fatal("<nodes-size> should be less or equal to zero");

    n->info[i].size = dim_size;
  }
  va_end(args);

  if (n->is_member) {
    int linear_rank = n->comm_rank;
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, linear_rank);
    _XCALABLEMP_calc_nodes_rank(n, linear_rank);
  }
  else {
    _XCALABLEMP_check_nodes_size_DYNAMIC(n, n->comm_size, _XCALABLEMP_N_INVALID_RANK);
    _XCALABLEMP_disable_nodes_rank(n);
  }

  *nodes = n;
}

void _XCALABLEMP_finalize_nodes(_XCALABLEMP_nodes_t *nodes) {
  _XCALABLEMP_ASSERT(nodes != NULL);

  if (nodes->is_member) {
    _XCALABLEMP_finalize_comm(nodes->comm);
  }

  _XCALABLEMP_free(nodes);
}

_Bool _XCALABLEMP_exec_task_GLOBAL_PART(int ref_lower, int ref_upper, int ref_stride) {
  _XCALABLEMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XCALABLEMP_world_size);

  _XCALABLEMP_nodes_t *n = _XCALABLEMP_init_nodes_struct_NODES_NUMBER(0, ref_lower, ref_upper, ref_stride);
  if (n->is_member) {
    _XCALABLEMP_push_nodes(n);
    return true;
  }
  else {
    _XCALABLEMP_finalize_nodes(n);
    return false;
  }
}

_Bool _XCALABLEMP_exec_task_NODES_ENTIRE(_XCALABLEMP_nodes_t *ref_nodes) {
  _XCALABLEMP_ASSERT(ref_nodes != NULL);

  if (ref_nodes->is_member) {
    _XCALABLEMP_push_nodes(ref_nodes);
    return true;
  }
  else {
    return false;
  }
}

_Bool _XCALABLEMP_exec_task_NODES_PART(int get_upper, _XCALABLEMP_nodes_t *ref_nodes, ...) {
  _XCALABLEMP_ASSERT(ref_nodes != NULL);

  if (!ref_nodes->is_member) {
    return false;
  }

  int color = 1;
  _Bool is_member = true;
  int acc_nodes_size = 1;
  int ref_dim = ref_nodes->dim;
  int ref_lower, ref_upper, ref_stride;

  va_list args;
  va_start(args, ref_nodes);
  for (int i = 0; i < ref_dim; i++) {
    int size = ref_nodes->info[i].size;
    int rank = ref_nodes->info[i].rank;

    if (va_arg(args, int) == 1) {
      color += (acc_nodes_size * rank);
    }
    else {
      ref_lower = va_arg(args, int);
      if ((i == (ref_dim - 1)) && (get_upper == 1)) {
        ref_upper = size;
      }
      else {
        ref_upper = va_arg(args, int);
      }
      ref_stride = va_arg(args, int);

      _XCALABLEMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, size);

      is_member = is_member && _XCALABLEMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, rank);
    }

    acc_nodes_size *= size;
  }

  MPI_Comm *comm = _XCALABLEMP_alloc(sizeof(MPI_Comm));
  if (!is_member) {
    color = 0;
  }

  MPI_Comm_split(*(ref_nodes->comm), color, ref_nodes->comm_rank, comm);

  if (is_member) {
    _XCALABLEMP_push_comm(comm);
    return true;
  }
  else {
    _XCALABLEMP_finalize_comm(comm);
    return false;
  }
}

_XCALABLEMP_nodes_t *_XCALABLEMP_create_nodes_by_comm(MPI_Comm *comm) {
  _XCALABLEMP_ASSERT(comm != NULL);

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

  return n;
}
