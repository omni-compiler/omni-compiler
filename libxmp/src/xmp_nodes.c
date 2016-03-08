#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"
void xmp_dbg_printf(char *fmt,...);

// <nodes-ref> is { 1-ogigin in language | 0-origin in runtime }, needs converting
static _XMP_nodes_t *_XMP_create_new_nodes(int is_member, int dim, int comm_size, _XMP_comm_t *comm)
{
  _XMP_nodes_t *n = _XMP_alloc(sizeof(_XMP_nodes_t) + sizeof(_XMP_nodes_info_t) * (dim - 1));

  n->on_ref_id = _XMP_get_on_ref_id();
  n->is_member = is_member;
  n->dim       = dim;
  n->comm_size = comm_size;
  n->comm      = comm;

  if(is_member){
    int size, rank;
    MPI_Comm_size(*((MPI_Comm *)comm), &size);
    MPI_Comm_rank(*((MPI_Comm *)comm), &rank);

    n->comm_rank = rank;
    if(size != comm_size)
      _XMP_fatal("cannot create a new nodes descriptor: communicator size is not correct");
  }
  else{
    n->comm_rank = _XMP_N_INVALID_RANK;
  }

  return n;
}

// args are 1-origin
static void _XMP_validate_nodes_ref(int *lower, int *upper, int *stride, int size)
{
  // setup temporary variables
  int l, u, s = *(stride);
  if(s > 0){
    l = *lower;
    u = *upper;
  }
  else if(s < 0){
    l = *upper;
    u = *lower;
  }
  else{
    _XMP_fatal("the stride of <nodes-ref> is 0");
    l = 0; u = 0; // dummy
  }

  // check boundary
  if(1 > l)
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-lower> is less than 1");

  if(l > u)
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is less than <ref-lower>");

  if(u > size)
    _XMP_fatal("<nodes-ref> is out of bounds, <ref-upper> is greater than the node size");

  // validate values
  if(s > 0){
    u = u - ((u - l) % s);
    *upper = u;
  }
  else{
    s = -s;
    l = l + ((u - l) % s);
    *lower = l;
    *upper = u;
    *stride = s;
  }

  // Convert 1-origin to 0-origin
  (*lower)--;
  (*upper)--;
}

static int _XMP_check_nodes_ref_inclusion(int lower, int upper, int stride, int size, int rank)
{
  if(rank < lower || rank > upper || ((rank - lower) % stride != 0))
    return _XMP_N_INT_FALSE;

  return _XMP_N_INT_TRUE;
}

static _XMP_nodes_inherit_info_t *_XMP_calc_inherit_info(_XMP_nodes_t *n)
{
  int dim = n->dim;
  _XMP_nodes_inherit_info_t *inherit_info = _XMP_alloc(sizeof(_XMP_nodes_inherit_info_t) * dim);

  for(int i=0;i<dim;i++){
    inherit_info[i].shrink = _XMP_N_INT_FALSE;
    inherit_info[i].lower  = 0;
    inherit_info[i].upper  = n->info[i].size - 1;
    inherit_info[i].stride = 1;
    inherit_info[i].size   = n->info[i].size;
  }

  return inherit_info;
}

static _XMP_nodes_inherit_info_t *_XMP_calc_inherit_info_by_ref(_XMP_nodes_t *n, int *shrink,
								int *lower, int *upper, int *stride)
{
  int dim = n->dim;
  _XMP_nodes_inherit_info_t *inherit_info = _XMP_alloc(sizeof(_XMP_nodes_inherit_info_t) * dim);

  for(int i=0;i<dim;i++){
    if(shrink[i]){
      inherit_info[i].shrink = _XMP_N_INT_TRUE;
    }
    else{
      inherit_info[i].shrink = _XMP_N_INT_FALSE;
      inherit_info[i].lower  = lower[i];
      inherit_info[i].upper  = upper[i];
      inherit_info[i].stride = stride[i];
    }
    
    inherit_info[i].size = n->info[i].size;
  }

  return inherit_info;
}

static void _XMP_init_nodes_info(_XMP_nodes_t *n, int *dim_size, int is_static)
{
  int acc_size    = 1;
  int dim         = n->dim;
  int linear_size = n->comm_size;
  
  for(int i=0;i<dim-1;i++){
    int dim_size_temp = dim_size[i];
    n->info[i].size   = dim_size_temp;
    acc_size         *= dim_size_temp;
  }

  if(is_static){
    int dim_size_temp   = dim_size[dim-1];
    n->info[dim-1].size = dim_size_temp;
    acc_size           *= dim_size_temp;

    if(acc_size != linear_size)
      _XMP_fatal("incorrect communicator size");
  }
  else{
    if(acc_size > linear_size)
      _XMP_fatal("indicated communicator size is bigger than the actual communicator size");

    if((linear_size % acc_size) != 0)
      _XMP_fatal("cannot determine communicator size dynamically");

    int end_size = linear_size / acc_size;
    n->info[dim-1].size = end_size;
  }

  if(n->is_member){
    int acc_size = 1;
    for(int i=0;i<dim;i++){
      int dim_size_temp = n->info[i].size;
      n->info[i].rank = (n->comm_rank / acc_size) % dim_size_temp;
      acc_size *= dim_size_temp;
    }
  }
}

static void _XMP_set_task_desc(_XMP_task_desc_t *desc, _XMP_nodes_t *n, int execute,
                               _XMP_nodes_t *ref_nodes, int *ref_lower, int *ref_upper, int *ref_stride)
{
  desc->nodes     = n;
  desc->execute   = execute;
  desc->on_ref_id = ref_nodes->on_ref_id;

  int dim = ref_nodes->dim;
  for(int i=0;i<dim;i++){
    desc->ref_lower[i]  = ref_lower[i];
    desc->ref_upper[i]  = ref_upper[i];
    desc->ref_stride[i] = ref_stride[i];
  }
}

static int _XMP_compare_task_exec_cond(_XMP_task_desc_t *task_desc, _XMP_nodes_t *ref_nodes,
                                       int *ref_lower, int *ref_upper, int *ref_stride)
{
  // FIXME can use compare_nodes?
  if(ref_nodes->on_ref_id != task_desc->on_ref_id)
    return _XMP_N_INT_FALSE;

  int dim = ref_nodes->dim;
  for(int i=0;i<dim;i++){
    if(task_desc->ref_stride[i] == -1 && ref_stride[i] == -1) continue;

    // NOTE: task_desc->ref is 0-origin. ref is 1-origine.
    if((task_desc->ref_lower[i] != ref_lower[i] - 1) ||
       (task_desc->ref_upper[i] != ref_upper[i] - 1) ||
       (task_desc->ref_stride[i] != ref_stride[i]))
      return _XMP_N_INT_FALSE;
  }

  return _XMP_N_INT_TRUE;
}

static int _XMP_compare_nodes(_XMP_nodes_t *a, _XMP_nodes_t *b)
{
  if(a->on_ref_id == b->on_ref_id)
    return _XMP_N_INT_TRUE;

  if(a->comm_size == _XMP_world_size && a->comm_size == b->comm_size)
    return _XMP_N_INT_TRUE;

  // compare nodes dim
  if(a->dim != b->dim) return _XMP_N_INT_FALSE;

  // compare nodes size
  int dim = a->dim;
  for(int i=0;i<dim;i++)
    if(a->info[i].size != b->info[i].size)
      return _XMP_N_INT_FALSE;

  // compare inherit nodes
  _XMP_nodes_t *inherit_nodes = a->inherit_nodes;
  if(inherit_nodes != b->inherit_nodes)
    return _XMP_N_INT_FALSE;

  if(inherit_nodes != NULL){
    int inherit_nodes_dim = inherit_nodes->dim;
    for(int i=0;i<inherit_nodes_dim;i++){
      _XMP_nodes_inherit_info_t *a_inherit_info = &(a->inherit_info[i]);
      _XMP_nodes_inherit_info_t *b_inherit_info = &(b->inherit_info[i]);

      int shrink = a_inherit_info->shrink;
      if(shrink != b_inherit_info->shrink)
        return _XMP_N_INT_FALSE;

      if(!shrink){
        if ((a_inherit_info->lower != b_inherit_info->lower) ||
            (a_inherit_info->upper != b_inherit_info->upper) ||
            (a_inherit_info->stride != b_inherit_info->stride))
          return _XMP_N_INT_FALSE;
      }
    }
  }
  return _XMP_N_INT_TRUE;
}

_XMP_nodes_t *_XMP_create_temporary_nodes(_XMP_nodes_t *n)
{
  int is_member       = n->is_member;
  int onto_nodes_dim  = n->dim;
  int onto_nodes_size = n->comm_size;
  int dim_size[onto_nodes_dim];
  _XMP_nodes_t *new_node = _XMP_create_new_nodes(is_member, onto_nodes_dim, onto_nodes_size, n->comm);
  _XMP_nodes_inherit_info_t *inherit_info = _XMP_alloc(sizeof(_XMP_nodes_inherit_info_t) * onto_nodes_dim);

  for(int i=0;i<onto_nodes_dim;i++){
    inherit_info[i].shrink = _XMP_N_INT_FALSE;
    inherit_info[i].lower  = 1;
    inherit_info[i].upper  = n->info[i].size;
    inherit_info[i].stride = 1;
    dim_size[i]            = n->info[i].size;
  }

  new_node->inherit_nodes = new_node;
  new_node->inherit_info  = inherit_info;

  _XMP_init_nodes_info(new_node, dim_size, _XMP_N_INT_TRUE);

  new_node->info[0].multiplier = 1;
  for(int i=1;i<onto_nodes_dim;i++)
    new_node->info[i].multiplier = new_node->info[i-1].multiplier * dim_size[i-1];

  return new_node;
}

static int get_subcomm_color(int dim, _XMP_nodes_t* n, int shrink[dim])
{
  int acc_nodes_size = 1, color = 1;
  
  for(int i=0;i<dim;i++){
    int size = n->info[i].size;
    int rank = n->info[i].rank;

    if(shrink[i])
      color += (acc_nodes_size * rank);

    acc_nodes_size *= size;
  }

  return color;
}

// Create sub communicators:
//   For example, when a node set is p(i,j,k),
//   eight sub communicators, p(:,:,:), p(*,:,:), p(:,*,:),
//   p(*,*,:), p(:,:,*), p(*,:,*), p(:,*,*), p(*,*,*) are created.
//   The number of sub communicators is 2^{dim}.
//   Note that the sub communicators are made from only XMP entire node set.
static _XMP_comm_t* create_subcomm(_XMP_nodes_t* n)
{
  if(n->attr != _XMP_ENTIRE_NODES)
    _XMP_fatal("Sub communicators are made from only XMP entire node set.\n");
  
  // Set shrink:
  //   For example, When dim is 3, following values are set in shrink[][].
  //   shrink[][] = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}, {0,0,1},
  //                 {1,0,1}, {0,1,1}, {1,1,1}}
  //   "1" means "*", "0" means ":".
  //   {1,0,0} means p(*,:,:).
  //   {0,1,1} means p(:,*,*).
  //   {0,0,0} meams p(:,:,:) which is a reference communicator.
  //   {1,1,1} means p(*,*,*) which is a MPI_COMM_SELF.
  int dim       = n->dim;
  int num_comms = 1<<dim;
  int shrink[num_comms][dim];
  
  for(int i=0;i<num_comms;i++)
    for(int j=0;j<dim;j++)
      shrink[i][j] = (i&(1<<j))/(1<<j);

  // Create sub communicators
  MPI_Comm *subcomm  = malloc(sizeof(MPI_Comm) * num_comms);
  MPI_Comm *ref_comm = (MPI_Comm *)n->comm;
  for(int i=1;i<num_comms-1;i++){
    int color = get_subcomm_color(dim, n, shrink[i]);
    MPI_Comm_split(*ref_comm, color, _XMP_world_rank, &subcomm[i]);
  }
  MPI_Comm_dup(*ref_comm, &subcomm[0]);               // i == 0
  MPI_Comm_dup(MPI_COMM_SELF, &subcomm[num_comms-1]); // i == num_comms-1

  return (_XMP_comm_t*)subcomm;
}

_XMP_nodes_t *_XMP_init_nodes_struct_GLOBAL(int dim, int *dim_size, int is_static)
{
  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(MPI_COMM_WORLD, comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(_XMP_N_INT_TRUE, dim, _XMP_world_size, (_XMP_comm_t *)comm);

  // calc inherit info
  n->inherit_nodes = NULL;
  n->inherit_info  = NULL;
  n->attr          = _XMP_ENTIRE_NODES;

  // set dim_size if XMP_NODE_SIZEn is set.
  if(!is_static){
    is_static = _XMP_N_INT_TRUE;
    for(int i=0;i<dim;i++){
      if(dim_size[i] == -1){
	char name[20];
	sprintf(name, "XMP_NODE_SIZE%d", i);
	char *size = getenv(name);

	if(!size){
	  if(i == dim - 1){
	    is_static = _XMP_N_INT_FALSE;
	    break;
	  }
	  else _XMP_fatal("XMP_NODE_SIZE not specified although '*' is in the dimension of a node array\n");
	}
	else{
	  dim_size[i] = atoi(size);
	  if(dim_size[i] <= 0 || dim_size[i] > _XMP_world_size)
	    _XMP_fatal("Wrong value in XMP_NODE_SIZE\n");
	}
      }
    }
  }

  _XMP_init_nodes_info(n, dim_size, is_static);

  n->info[0].multiplier = 1;
  for(int i=1;i<dim;i++)
    n->info[i].multiplier = n->info[i-1].multiplier * dim_size[i-1];

  n->subcomm = create_subcomm(n);
  
  return n;
}

_XMP_nodes_t *_XMP_init_nodes_struct_EXEC(int dim, int *dim_size, int is_static)
{
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  int size = exec_nodes->comm_size;

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_dup(*((MPI_Comm *)exec_nodes->comm), comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(_XMP_N_INT_TRUE, dim, size, (_XMP_comm_t *)comm);

  // calc inherit info
  _XMP_nodes_t *inherit_nodes = _XMP_get_execution_nodes();
  n->inherit_nodes = inherit_nodes;
  n->inherit_info  = _XMP_calc_inherit_info(inherit_nodes);
  n->attr          = _XMP_EXECUTING_NODES;

  // calc info
  _XMP_init_nodes_info(n, dim_size, is_static);

  n->info[0].multiplier = 1;
  for(int i=1;i<dim;i++)
    n->info[i].multiplier = n->info[i-1].multiplier * dim_size[i-1];
  
  return n;
}

_XMP_nodes_t *_XMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper,
						  int ref_stride, int *dim_size, int is_static)
{
  _XMP_validate_nodes_ref(&ref_lower, &ref_upper, &ref_stride, _XMP_world_size);
  int is_member = _XMP_check_nodes_ref_inclusion(ref_lower, ref_upper, ref_stride, _XMP_world_size, _XMP_world_rank);

  MPI_Comm *comm = _XMP_alloc(sizeof(MPI_Comm));
  MPI_Comm_split(*((MPI_Comm *)(_XMP_get_execution_nodes())->comm), is_member, _XMP_world_rank, comm);

  _XMP_nodes_t *n = _XMP_create_new_nodes(is_member, dim, _XMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride),
                                          (_XMP_comm_t *)comm);

  // calc inherit info
  int shrink[1] = {_XMP_N_INT_FALSE};
  int l[1] = {ref_lower};
  int u[1] = {ref_upper};
  int s[1] = {ref_stride};
  n->inherit_nodes = _XMP_world_nodes;
  n->inherit_info = _XMP_calc_inherit_info_by_ref(_XMP_world_nodes, shrink, l, u, s);
  n->attr = _XMP_ENTIRE_NODES;

  // calc info
  _XMP_init_nodes_info(n, dim_size, is_static);

  n->info[0].multiplier = 1;
  for(int i=1;i<dim;i++)
    n->info[i].multiplier = n->info[i-1].multiplier * dim_size[i-1];

  n->subcomm = create_subcomm(n);
  
  return n;
}

// Check whether to use a sub communicator or not.
static int check_subcomm(_XMP_nodes_t *n, int *lower, int *upper, int *stride, int *shrink)
{
  if(n->attr != _XMP_ENTIRE_NODES) return _XMP_N_INT_FALSE;

  int ref_dim = n->dim;
  for(int i=0;i<ref_dim;i++){
    if(!shrink[i]){
      int size   = n->info[i].size;
      int length = (upper[i] - lower[i] + 1) / stride[i];
      
      if(length == 1)
	continue;
      else if(length != size)
	return _XMP_N_INT_FALSE;
    }
  }
  
  return _XMP_N_INT_TRUE;
}

// When getting sub-communicator
//  For example, When n->comm is (*,:,:), return n->subcomm[2]
//               When n->comm is (*,*,:), return n->subcomm[3]
//               When n->comm is (*,:,*), return n->subcomm[5]
// "*" is 1, ":" is 0.
// (*,:,:) -> (1,0,0) -> 001(binary digit) -> 2(decimal digit)
// (*,*,:) -> (1,1,0) -> 011(binary digit) -> 3(decimal digit)
// (*,:,*) -> (1,0,1) -> 101(binary digit) -> 5(decimal digit)
static _XMP_comm_t* get_subcomm(_XMP_nodes_t *n, int *lower, int *upper, int *stride, int *shrink)
{
  int dim = n->dim;
  int acc = 0;
  for(int i=0;i<dim;i++){
    if(shrink[i])
      acc += 1<<i;
    else{
      int length = (upper[i] - lower[i] + 1) / stride[i];
      if(length == 1)
	acc += 1<<i;
    }
  }

  MPI_Comm *subcomm = (MPI_Comm *)(n->subcomm);
  return (_XMP_comm_t*)&subcomm[acc];
}

_XMP_nodes_t *_XMP_init_nodes_struct_NODES_NAMED(int dim, _XMP_nodes_t *ref_nodes, int *shrink, int *ref_lower,
						 int *ref_upper, int *ref_stride, int *dim_size, int is_static)
{
  int ref_dim       = ref_nodes->dim;
  int is_ref_member = ref_nodes->is_member;
  int color         = 1;
  int is_member     = _XMP_N_INT_TRUE;
  
  if(is_ref_member){
    int acc_nodes_size = 1;
    for(int i=0;i<ref_dim;i++){
      int size = ref_nodes->info[i].size;
      int rank = ref_nodes->info[i].rank;

      if(shrink[i]){
        color += (acc_nodes_size * rank);
      }
      else{
        _XMP_validate_nodes_ref(&ref_lower[i], &ref_upper[i], &ref_stride[i], size);
        is_member = is_member && _XMP_check_nodes_ref_inclusion(ref_lower[i], ref_upper[i], ref_stride[i], size, rank);
      }

      acc_nodes_size *= size;
    }

    if(!is_member){
      color = 0;
    }
  }
  else{
    is_member = _XMP_N_INT_FALSE;
    color = 0;
  }

  int comm_size = 1;
  for(int i=0;i<ref_dim;i++)
    if(!shrink[i])
      comm_size *= _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);

  MPI_Comm *comm;
  int use_subcomm;  
  if(check_subcomm(ref_nodes, ref_lower, ref_upper, ref_stride, shrink)){
    use_subcomm = _XMP_N_INT_TRUE;
    comm        = (MPI_Comm *)get_subcomm(ref_nodes, ref_lower, ref_upper, ref_stride, shrink);
    //    MPI_Comm_dup(*(MPI_Comm *)get_subcomm(ref_nodes, ref_lower, ref_upper, ref_stride, shrink), comm);
  }
  else if(comm_size == 1){
    use_subcomm = _XMP_N_INT_FALSE;
    comm        = _XMP_alloc(sizeof(MPI_Comm));
    MPI_Comm_dup(MPI_COMM_SELF, comm);
  }
  else{
    use_subcomm = _XMP_N_INT_FALSE;
    comm        = _XMP_alloc(sizeof(MPI_Comm));
    MPI_Comm_split(*((MPI_Comm *)ref_nodes->comm), color, _XMP_world_rank, comm);
  }

  _XMP_nodes_t *n = _XMP_create_new_nodes(is_member, dim, comm_size, (_XMP_comm_t *)comm);

  // calc inherit info
  n->inherit_nodes = ref_nodes;
  n->inherit_info  = _XMP_calc_inherit_info_by_ref(ref_nodes, shrink, ref_lower, ref_upper, ref_stride);
  n->attr          = _XMP_EQUIVALENCE_NODES;
  n->use_subcomm   = use_subcomm;

  // calc info
  _XMP_init_nodes_info(n, dim_size, is_static);

  n->info[0].multiplier = 1;
  for(int i=1;i<dim;i++)
    n->info[i].multiplier = n->info[i-1].multiplier * dim_size[i-1];

  return n;
}

void _XMP_init_nodes_STATIC_GLOBAL(_XMP_nodes_t **nodes, int dim, ...)
{
  int dim_size[dim];

  va_list args;
  va_start(args, dim);
  for(int i=0;i<dim;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0){
      _XMP_fatal("<nodes-size> should be more than zero");
    }
    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_GLOBAL(dim, dim_size, _XMP_N_INT_TRUE);
  *nodes = n;

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_DYNAMIC_GLOBAL(_XMP_nodes_t **nodes, int dim, ...)
{
  int dim_size[dim];
  int *dim_size_p[dim];

  va_list args;
  va_start(args, dim);
  for(int i=0;i<dim;i++){
    dim_size_p[i] = NULL;
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp == -1){
      dim_size_p[i] = va_arg(args, int *);
    }
    else if (dim_size_temp <= 0){
      _XMP_fatal("<nodes-size> should be more than zero");
    }

    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_GLOBAL(dim, dim_size, _XMP_N_INT_FALSE);
  *nodes = n;

  // int *last_dim_size_p = va_arg(args, int *);
  // *last_dim_size_p = n->info[dim - 1].size; 
  for(int i=0;i<dim;i++){
    if(dim_size_p[i])
      *dim_size_p[i] = n->info[i].size;
  }

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_STATIC_EXEC(_XMP_nodes_t **nodes, int dim, ...) {
  int dim_size[dim];

  va_list args;
  va_start(args, dim);
  for (int i=0;i<dim;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0)
      _XMP_fatal("<nodes-size> should be less or equal to zero");

    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_EXEC(dim, dim_size, _XMP_N_INT_TRUE);
  *nodes = n;

  for (int i=0;i<dim;i++) {
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_DYNAMIC_EXEC(_XMP_nodes_t **nodes, int dim, ...) {
  int dim_size[dim - 1];

  va_list args;
  va_start(args, dim);
  for(int i=0;i<dim-1;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0)
      _XMP_fatal("<nodes-size> should be less or equal to zero");

    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_EXEC(dim, dim_size, _XMP_N_INT_FALSE);
  *nodes = n;

  int *last_dim_size_p = va_arg(args, int *);
  *last_dim_size_p = n->info[dim - 1].size;
  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_STATIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                         int ref_lower, int ref_upper, int ref_stride, ...)
{
  int dim_size[dim];

  va_list args;
  va_start(args, ref_stride);
  for(int i=0;i<dim;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0)
      _XMP_fatal("<nodes-size> should be less or equal to zero");

    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n =  _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, 
							 ref_stride, dim_size, _XMP_N_INT_TRUE);
  *nodes = n;

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_DYNAMIC_NODES_NUMBER(_XMP_nodes_t **nodes, int dim,
                                          int ref_lower, int ref_upper, int ref_stride, ...)
{
  int dim_size[dim - 1];

  va_list args;
  va_start(args, ref_stride);
  for(int i=0;i<dim-1;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0)
      _XMP_fatal("<nodes-size> should be less or equal to zero");

    dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n =  _XMP_init_nodes_struct_NODES_NUMBER(dim, ref_lower, ref_upper, 
							 ref_stride, dim_size, _XMP_N_INT_FALSE);
  *nodes = n;

  int *last_dim_size_p = va_arg(args, int *);
  *last_dim_size_p = n->info[dim - 1].size;

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_STATIC_NODES_NAMED(_XMP_nodes_t **nodes, int dim, _XMP_nodes_t *ref_nodes, ...)
{
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim], ref_upper[ref_dim], ref_stride[ref_dim];
  int dim_size[dim];

  va_list args;
  va_start(args, ref_nodes);
  for(int i=0;i<ref_dim;i++){
    shrink[i] = va_arg(args, int);
    if(!shrink[i]){
      ref_lower[i]  = va_arg(args, int);
      ref_upper[i]  = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  for(int i=0;i<dim;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0){
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    }
    else{
      dim_size[i] = dim_size_temp;
    }
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper,
						       ref_stride, dim_size, _XMP_N_INT_TRUE);
  *nodes = n;

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_init_nodes_DYNAMIC_NODES_NAMED(_XMP_nodes_t **nodes, int dim, _XMP_nodes_t *ref_nodes, ...)
{
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim];
  int ref_lower[ref_dim], ref_upper[ref_dim], ref_stride[ref_dim];
  int dim_size[dim-1];

  va_list args;
  va_start(args, ref_nodes);
  for(int i=0;i<ref_dim;i++){
    shrink[i] = va_arg(args, int);
    if(!shrink[i]){
      ref_lower[i]  = va_arg(args, int);
      ref_upper[i]  = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }

  for(int i=0;i<dim-1;i++){
    int dim_size_temp = va_arg(args, int);
    if(dim_size_temp <= 0)
      _XMP_fatal("<nodes-size> should be less or equal to zero");
    else
      dim_size[i] = dim_size_temp;
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(dim, ref_nodes, shrink, ref_lower, ref_upper,
						       ref_stride, dim_size, _XMP_N_INT_FALSE);
  *nodes = n;

  int *last_dim_size_p = va_arg(args, int *);
  *last_dim_size_p = n->info[dim - 1].size;

  for(int i=0;i<dim;i++){
    int *rank_p = va_arg(args, int *);
    *rank_p = n->info[i].rank;
  }
  va_end(args);
}

void _XMP_finalize_nodes(_XMP_nodes_t *nodes)
{
  if(!nodes->use_subcomm)
    _XMP_finalize_comm(nodes->comm);
  _XMP_free(nodes->inherit_info);
  _XMP_free(nodes);
}

int _XMP_exec_task_GLOBAL_PART(_XMP_task_desc_t **task_desc, int ref_lower, int ref_upper, int ref_stride)
{
  int lower[1], upper[1], stride[1];
  
  lower[0] = ref_lower;
  upper[0] = ref_upper;
  stride[0] = ref_stride;

  _XMP_task_desc_t *desc = NULL;
  if(*task_desc == NULL){
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
  }
  else{
    desc = *task_desc;
    if(_XMP_compare_task_exec_cond(desc, _XMP_world_nodes, lower, upper, stride)){
      if(desc->execute){
        _XMP_push_nodes(desc->nodes);
        return _XMP_N_INT_TRUE;
      }
      else {
        return _XMP_N_INT_FALSE;
      }
    }
    else{
      if(desc->nodes != NULL){
        _XMP_finalize_nodes(desc->nodes);
      }
    }
  }

  _XMP_nodes_t *n = NULL;
  _XMP_init_nodes_STATIC_NODES_NUMBER(&n, 1, ref_lower, ref_upper, ref_stride, _XMP_M_COUNT_TRIPLETi(ref_lower, ref_upper, ref_stride));
  _XMP_set_task_desc(desc, n, n->is_member, _XMP_world_nodes, lower, upper, stride);
  
  if(n->is_member){
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  }
  else{
    return _XMP_N_INT_FALSE;
  }
}

int _XMP_exec_task_NODES_ENTIRE(_XMP_task_desc_t **task_desc, _XMP_nodes_t *ref_nodes)
{
  if(ref_nodes->is_member){
    _XMP_push_nodes(ref_nodes);
    return true;
  }
  else{
    return false;
  }
}

int _XMP_exec_task_NODES_ENTIRE_nocomm(_XMP_nodes_t *ref_nodes)
{
  return ref_nodes->is_member;
}

void _XMP_exec_task_NODES_FINALIZE(_XMP_task_desc_t *task_desc)
{
  if(task_desc == NULL) return;
  
  if(xmp_is_async()){
    // Keep a node descriptor. After wait_async directive,
    // the node descriptor is freed.
    _XMP_nodes_dealloc_after_wait_async(task_desc->nodes); 
    _XMP_free(task_desc);
    return;
  }

  _XMP_finalize_nodes(task_desc->nodes);
  _XMP_free(task_desc);
}

int _XMP_exec_task_NODES_PART(_XMP_task_desc_t **task_desc, _XMP_nodes_t *ref_nodes, ...)
{
  va_list args;
  va_start(args, ref_nodes);
  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim], ref_lower[ref_dim], ref_upper[ref_dim], ref_stride[ref_dim];
  
  int acc_dim_size = 1;
  for(int i=0;i<ref_dim;i++){
    shrink[i] = va_arg(args, int);
    if(!shrink[i]){
      ref_lower[i]  = va_arg(args, int);
      ref_upper[i]  = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
      acc_dim_size *= _XMP_M_COUNT_TRIPLETi(ref_lower[i], ref_upper[i], ref_stride[i]);
    }
    else{
      ref_lower[i]  = 0;
      ref_upper[i]  = 0;
      ref_stride[i] = -1;
    }      
  }
  va_end(args);

  _XMP_task_desc_t *desc = NULL;
  if(*task_desc == NULL){
    desc = (_XMP_task_desc_t *)_XMP_alloc(sizeof(_XMP_task_desc_t));
    *task_desc = desc;
    //    printf("communicator created.\n");
  }
  else{
    desc = *task_desc;
    if(_XMP_compare_task_exec_cond(desc, ref_nodes, ref_lower, ref_upper, ref_stride)){
      if(desc->execute){
        _XMP_push_nodes(desc->nodes);
	//	printf("communicator reused.\n");
        return _XMP_N_INT_TRUE;
      }
      else{
        return _XMP_N_INT_FALSE;
      }
    }
    else{
      if(desc->nodes != NULL){
        _XMP_finalize_nodes(desc->nodes);
	//	printf("communicator freed.\n");
      }
    }
  }

  _XMP_nodes_t *n = _XMP_init_nodes_struct_NODES_NAMED(1, ref_nodes, shrink, ref_lower, ref_upper, ref_stride,
                                                       &acc_dim_size, _XMP_N_INT_TRUE);
  _XMP_set_task_desc(desc, n, n->is_member, ref_nodes, ref_lower, ref_upper, ref_stride);
  
  if(n->is_member){
    _XMP_push_nodes(n);
    return _XMP_N_INT_TRUE;
  }
  else{
    return _XMP_N_INT_FALSE;
  }
}

int _XMP_exec_task_NODES_PART_nocomm(_XMP_nodes_t *ref_nodes, ...)
{
  if (!ref_nodes->is_member) return _XMP_N_INT_FALSE;

  int ref_dim = ref_nodes->dim;
  int shrink[ref_dim], ref_lower[ref_dim], ref_upper[ref_dim], ref_stride[ref_dim];

  va_list args;
  va_start(args, ref_nodes);

  for(int i=0;i<ref_dim;i++){
    shrink[i] = va_arg(args, int);
    if(!shrink[i]){
      ref_lower[i]  = va_arg(args, int);
      ref_upper[i]  = va_arg(args, int);
      ref_stride[i] = va_arg(args, int);
    }
  }
  va_end(args);

  for(int i=0;i<ref_dim;i++){
    if(shrink[i]) continue;

    int me = ref_nodes->info[i].rank + 1;

    if (me < ref_lower[i] || ref_upper[i] < me) return _XMP_N_INT_FALSE;
    if ((me-ref_lower[i]) % ref_stride[i] != 0) return _XMP_N_INT_FALSE;
  }

  return _XMP_N_INT_TRUE;
}

// FIXME do not use this function
_XMP_nodes_t *_XMP_create_nodes_by_comm(int is_member, _XMP_comm_t *comm)
{
  int size;
  MPI_Comm_size(*((MPI_Comm *)comm), &size);

  _XMP_nodes_t *n = _XMP_create_new_nodes(is_member, 1, size, comm);

  n->info[0].size = size;
  if(is_member)
    MPI_Comm_rank(*((MPI_Comm *)comm), &(n->info[0].rank));

  // calc inherit info
  n->inherit_nodes      = NULL;
  n->inherit_info       = NULL;
  n->info[0].multiplier = 1;

  return n;
}

void _XMP_calc_rank_array(_XMP_nodes_t *n, int *rank_array, int linear_rank)
{
  int j = linear_rank;
  for(int i=n->dim-1;i>=0;i--){
    rank_array[i] = j / n->info[i].multiplier;
    j = j % n->info[i].multiplier;
  }
}

int _XMP_calc_linear_rank(_XMP_nodes_t *n, int *rank_array)
{
  int acc_rank = 0;
  int acc_nodes_size = 1;
  int nodes_dim = n->dim;
  
  for(int i=0;i<nodes_dim;i++){
    acc_rank += (rank_array[i]) * acc_nodes_size;
    acc_nodes_size *= n->info[i].size;
  }

  return acc_rank;
}

int _XMP_calc_linear_rank_on_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes)
{
  if(_XMP_compare_nodes(n, target_nodes)){
    return _XMP_calc_linear_rank(n, rank_array);
  }
  else{
    _XMP_nodes_t *inherit_nodes = n->inherit_nodes;
    if(inherit_nodes == NULL){
      // FIXME implement
      _XMP_fatal("unsupported case: gmove");
      return _XMP_N_INVALID_RANK; // XXX dummy;
    }
    else{
      int inherit_nodes_dim = inherit_nodes->dim;
      int *new_rank_array = _XMP_alloc(sizeof(int) * inherit_nodes_dim);
      _XMP_nodes_inherit_info_t *inherit_info = n->inherit_info;

      int j = 0;
      for(int i=0;i<inherit_nodes_dim;i++){
        if(inherit_info[i].shrink){
          // FIXME how implement ???
          new_rank_array[i] = 0;
        }
	else{
          new_rank_array[i] = ((inherit_info[i].stride) * rank_array[j]) + (inherit_info[i].lower);
          j++;
        }
      }

      int ret = _XMP_calc_linear_rank_on_target_nodes(inherit_nodes, new_rank_array, target_nodes);
      _XMP_free(new_rank_array);
      return ret;
    }
  }
}


//
// check if n and (target_n or its ancestor) match and calc target_ncoord on target_n
// corresponding to ncoord on n
//
_Bool _XMP_calc_coord_on_target_nodes2(_XMP_nodes_t *n, int *ncoord, 
				       _XMP_nodes_t *target_n, int *target_ncoord)
{
  if(n == target_n){
    //printf("%d, %d\n", n->dim, target_n->dim);
    memcpy(target_ncoord, ncoord, sizeof(int) * n->dim);
    return true;
  }
  else if (n->attr == _XMP_ENTIRE_NODES && target_n->attr == _XMP_ENTIRE_NODES){
    int rank = _XMP_calc_linear_rank(n, ncoord);
    _XMP_calc_rank_array(target_n, target_ncoord, rank);
    //xmp_dbg_printf("ncoord = %d, target_ncoord = %d\n", ncoord[0], target_ncoord[0]);
    return true;
  }

  _XMP_nodes_t *target_p = target_n->inherit_nodes;
  if(target_p){
    int target_pcoord[_XMP_N_MAX_DIM];
    if (_XMP_calc_coord_on_target_nodes2(n, ncoord, target_p, target_pcoord)){
      //int target_prank = _XMP_calc_linear_rank(target_p, target_pcoord);
      /* printf("dim = %d, m0 = %d, m1 = %d\n", */
      /* 	     target_n->dim, target_n->info[0].multiplier, target_n->info[1].multiplier); */
      //_XMP_calc_rank_array(target_n, target_ncoord, target_prank);

      _XMP_nodes_inherit_info_t *inherit_info = target_n->inherit_info;
      int target_rank = 0;
      int multiplier  = 1;

      for(int i=0;i<target_p->dim;i++){
      	if(inherit_info[i].shrink){
	  ;
      	}
      	else{
      	  int target_rank_dim = (target_pcoord[i] - inherit_info[i].lower) / inherit_info[i].stride;
	  target_rank += multiplier * target_rank_dim;
	  multiplier *= inherit_info[i].size;
      	}
      }

      _XMP_calc_rank_array(target_n, target_ncoord, target_rank);

      return true;
    }
  }
  return false;
}
    
//
// check if n and target_n, or their ancestors, match and calc target_ncoord on target_n
// corresponding to ncoord on n
//
_Bool _XMP_calc_coord_on_target_nodes(_XMP_nodes_t *n, int *ncoord, 
				      _XMP_nodes_t *target_n, int *target_ncoord)
{
  if(_XMP_calc_coord_on_target_nodes2(n, ncoord, target_n, target_ncoord))
    return true;

  _XMP_nodes_t *p = n->inherit_nodes;
  if(p){
    int pcoord[_XMP_N_MAX_DIM];
    int rank = _XMP_calc_linear_rank(n, ncoord);
    //_XMP_calc_rank_array(p, pcoord, rank);
     
    _XMP_nodes_inherit_info_t *inherit_info = n->inherit_info;
    int multiplier[_XMP_N_MAX_DIM];

    int dim = p->dim;
    multiplier[0] = 1;
    for(int i=1;i<dim;i++)
      multiplier[i] = multiplier[i-1] * inherit_info[i].size;

    int j = rank;
    for (int i=dim;i>= 0;i--){
      if(inherit_info[i].shrink){
    	pcoord[i] = p->info[i].rank;
      }
      else{
	int rank_dim = j / multiplier[i];
    	pcoord[i] = inherit_info[i].stride * rank_dim + inherit_info[i].lower;
	j = j % multiplier[i];
      }
    }
    
    if(_XMP_calc_coord_on_target_nodes(p, pcoord, target_n, target_ncoord))
      return true;
  }

  return false;
}


/* _XMP_nodes_t *get_common_ancestor_nodes(_XMP_nodes_t *n0, _XMP_nodes_t *n1){ */

/*   if (_XMP_compare_nodes(n0, n1)) return n0; */

/*   _XMP_nodes_t *p1 = n1->inherit_nodes; */
/*   while (p1){ */
/*     if (_XMP_compare_nodes(n0, p1)) return n0; */
/*     p1 = p1->inherit_nodes; */
/*   } */

/*   _XMP_nodes_t *p0 = n0->inherit_nodes; */
/*   if (p0) return get_common_ancestor_nodes(p0, n1); */

/*   return NULL; */

/* } */

_XMP_nodes_ref_t *_XMP_init_nodes_ref(_XMP_nodes_t *n, int *rank_array)
{
  _XMP_nodes_ref_t *nodes_ref = _XMP_alloc(sizeof(_XMP_nodes_ref_t));
  int dim = n->dim;
  int *new_rank_array = _XMP_alloc(sizeof(int) * dim);

  int shrink_nodes_size = 1;
  for(int i=0;i<dim;i++){
    new_rank_array[i] = rank_array[i];
    if(new_rank_array[i] == _XMP_N_UNSPECIFIED_RANK){
      shrink_nodes_size *= (n->info[i].size);
    }
  }

  nodes_ref->nodes = n;
  nodes_ref->ref = new_rank_array;
  nodes_ref->shrink_nodes_size = shrink_nodes_size;

  return nodes_ref;
}

void _XMP_finalize_nodes_ref(_XMP_nodes_ref_t *nodes_ref)
{
  _XMP_free(nodes_ref->ref);
  _XMP_free(nodes_ref);
}

_XMP_nodes_ref_t *_XMP_create_nodes_ref_for_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes)
{
  if(_XMP_compare_nodes(n, target_nodes)){
    return _XMP_init_nodes_ref(n, rank_array);
  }
  else{
    _XMP_nodes_t *inherit_nodes = n->inherit_nodes;
    if(inherit_nodes == NULL){
      // FIXME implement
      _XMP_fatal("unsupported case: gmove");
      return NULL; // XXX dummy;
      // return _XMP_init_nodes_ref(n, rank_array);
    }
    else{
      int inherit_nodes_dim = inherit_nodes->dim;
      int *new_rank_array = _XMP_alloc(sizeof(int) * inherit_nodes_dim);
      _XMP_nodes_inherit_info_t *inherit_info = n->inherit_info;

      int j = 0;
      for(int i = 0; i < inherit_nodes_dim; i++){
        if(inherit_info[i].shrink){
          new_rank_array[i] = _XMP_N_UNSPECIFIED_RANK;
        }
	else{
          new_rank_array[i] = ((inherit_info[i].stride) * rank_array[j]) + (inherit_info[i].lower);
          j++;
        }
      }

      _XMP_nodes_ref_t *ret = _XMP_create_nodes_ref_for_target_nodes(inherit_nodes, new_rank_array, target_nodes);
      _XMP_free(new_rank_array);
      return ret;
    }
  }
}

void _XMP_translate_nodes_rank_array_to_ranks(_XMP_nodes_t *nodes, int *ranks, int *rank_array, int shrink_nodes_size)
{
  int calc_flag = _XMP_N_INT_TRUE;
  int nodes_dim = nodes->dim;
  
  for(int i=0;i<nodes_dim;i++){
    if(rank_array[i] == _XMP_N_UNSPECIFIED_RANK){
      calc_flag = _XMP_N_INT_FALSE;
      int nodes_size = nodes->info[i].size;
      int new_shrink_nodes_size = shrink_nodes_size / nodes_size;
      for(int j=0;j<nodes_size;j++){
        rank_array[i] = j;
        _XMP_translate_nodes_rank_array_to_ranks(nodes, ranks + (j * new_shrink_nodes_size), rank_array, new_shrink_nodes_size);
      }
    }
  }

  if(calc_flag)
    *ranks = _XMP_calc_linear_rank(nodes, rank_array);
}

int _XMP_get_next_rank(_XMP_nodes_t *nodes, int *rank_array)
{
  int i, dim = nodes->dim;
  for(i=0;i<dim;i++){
    int size = nodes->info[i].size;
    (rank_array[i])++;
    if(rank_array[i] == size)
      rank_array[i] = 0;
    else
      break;
  }

  if(i == dim)
    return _XMP_N_INT_FALSE;
  else
   return _XMP_N_INT_TRUE;
}

int _XMP_calc_nodes_index_from_inherit_nodes_index(_XMP_nodes_t *nodes, int inherit_nodes_index)
{
  _XMP_nodes_t *inherit_nodes = nodes->inherit_nodes;
  if(inherit_nodes == NULL)
    _XMP_fatal("inherit nodes is NULL");

  int nodes_index = 0;
  int inherit_nodes_index_count = 0;
  int inherit_nodes_dim = inherit_nodes->dim;
  for(int i=0;i<inherit_nodes_dim;i++,inherit_nodes_index_count++){
    if(inherit_nodes_index_count == inherit_nodes_index){
      return nodes_index;
    }
    else{
      if(!nodes->inherit_info[i].shrink)
        nodes_index++;
    }
  }

  _XMP_fatal("the function does not reach here");
  return 0;
}
