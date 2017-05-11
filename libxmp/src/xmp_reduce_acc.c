#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "xmp_internal.h"

extern void _XMP_reduction(void *data_addr, int count, int datatype, int op,
			   _XMP_object_ref_t *r_desc, int num_locs);
extern void *_XMP_reduction_loc_vars[_XMP_N_MAX_LOC_VAR];
extern int _XMP_reduction_loc_types[_XMP_N_MAX_LOC_VAR];
extern void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
extern void _XMP_reduce_gpu_CLAUSE(void *data_addr, int count, int datatype, int op);

static char comm_mode = -1;

static void set_comm_mode()
{
  if(comm_mode < 0){
    char *mode_str = getenv("XACC_COMM_MODE");
    if(mode_str !=  NULL){
      comm_mode = atoi(mode_str);
    }else{
      comm_mode = 0;
    }
  }
}

#ifdef _XMP_TCA
static char enable_hybrid = -1;

static void set_hybrid_comm()
{
  if (enable_hybrid < 0) {
    char *flag = getenv("XACC_ENABLE_HYBRID");
    if (flag != NULL) {
      enable_hybrid = atoi(flag);
    } else {
      enable_hybrid = 0;
    }
  }
}
#endif

void _XMP_reduce_acc_NODES_ENTIRE(_XMP_nodes_t *nodes, void *data_addr, int count, int datatype, int op)
{
  set_comm_mode();

  if(0){
    //
  }else{
#ifdef _XMP_TCA
    set_hybrid_comm();
    if (enable_hybrid) {
      _XMP_reduce_hybrid_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
    } else {
      _XMP_reduce_tca_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
    }
#else
    _XMP_reduce_gpu_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
#endif
  }
}

void _XMP_reduce_acc_FLMM_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op, int num_locs, ...)
{
  _XMP_fatal("_XMP_reduce_acc_FLMM_NODES_ENTIRE is unimplemented");
}

void _XMP_reduce_acc_CLAUSE(void *data_addr, int count, int datatype, int op)
{
  set_comm_mode();

  if(0){
    //
  }else{
#ifdef _XMP_TCA
    //
#else
    _XMP_reduce_gpu_CLAUSE(data_addr, count, datatype, op);
#endif

  }
}

void _XMP_reduce_acc_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...)
{
  _XMP_fatal("_XMP_reduce_acc_FLMM_CLAUSE is unimplemented");
}


void _XMP_reduction_acc(void *data_addr, int count, int datatype, int op,
			_XMP_object_ref_t *r_desc, int num_locs)
{
  set_comm_mode();

  if(num_locs > 0){
    _XMP_fatal("XACC doesn't support firstmax, firstmin, lastmax, and lastmin currently\n");
  }

  if(comm_mode >= 1){
    _XMP_reduction(data_addr, count, datatype, op, r_desc, num_locs);
    return;
  }

  _XMP_nodes_t *nodes;

  if (r_desc){
    if (_XMP_is_entire(r_desc)){
      if (r_desc->ref_kind == XMP_OBJ_REF_NODES){
	nodes = r_desc->n_desc;
      }
      else {
	nodes = r_desc->t_desc->onto_nodes;
      }
      if (num_locs == 0) _XMP_reduce_acc_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
      else _XMP_reduce_acc_FLMM_NODES_ENTIRE(nodes, data_addr, count, datatype, op, num_locs,
					     _XMP_reduction_loc_vars, _XMP_reduction_loc_types);
    }
    else {
      _XMP_nodes_t *n;
      _XMP_create_task_nodes(&n, r_desc);
      if (_XMP_test_task_on_nodes(n)){
      	nodes = _XMP_get_execution_nodes();
      	if (num_locs == 0) _XMP_reduce_acc_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
      	else _XMP_reduce_acc_FLMM_NODES_ENTIRE(nodes, data_addr, count, datatype, op, num_locs,
					       _XMP_reduction_loc_vars, _XMP_reduction_loc_types);
      	_XMP_end_task();
      }
      _XMP_finalize_nodes(n);
    }

  }
  else {
    nodes = _XMP_get_execution_nodes();
    if (num_locs == 0) _XMP_reduce_acc_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
    else _XMP_reduce_acc_FLMM_NODES_ENTIRE(nodes, data_addr, count, datatype, op, num_locs,
					   _XMP_reduction_loc_vars, _XMP_reduction_loc_types);
  }
}
