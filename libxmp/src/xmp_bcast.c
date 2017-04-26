#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 


#include <stdio.h>
//#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

/* void _XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size){ */
/*   _XMP_RETURN_IF_SINGLE; */

/*   if(!bcast_nodes->is_member) return; */

/* #ifdef _XMP_MPI3 */
/*   if(xmp_is_async()){ */
/*     _XMP_async_comm_t *async = _XMP_get_current_async(); */
/*     MPI_Ibcast(addr, count*datatype_size, MPI_BYTE, _XMP_N_DEFAULT_ROOT_RANK, */
/*     	       *((MPI_Comm *)bcast_nodes->comm), &async->reqs[async->nreqs]); */
/*     async->nreqs++; */
/*   } */
/*   else */
/* #endif */
/*     MPI_Bcast(addr, count*datatype_size, MPI_BYTE, _XMP_N_DEFAULT_ROOT_RANK, */
/* 	      *((MPI_Comm *)bcast_nodes->comm)); */
/* } */

//
// no need for supporting this pattern yet
//

/* void _XMP_bcast_NODES_ENTIRE_GLOBAL(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size, */
/*                                     int from_lower, int from_upper, int from_stride) { */
/*   _XMP_RETURN_IF_SINGLE; */

/*   if (!bcast_nodes->is_member) { */
/*     return; */
/*   } */

/*   // check <from-ref> */
/*   if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) { */
/*     _XMP_fatal("broadcast failed, multiple source nodes indicated"); */
/*   } */

/*   // setup type */
/*   MPI_Datatype mpi_datatype; */
/*   MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype); */
/*   MPI_Type_commit(&mpi_datatype); */

/*   // bcast */
/*   MPI_Bcast(addr, count, mpi_datatype, from_lower, *((MPI_Comm *)bcast_nodes->comm)); */

/*   MPI_Type_free(&mpi_datatype); */
/* } */

/* // FIXME read spec */
/* void _XMP_bcast_NODES_ENTIRE_NODES(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size, */
/* 				   _XMP_nodes_t *from_nodes, ...) { */
/*   va_list args; */
/*   va_start(args, from_nodes); */

/*   _XMP_bcast_NODES_ENTIRE_NODES_V(bcast_nodes, addr, count, datatype_size, from_nodes, args); */

/*   va_end(args); */
/* } */

/* void _XMP_bcast_NODES_ENTIRE_NODES_V(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size, */
/* 				     _XMP_nodes_t *from_nodes, va_list args) { */
/*   _XMP_RETURN_IF_SINGLE; */

/*   if (!bcast_nodes->is_member) { */
/*     return; */
/*   } */

/*   if (!from_nodes->is_member) { */
/*     _XMP_fatal("broadcast failed, cannot find the source node"); */
/*   } */

/*   // calc source nodes number */
/*   int root = 0; */
/*   int acc_nodes_size = 1; */
/*   int from_dim = from_nodes->dim; */
/*   int from_lower, from_upper, from_stride; */
/*   _XMP_nodes_inherit_info_t  *inherit_info = bcast_nodes->inherit_info; */

/*   if(inherit_info == NULL){ */
/*     for (int i = 0; i < from_dim; i++) { */
/*       int size = from_nodes->info[i].size; */
/*       if(inherit_info != NULL){ */
/* 	if(inherit_info[i].shrink == true) */
/* 	  continue; */
/* 	size = inherit_info[i].upper - inherit_info[i].lower + 1; */
/* 	if(size == 0) continue; */
/*       } */
/*       int rank = from_nodes->info[i].rank; */

/*       if (va_arg(args, int) == 1) { */
/* 	root += (acc_nodes_size * rank); */
/*       } */
/*       else { */
/* 	from_lower = va_arg(args, int) - 1; */
/* 	from_upper = va_arg(args, int) - 1; */
/* 	from_stride = va_arg(args, int); */
	
/* 	// check <from-ref> */
/* 	if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) { */
/* 	  _XMP_fatal("multiple source nodes indicated in bcast directive"); */
/* 	} */

/* 	root += (acc_nodes_size * (from_lower)); */
/*       } */
      
/*       acc_nodes_size *= size; */
/*     } */
/*   } */
/*   else{ */
/*     int inherit_node_dim = bcast_nodes->inherit_nodes->dim; */

/*     for (int i = 0; i < inherit_node_dim; i++) { */

/*       if(inherit_info[i].shrink) // skip i */
/* 	continue; */

/*       int size = inherit_info[i].upper - inherit_info[i].lower + 1; */
      
/*       if(size == 0) {  // skip arguments */
/* 	va_arg(args, int);   // is_astrisk  */
/* 	va_arg(args, int);   // from_lower */
/* 	va_arg(args, int);   // from_upper */
/* 	va_arg(args, int);   // from_stride */
/* 	continue; */
/*       } */

/*       int is_astrisk = va_arg(args, int); */
/*       if (is_astrisk == 1){ */
/* 	int rank = from_nodes->info[i].rank; */
/* 	root += (acc_nodes_size * rank); */
/*       } */
/*       else { */
/* 	from_lower = va_arg(args, int) - 1; */
/* 	from_upper = va_arg(args, int) - 1; */
/* 	va_arg(args, int); // skip from_stride */

/* 	// check <from-ref>  */
/* 	if(from_lower != from_upper) */
/* 	  _XMP_fatal("multiple source nodes indicated in bcast directive"); */

/* 	root += (acc_nodes_size * (from_lower - inherit_info[i].lower)); */
/*       } */
      
/*       acc_nodes_size *= size; */
/*     } */
/*   } */

/* #ifdef _XMP_MPI3 */
/*   if(xmp_is_async()){ */
/*     _XMP_async_comm_t *async = _XMP_get_current_async(); */
/*     MPI_Ibcast(addr, count*datatype_size, MPI_BYTE, root, */
/*     	       *((MPI_Comm *)bcast_nodes->comm), &async->reqs[async->nreqs]); */
/*     async->nreqs++; */
/*   } */
/*   else */
/* #endif */
/*   MPI_Bcast(addr, count*datatype_size, MPI_BYTE, root, */
/* 	    *((MPI_Comm *)bcast_nodes->comm)); */

/*   /\* // setup type *\/ */
/*   /\* MPI_Datatype mpi_datatype; *\/ */
/*   /\* MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype); *\/ */
/*   /\* MPI_Type_commit(&mpi_datatype); *\/ */

/*   /\* // bcast *\/ */
/*   /\* MPI_Bcast(addr, count, mpi_datatype, root, *((MPI_Comm *)bcast_nodes->comm)); *\/ */

/*   /\* MPI_Type_free(&mpi_datatype); *\/ */
/* } */


void _XMP_bcast_on_nodes(void *data_addr, int count, int size,
			 _XMP_object_ref_t *from_desc, _XMP_object_ref_t *on_desc){

  _XMP_ASSERT(!on_desc || on_desc->ref_kind == XMP_OBJ_REF_NODES);

  int root = 0;

  _XMP_RETURN_IF_SINGLE;

  // calc source nodes number
  if (from_desc){
    if (from_desc->ref_kind != XMP_OBJ_REF_NODES)
      _XMP_fatal("Type of the From and ON clauses must be the same.");

    _XMP_nodes_t *from = from_desc->n_desc;

    if (!from->is_member) 
      _XMP_fatal("broadcast failed, cannot find the source node");

    if (on_desc){
      if (on_desc->n_desc != from)
      	_XMP_fatal("Node arrays in the FROM and ON clauses must be the same.");

      int acc_nodes_size = 1;

      for (int i = 0; i < from->dim; i++){
	switch (on_desc->subscript_type[i]){
	case SUBSCRIPT_SCALAR:
	  if (from_desc->REF_INDEX[i] != on_desc->REF_INDEX[i]){
	    _XMP_fatal("A subscript of the from-node must be the same "
		       "with that of the on-node that is a scalar.");
	  }
	  break;
	case SUBSCRIPT_ASTERISK:
	  if ((from_desc)->subscript_type[i] != SUBSCRIPT_ASTERISK){
	    _XMP_fatal("A subscript of the from-node must be '*' "
		       "when the corresponding subscript of on-node is '*'");
	  }
	  break;
	case SUBSCRIPT_TRIPLET:
	case SUBSCRIPT_NONE: {
	  int from_idx = from_desc->REF_INDEX[i];
	  int on_lb = on_desc->REF_LBOUND[i];
	  int on_ub = on_desc->REF_UBOUND[i];
	  int on_st = on_desc->REF_STRIDE[i];
	  if (from_idx < on_lb || from_idx > on_ub || (from_idx - on_lb) % on_st != 0){
	    _XMP_fatal("A subscript of the from-node is out of the on-node bound");
	  }
	  root += (acc_nodes_size * ((from_idx - on_lb) / on_st));
	  acc_nodes_size *= _XMP_M_COUNT_TRIPLETi(on_lb, on_ub, on_st);
	  break;
	}
	}
      }
    }
    else {
      int acc_nodes_size = 1;
      for (int i = 0; i < from->dim; i++){
	root += (acc_nodes_size * (from_desc->REF_INDEX[i] - 1));
	acc_nodes_size *= from->info[i].size;
      }
    }
  }

  _XMP_nodes_t *on;
  if (on_desc){
    if (_XMP_is_entire(on_desc)){
      on = on_desc->n_desc;

#ifdef _XMP_MPI3
      if (xmp_is_async()){
	_XMP_async_comm_t *async = _XMP_get_current_async();
	MPI_Ibcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm),
		   &async->reqs[async->nreqs]);
	async->nreqs++;
      }
      else
#endif
	MPI_Bcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm));
    }
    else {
      _XMP_nodes_t *n;
      _XMP_create_task_nodes(&n, on_desc);
      if (_XMP_test_task_on_nodes(n)){
	on = _XMP_get_execution_nodes();

#ifdef _XMP_MPI3
	if (xmp_is_async()){
	  _XMP_async_comm_t *async = _XMP_get_current_async();
	  MPI_Ibcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm),
		     &async->reqs[async->nreqs]);
	  async->nreqs++;
	}
	else
#endif
	  MPI_Bcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm));
      	_XMP_end_task();
      }
      
#ifdef _XMP_MPI3
      if (xmp_is_async())
	_XMP_nodes_dealloc_after_wait_async(n);
      else
#endif
	_XMP_finalize_nodes(n);
    }
  }
  else { // no on desc
    on = from_desc ? from_desc->n_desc : _XMP_get_execution_nodes();

#ifdef _XMP_MPI3
    if (xmp_is_async()){
      _XMP_async_comm_t *async = _XMP_get_current_async();
      MPI_Ibcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm),
		 &async->reqs[async->nreqs]);
      async->nreqs++;
    }
    else
#endif
      MPI_Bcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on->comm));
  }
}


void _XMP_bcast_on_template(void *data_addr, int count, int size,
			    _XMP_object_ref_t *from_desc, _XMP_object_ref_t *on_desc){

  _XMP_ASSERT(on_desc || on_desc->ref_kind == XMP_OBJ_REF_TEMPL);

  _XMP_RETURN_IF_SINGLE;

  _XMP_template_t *from;
  _XMP_template_t *on = on_desc->t_desc;

  int from_idx_in_nodes[_XMP_N_MAX_DIM];
  for (int i = 0; i < _XMP_N_MAX_DIM; i++) from_idx_in_nodes[i] = 0;

  // calc source nodes number
  if (from_desc){

    if (from_desc->ref_kind != XMP_OBJ_REF_TEMPL)
      _XMP_fatal("Type of the From and ON clauses must be the same.");

    from = from_desc->t_desc;

    if (!from->is_owner)
      _XMP_fatal("broadcast failed, cannot find the source node");

    if (on != from)
      _XMP_fatal("Templates in the FROM and ON clauses must be the same.");

    for (int i = 0; i < from->dim; i++){
      int from_idx = -1;
      switch (on_desc->subscript_type[i]){
      case SUBSCRIPT_SCALAR:
	if (from_desc->REF_INDEX[i] != on_desc->REF_INDEX[i]){
	  _XMP_fatal("A subscript of the from-template must be the same "
		     "with that of the on-template that is a scalar.");
	}
	from_idx = from_desc->REF_INDEX[i];
	break;

      case SUBSCRIPT_ASTERISK:
	if (from_desc->subscript_type[i] != SUBSCRIPT_ASTERISK){
	  _XMP_fatal("A subscript of the from-template must be '*' "
		     "when the corresponding subscript of on-template is '*'");
	}
	from_idx = from->chunk[i].par_lower;
	break;

      case SUBSCRIPT_TRIPLET:
      case SUBSCRIPT_NONE: {
	from_idx = from_desc->REF_INDEX[i];
	int on_lb = on_desc->REF_LBOUND[i];
	int on_ub = on_desc->REF_UBOUND[i];
	int on_st = on_desc->REF_STRIDE[i];
	if (from_idx < on_lb || from_idx > on_ub || (from_idx - on_lb) % on_st != 0){
	  _XMP_fatal("A subscript of the from-template is out of the on-template bound");
	}
	break;
      }
      }

      int j = from->chunk[i].onto_nodes_index;
      if (j != _XMP_N_NO_ONTO_NODES){ // 0-origin
	from_idx_in_nodes[j] = _XMP_calc_template_owner_SCALAR(from, i, (long long)from_idx);
      }
    }
  }

  int root = 0;
  _XMP_nodes_t *on_nodes;

  if (_XMP_is_entire(on_desc)){
    on_nodes = on->onto_nodes;

    if (from_desc){
      int acc_nodes_size = 1;
      for (int i = 0; i < on_nodes->dim; i++){
	root += (acc_nodes_size * from_idx_in_nodes[i]);
	acc_nodes_size *= on_nodes->info[i].size;
      }
    }

#ifdef _XMP_MPI3
    if (xmp_is_async()){
      _XMP_async_comm_t *async = _XMP_get_current_async();
      MPI_Ibcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on_nodes->comm), async->reqs);
      async->nreqs = 1;
    }
    else
#endif
      MPI_Bcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on_nodes->comm));
  }
  else {
    _XMP_nodes_t *n;
    _XMP_create_task_nodes(&n, on_desc);
    if (_XMP_test_task_on_nodes(n)){
      on_nodes = _XMP_get_execution_nodes();
      
      if (from_desc){
	int acc_nodes_size = 1;
	for (int i = 0; i < on_nodes->inherit_nodes->dim; i++){
	  if (on_nodes->inherit_info[i].shrink) continue;
	  int inherit_lb = on_nodes->inherit_info[i].lower;
	  int inherit_ub = on_nodes->inherit_info[i].upper;
	  int inherit_st = on_nodes->inherit_info[i].stride;
	  root += (acc_nodes_size * ((from_idx_in_nodes[i] - inherit_lb) / inherit_st));
	  acc_nodes_size *= _XMP_M_COUNT_TRIPLETi(inherit_lb, inherit_ub, inherit_st);
	}
      }

#ifdef _XMP_MPI3
      if (xmp_is_async()){
	_XMP_async_comm_t *async = _XMP_get_current_async();
	MPI_Ibcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on_nodes->comm), async->reqs);
	async->nreqs = 1;
      }
      else
#endif
	MPI_Bcast(data_addr, count*size, MPI_BYTE, root, *((MPI_Comm *)on_nodes->comm));

      _XMP_end_task();
    }
#ifdef _XMP_MPI3
    if (xmp_is_async())
      _XMP_nodes_dealloc_after_wait_async(n);
    else
#endif
      _XMP_finalize_nodes(n);
  }
}


void _XMP_bcast(void *data_addr, int count, int size,
		_XMP_object_ref_t *from_desc, _XMP_object_ref_t *on_desc)
{
  if (on_desc && on_desc->ref_kind == XMP_OBJ_REF_TEMPL)
    _XMP_bcast_on_template(data_addr, count, size, from_desc, on_desc);
  else
    _XMP_bcast_on_nodes(data_addr, count, size, from_desc, on_desc);
}
