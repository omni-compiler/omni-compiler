#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 


#include <stdio.h>
#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void _XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size){
  _XMP_RETURN_IF_SINGLE;

#ifdef _XMPT
  xmpt_tool_data_t *data = NULL;
#endif
    
  if(!bcast_nodes->is_member) return;

#ifdef _XMP_MPI3
  if(xmp_is_async()){
    _XMP_async_comm_t *async = _XMP_get_current_async();

#ifdef _XMPT
    xmpt_async_id_t async_id = async->async_id;
    if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_begin]){
      struct _xmpt_subscript_t from_subsc;
      (*(xmpt_event_bcast_begin_async_t)xmpt_callback[xmpt_event_bcast_begin])(
        addr,
	count * datatype_size,
	bcast_nodes,
	&from_subsc,
	&on_desc,
	&on_subsc,
        async_id,
	data);
    }
#endif

    MPI_Ibcast(addr, count*datatype_size, MPI_BYTE, _XMP_N_DEFAULT_ROOT_RANK,
    	       *((MPI_Comm *)bcast_nodes->comm), &async->reqs[async->nreqs]);
    async->nreqs++;
  }
  else {
#endif

#ifdef _XMPT
    if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_begin]){
      struct _xmpt_subscript_t from_subsc;
      (*(xmpt_event_bcast_begin_t)xmpt_callback[xmpt_event_bcast_begin])(
        addr,
	count * datatype_size,
	bcast_nodes,
	&from_subsc,
	&on_desc,
	&on_subsc,
	data);
    }
#endif

    MPI_Bcast(addr, count*datatype_size, MPI_BYTE, _XMP_N_DEFAULT_ROOT_RANK,
	      *((MPI_Comm *)bcast_nodes->comm));

#ifdef _XMP_MPI3
  }
#endif

#ifdef _XMPT
  if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_end])
    (*(xmpt_event_end_t)xmpt_callback[xmpt_event_bcast_end])(data);
#endif
  
}

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

// FIXME read spec
void _XMP_bcast_NODES_ENTIRE_NODES(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                   _XMP_nodes_t *from_nodes, ...) {
  va_list args;
  va_start(args, from_nodes);

  _XMP_bcast_NODES_ENTIRE_NODES_V(bcast_nodes, addr, count, datatype_size, from_nodes, args);

  va_end(args);
}

void _XMP_bcast_NODES_ENTIRE_NODES_V(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
				     _XMP_nodes_t *from_nodes, va_list args) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  if (!from_nodes->is_member) {
    _XMP_fatal("broadcast failed, cannot find the source node");
  }

  // calc source nodes number
  int root = 0;
  int acc_nodes_size = 1;
  int from_dim = from_nodes->dim;
  int from_lower, from_upper, from_stride;
  _XMP_nodes_inherit_info_t  *inherit_info = bcast_nodes->inherit_info;

#ifdef _XMPT
  xmpt_tool_data_t *data = NULL;
  struct _xmpt_subscript_t from_subsc;
  from_subsc.ndims = from_nodes->dim;;
  from_subsc.omit = 0;
  va_list args2;
  va_copy(args2, args);
  for (int i = 0; i < from_subsc.ndims; i++){
    int rank = from_nodes->info[i].rank;
    if (va_arg(args2, int) == 1) {
      from_subsc.lbound[i] = rank + 1;
      from_subsc.ubound[i] = rank + 1;
      from_subsc.marker[i] = 1;
    }
    else {
      from_subsc.lbound[i] = va_arg(args2, int);
      from_subsc.ubound[i] = va_arg(args2, int);
      from_subsc.marker[i] = va_arg(args2, int);
    }
  }
#endif

  if(inherit_info == NULL){
    for (int i = 0; i < from_dim; i++) {
      int size = from_nodes->info[i].size;
      if(inherit_info != NULL){
	if(inherit_info[i].shrink == true)
	  continue;
	size = inherit_info[i].upper - inherit_info[i].lower + 1;
	if(size == 0) continue;
      }
      int rank = from_nodes->info[i].rank;

      if (va_arg(args, int) == 1) {
	root += (acc_nodes_size * rank);
      }
      else {
	from_lower = va_arg(args, int) - 1;
	from_upper = va_arg(args, int) - 1;
	from_stride = va_arg(args, int);
	
	// check <from-ref>
	if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
	  _XMP_fatal("multiple source nodes indicated in bcast directive");
	}

	root += (acc_nodes_size * (from_lower));
      }
      
      acc_nodes_size *= size;
    }
  }
  else{
    int inherit_node_dim = bcast_nodes->inherit_nodes->dim;

    for (int i = 0; i < inherit_node_dim; i++) {

      if(inherit_info[i].shrink) // skip i
	continue;

      int size = inherit_info[i].upper - inherit_info[i].lower + 1;
      
      if(size == 0) {  // skip arguments
	va_arg(args, int);   // is_astrisk 
	va_arg(args, int);   // from_lower
	va_arg(args, int);   // from_upper
	va_arg(args, int);   // from_stride
	continue;
      }

      int is_astrisk = va_arg(args, int);
      if (is_astrisk == 1){
	int rank = from_nodes->info[i].rank;
	root += (acc_nodes_size * rank);
      }
      else {
	from_lower = va_arg(args, int) - 1;
	from_upper = va_arg(args, int) - 1;
	va_arg(args, int); // skip from_stride

	// check <from-ref> 
	if(from_lower != from_upper)
	  _XMP_fatal("multiple source nodes indicated in bcast directive");

	root += (acc_nodes_size * (from_lower - inherit_info[i].lower));
      }
      
      acc_nodes_size *= size;
    }
  }

#ifdef _XMP_MPI3
  if (xmp_is_async()){
    _XMP_async_comm_t *async = _XMP_get_current_async();

#ifdef _XMPT
    xmpt_async_id_t async_id = async->async_id;
    if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_begin]){
      (*(xmpt_event_bcast_begin_async_t)xmpt_callback[xmpt_event_bcast_begin])(
        addr,
	count * datatype_size,
	from_nodes,
	&from_subsc,
	&on_desc,
	&on_subsc,
        async_id,
	data);
    }
#endif

    MPI_Ibcast(addr, count*datatype_size, MPI_BYTE, root,
    	       *((MPI_Comm *)bcast_nodes->comm), &async->reqs[async->nreqs]);
    async->nreqs++;
  }
  else {
#endif

#ifdef _XMPT
    if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_begin]){
      (*(xmpt_event_bcast_begin_t)xmpt_callback[xmpt_event_bcast_begin])(
        addr,
	count * datatype_size,
	from_nodes,
	&from_subsc,
	&on_desc,
	&on_subsc,
	data);
    }
#endif

    MPI_Bcast(addr, count*datatype_size, MPI_BYTE, root,
	    *((MPI_Comm *)bcast_nodes->comm));

#ifdef _XMP_MPI3
  }
#endif
  
#ifdef _XMPT
  if (xmpt_enabled && xmpt_callback[xmpt_event_bcast_end])
    (*(xmpt_event_end_t)xmpt_callback[xmpt_event_bcast_end])(data);
#endif

  /* // setup type */
  /* MPI_Datatype mpi_datatype; */
  /* MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype); */
  /* MPI_Type_commit(&mpi_datatype); */

  /* // bcast */
  /* MPI_Bcast(addr, count, mpi_datatype, root, *((MPI_Comm *)bcast_nodes->comm)); */

  /* MPI_Type_free(&mpi_datatype); */
}
