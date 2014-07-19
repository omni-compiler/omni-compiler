/*
 * $TSUKUBA_Release: Omni OpenMP/XcalableMP Compiler Software Version 0.6.0 (alpha) $
 * $TSUKUBA_Copyright:
 *  Copyright (C) 2010-2011 University of Tsukuba, 
 *  	      2012  University of Tsukuba and Riken AICS
 *  
 *  This software is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version
 *  2.1 published by the Free Software Foundation.
 *  
 *  Please check the Copyright and License information in the files named
 *  COPYRIGHT and LICENSE under the top  directory of the Omni Compiler
 *  Software release kit.
 *  
 *  * The specification of XcalableMP has been designed by the XcalableMP
 *    Specification Working Group (http://www.xcalablemp.org/).
 *  
 *  * The development of this software was partially supported by "Seamless and
 *    Highly-productive Parallel Programming Environment for
 *    High-performance computing" project funded by Ministry of Education,
 *    Culture, Sports, Science and Technology, Japan.
 *  $
 */
#include <stdio.h>
#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

void _XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, _XMP_N_DEFAULT_ROOT_RANK, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

void _XMP_bcast_NODES_ENTIRE_GLOBAL(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                    int from_lower, int from_upper, int from_stride) {
  _XMP_RETURN_IF_SINGLE;

  if (!bcast_nodes->is_member) {
    return;
  }

  // check <from-ref>
  if (_XMP_M_COUNT_TRIPLETi(from_lower, from_upper, from_stride) != 1) {
    _XMP_fatal("broadcast failed, multiple source nodes indicated");
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, from_lower, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}

// FIXME read spec
void _XMP_bcast_NODES_ENTIRE_NODES(_XMP_nodes_t *bcast_nodes, void *addr, int count, size_t datatype_size,
                                   _XMP_nodes_t *from_nodes, ...) {
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
  va_list args;
  va_start(args, from_nodes);

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
      if(inherit_info[i].shrink){
	va_arg(args, int);
	continue;
      }

      int size = inherit_info[i].upper - inherit_info[i].lower + 1;
      
      if(size == 0) {  // skip arguments
	va_arg(args, int);   // is_astrisk 
	va_arg(args, int);   // from_lower
	va_arg(args, int);   // from_upper
	va_arg(args, int);   // from_stride
	continue;
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
	root += (acc_nodes_size * (from_lower - inherit_info[i].lower));
      }
      
      acc_nodes_size *= size;
    }
  }

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(datatype_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // bcast
  MPI_Bcast(addr, count, mpi_datatype, root, *((MPI_Comm *)bcast_nodes->comm));

  MPI_Type_free(&mpi_datatype);
}
