#ifndef _MPINPB_H
#define _MPINPB_H

#include <mpi.h>
/*
struct {
  int me,nprocs,root,dp_type;
} mpistuff_;
*/
int me,nprocs,root;
MPI_Datatype dp_type;

#endif
