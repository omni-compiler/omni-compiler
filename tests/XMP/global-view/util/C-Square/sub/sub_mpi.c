#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int sub_mpi(MPI_Comm *comm) {

  int rank, irank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 2){
    MPI_Comm_rank(*comm, &irank);
    if(irank == 1){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR rank=%d\n",irank);
      exit(1);
    }
  }

  return 0;
}
