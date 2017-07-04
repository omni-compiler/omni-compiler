#include <stdio.h>
#include "mpi.h"
#include "xmp.h"

extern int sub_mpi(MPI_Comm *comm);

#pragma xmp nodes p(4)

int main(int argc, char **argv) {

  xmp_init_mpi(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#pragma xmp task on p(2:3)
{
  MPI_Comm comm;
  comm = xmp_get_mpi_comm();
  sub_mpi(&comm);
}

  xmp_finalize_mpi();

  return 0;
}
