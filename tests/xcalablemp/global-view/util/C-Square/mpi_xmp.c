#include "mpi.h"
#include "xmp.h"
extern int ixmp_sub();

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  xmp_init(MPI_COMM_WORLD);

  ixmp_sub();

  xmp_finalize();
  MPI_Finalize();
  return 0;
}
