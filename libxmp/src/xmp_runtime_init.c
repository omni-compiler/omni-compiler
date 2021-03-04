#include "xmp_internal.h"

extern void xmpc_traverse_init();
extern void xmpc_traverse_finalize();

extern int _XMP_runtime_working;

extern void _XMP_init_no_traverse(int argc, char** argv, MPI_Comm comm);

void _XMP_init(int argc, char** argv, MPI_Comm comm)
{
  int do_traverse = !_XMP_runtime_working;

  _XMP_init_no_traverse(argc,argv,comm);

  if (do_traverse) xmpc_traverse_init();
}

void xmp_init_all(int argc, char* argv[])
{
  _XMP_init(argc, argv, MPI_COMM_WORLD);
}

extern void _XMP_finalize_no_traverse(bool isFinalize);

void _XMP_finalize(bool isFinalize)
{
  if (_XMP_runtime_working) xmpc_traverse_finalize();
  _XMP_finalize_no_traverse(isFinalize);
}

void xmp_finalize_all()
{
  _XMP_finalize(true);
}

void xmp_init_mpi(int *argc, char ***argv) {}
void xmp_finalize_mpi(void) {}

void xmp_init_py(MPI_Fint comm) {
  _XMP_init(1, NULL, MPI_Comm_f2c(comm));
}

void xmp_init(MPI_Comm comm)
{
  _XMP_init(1, NULL, comm);
}

void xmp_finalize()
{
  _XMP_finalize(false);
}
