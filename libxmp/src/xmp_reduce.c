#include <string.h>
#include "xmp_constant.h"
#include "xmp_internal.h"

static void _XCALABLEMP_setup_reduce_info(MPI_Datatype *mpi_datatype, size_t *datatype_size, MPI_Op *mpi_op,
                                          int datatype, int op);

static void _XCALABLEMP_setup_reduce_info(MPI_Datatype *mpi_datatype, size_t *datatype_size, MPI_Op *mpi_op,
                                          int datatype, int op) {
    // data type
    switch (datatype) {
//    case _XCALABLEMP_N_TYPE_BOOL:
//      { *datatype_size = sizeof(_Bool);			*mpi_datatype = MPI_C_BOOL;			break; }
    case _XCALABLEMP_N_TYPE_CHAR:
      { *datatype_size = sizeof(char);				*mpi_datatype = MPI_SIGNED_CHAR;		break; }
    case _XCALABLEMP_N_TYPE_UNSIGNED_CHAR:
      { *datatype_size = sizeof(unsigned char);			*mpi_datatype = MPI_UNSIGNED_CHAR;		break; }
    case _XCALABLEMP_N_TYPE_SHORT:
      { *datatype_size = sizeof(short);				*mpi_datatype = MPI_SHORT;			break; }
    case _XCALABLEMP_N_TYPE_UNSIGNED_SHORT:
      { *datatype_size = sizeof(unsigned short);		*mpi_datatype = MPI_UNSIGNED_SHORT;		break; }
    case _XCALABLEMP_N_TYPE_INT:
      { *datatype_size = sizeof(int);				*mpi_datatype = MPI_INT;			break; }
    case _XCALABLEMP_N_TYPE_UNSIGNED_INT:
      { *datatype_size = sizeof(unsigned int);			*mpi_datatype = MPI_UNSIGNED;			break; }
    case _XCALABLEMP_N_TYPE_LONG:
      { *datatype_size = sizeof(long);				*mpi_datatype = MPI_LONG;			break; }
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONG:
      { *datatype_size = sizeof(unsigned long);			*mpi_datatype = MPI_UNSIGNED_LONG;		break; }
    case _XCALABLEMP_N_TYPE_LONGLONG:
      { *datatype_size = sizeof(long long);			*mpi_datatype = MPI_LONG_LONG;			break; }
    case _XCALABLEMP_N_TYPE_UNSIGNED_LONGLONG:
      { *datatype_size = sizeof(unsigned long long);		*mpi_datatype = MPI_UNSIGNED_LONG_LONG;		break; }
    case _XCALABLEMP_N_TYPE_FLOAT:
      { *datatype_size = sizeof(float);				*mpi_datatype = MPI_FLOAT;			break; }
    case _XCALABLEMP_N_TYPE_DOUBLE:
      { *datatype_size = sizeof(double);			*mpi_datatype = MPI_DOUBLE;			break; }
    case _XCALABLEMP_N_TYPE_LONG_DOUBLE:
      { *datatype_size = sizeof(long double);			*mpi_datatype = MPI_LONG_DOUBLE;		break; }
//    case _XCALABLEMP_N_TYPE_FLOAT_IMAGINARY:
//      { *datatype_size = sizeof(float _Imaginary);		*mpi_datatype = MPI_FLOAT;			break; }
//    case _XCALABLEMP_N_TYPE_DOUBLE_IMAGINARY:
//      { *datatype_size = sizeof(double _Imaginary);		*mpi_datatype = MPI_DOUBLE;			break; }
//    case _XCALABLEMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
//      { *datatype_size = sizeof(long double _Imaginary);	*mpi_datatype = MPI_LONG_DOUBLE;		break; }
//    case _XCALABLEMP_N_TYPE_FLOAT_COMPLEX:
//      { *datatype_size = sizeof(float _Complex);		*mpi_datatype = MPI_C_FLOAT_COMPLEX;		break; }
//    case _XCALABLEMP_N_TYPE_DOUBLE_COMPLEX:
//      { *datatype_size = sizeof(double _Complex);		*mpi_datatype = MPI_C_DOUBLE_COMPLEX;		break; }
//    case _XCALABLEMP_N_TYPE_LONG_DOUBLE_COMPLEX:
//      { *datatype_size = sizeof(long double _Complex);	*mpi_datatype = MPI_C_LONG_DOUBLE_COMPLEX;	break; }
    default:
      _XCALABLEMP_fatal("unknown data type for reduction");
  }

  // operation
  switch (op) {
    case _XCALABLEMP_N_REDUCE_SUM:
      { *mpi_op = MPI_SUM;	break; }
    case _XCALABLEMP_N_REDUCE_PROD:
      { *mpi_op = MPI_PROD;	break; }
    default:
      _XCALABLEMP_fatal("unknown reduce operation");
  }
}

void _XCALABLEMP_reduce_NODES_ENTIRE(_XCALABLEMP_nodes_t *nodes, void *addr, int count, int datatype, int op) {
  if (nodes == NULL) return;

  // setup information
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  MPI_Op mpi_op;
  _XCALABLEMP_setup_reduce_info(&mpi_datatype, &datatype_size, &mpi_op, datatype, op);

  // reduce
  size_t n = datatype_size * count;
  void *temp_buffer = _XCALABLEMP_alloc(n);
  memcpy(temp_buffer, addr, n);

  MPI_Allreduce(temp_buffer, addr, count, mpi_datatype, mpi_op, *(nodes->comm));

  _XCALABLEMP_free(temp_buffer);
}

void _XCALABLEMP_reduce_EXEC(void *addr, int count, int datatype, int op) {
  // setup information
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  MPI_Op mpi_op;
  _XCALABLEMP_setup_reduce_info(&mpi_datatype, &datatype_size, &mpi_op, datatype, op);

  // reduce
  size_t n = datatype_size * count;
  void *temp_buffer = _XCALABLEMP_alloc(n);
  memcpy(temp_buffer, addr, n);

  _XCALABLEMP_nodes_t *nodes = _XCALABLEMP_get_execution_nodes();
  MPI_Allreduce(temp_buffer, addr, count, mpi_datatype, mpi_op, *(nodes->comm));

  _XCALABLEMP_free(temp_buffer);
}
