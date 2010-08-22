#include <string.h>
#include <stdarg.h>
#include "xmp_constant.h"
#include "xmp_internal.h"

static void _XCALABLEMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype);
static void _XCALABLEMP_setup_reduce_op(MPI_Op *mpi_op, int op);
static void _XCALABLEMP_setup_reduce_FLMM_op(MPI_Op *mpi_op, int op);

static void _XCALABLEMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype) {
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
}

static void _XCALABLEMP_setup_reduce_op(MPI_Op *mpi_op, int op) {
  switch (op) {
    case _XCALABLEMP_N_REDUCE_SUM:
      { *mpi_op = MPI_SUM;	break; }
    case _XCALABLEMP_N_REDUCE_PROD:
      { *mpi_op = MPI_PROD;	break; }
    case _XCALABLEMP_N_REDUCE_BAND:
      { *mpi_op = MPI_BAND;	break; }
    case _XCALABLEMP_N_REDUCE_LAND:
      { *mpi_op = MPI_LAND;	break; }
    case _XCALABLEMP_N_REDUCE_BOR:
      { *mpi_op = MPI_BOR;	break; }
    case _XCALABLEMP_N_REDUCE_LOR:
      { *mpi_op = MPI_LOR;	break; }
    case _XCALABLEMP_N_REDUCE_BXOR:
      { *mpi_op = MPI_BXOR;	break; }
    case _XCALABLEMP_N_REDUCE_LXOR:
      { *mpi_op = MPI_LXOR;	break; }
    case _XCALABLEMP_N_REDUCE_MAX:
      { *mpi_op = MPI_MAX;	break; }
    case _XCALABLEMP_N_REDUCE_MIN:
      { *mpi_op = MPI_MIN;	break; }
    case _XCALABLEMP_N_REDUCE_FIRSTMAX:
      { *mpi_op = MPI_MAX;	break; }
    case _XCALABLEMP_N_REDUCE_FIRSTMIN:
      { *mpi_op = MPI_MIN;	break; }
    case _XCALABLEMP_N_REDUCE_LASTMAX:
      { *mpi_op = MPI_MAX;	break; }
    case _XCALABLEMP_N_REDUCE_LASTMIN:
      { *mpi_op = MPI_MIN;	break; }
    default:
      _XCALABLEMP_fatal("unknown reduce operation");
  }
}

static void _XCALABLEMP_setup_reduce_FLMM_op(MPI_Op *mpi_op, int op) {
  switch (op) {
    case _XCALABLEMP_N_REDUCE_FIRSTMAX:
    case _XCALABLEMP_N_REDUCE_FIRSTMIN:
      { *mpi_op = MPI_MIN;      break; }
    case _XCALABLEMP_N_REDUCE_LASTMAX:
    case _XCALABLEMP_N_REDUCE_LASTMIN:
      { *mpi_op = MPI_MAX;      break; }
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
  _XCALABLEMP_setup_reduce_type(&mpi_datatype, &datatype_size, datatype);
  _XCALABLEMP_setup_reduce_op(&mpi_op, op);

  // reduce
  size_t n = datatype_size * count;
  void *temp_buffer = _XCALABLEMP_alloc(n);
  memcpy(temp_buffer, addr, n);

  MPI_Allreduce(temp_buffer, addr, count, mpi_datatype, mpi_op, *(nodes->comm));

  _XCALABLEMP_free(temp_buffer);
}

// #define _XCALABLEMP_reduce_EXEC(addr, count, datatype, op) \
// _XCALABLEMP_reduce_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op)

void _XCALABLEMP_reduce_FLMM_NODES_ENTIRE(_XCALABLEMP_nodes_t *nodes,
                                          void *addr, int count, int datatype, int op,
                                          int num_locs, ...) {
  if (nodes == NULL) return;

  // setup information
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  MPI_Op mpi_op;
  _XCALABLEMP_setup_reduce_type(&mpi_datatype, &datatype_size, datatype);
  _XCALABLEMP_setup_reduce_op(&mpi_op, op);

  // reduce
  size_t n = datatype_size * count;
  void *temp_buffer = _XCALABLEMP_alloc(n);
  memcpy(temp_buffer, addr, n);

  MPI_Allreduce(temp_buffer, addr, count, mpi_datatype, mpi_op, *(nodes->comm));

  va_list args;
  va_start(args, num_locs);
  for (int i = 0; i < num_locs; i++) {
    void *loc = va_arg(args, void *);
    int loc_datatype = va_arg(args, int);
  }
  va_end(args);

  _XCALABLEMP_free(temp_buffer);
}

// #define _XCALABLEMP_reduce_FLMM_EXEC(addr, count, datatype, op, num_locs, ...) \
// _XCALABLEMP_reduce_FLMM_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op, num_locs, __VA_ARGS__)
