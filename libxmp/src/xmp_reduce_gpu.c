#include <include/cuda_runtime.h>
#include <xmp_internal.h>

void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
void _XMP_reduce_gpu_FLMM_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op, int num_locs, ...);
void _XMP_reduce_gpu_CLAUSE(void *data_addr, int count, int datatype, int op);
void _XMP_reduce_gpu_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...);

//copy from xmp_reduce.c
static void _XMP_setup_reduce_type(MPI_Datatype *mpi_datatype, size_t *datatype_size, int datatype) {
  switch (datatype) {
//  case _XMP_N_TYPE_BOOL:
//    { *mpi_datatype = MPI_C_BOOL;			*datatype_size = sizeof(_Bool); 			break; }
    case _XMP_N_TYPE_CHAR:
      { *mpi_datatype = MPI_SIGNED_CHAR;		*datatype_size = sizeof(char); 				break; }
    case _XMP_N_TYPE_UNSIGNED_CHAR:
      { *mpi_datatype = MPI_UNSIGNED_CHAR;		*datatype_size = sizeof(unsigned char); 		break; }
    case _XMP_N_TYPE_SHORT:
      { *mpi_datatype = MPI_SHORT;			*datatype_size = sizeof(short); 			break; }
    case _XMP_N_TYPE_UNSIGNED_SHORT:
      { *mpi_datatype = MPI_UNSIGNED_SHORT;		*datatype_size = sizeof(unsigned short); 		break; }
    case _XMP_N_TYPE_INT:
      { *mpi_datatype = MPI_INT;			*datatype_size = sizeof(int); 				break; }
    case _XMP_N_TYPE_UNSIGNED_INT:
      { *mpi_datatype = MPI_UNSIGNED;			*datatype_size = sizeof(unsigned int); 			break; }
    case _XMP_N_TYPE_LONG:
      { *mpi_datatype = MPI_LONG;			*datatype_size = sizeof(long); 				break; }
    case _XMP_N_TYPE_UNSIGNED_LONG:
      { *mpi_datatype = MPI_UNSIGNED_LONG;		*datatype_size = sizeof(unsigned long); 		break; }
    case _XMP_N_TYPE_LONGLONG:
      { *mpi_datatype = MPI_LONG_LONG;			*datatype_size = sizeof(long long); 			break; }
    case _XMP_N_TYPE_UNSIGNED_LONGLONG:
      { *mpi_datatype = MPI_UNSIGNED_LONG_LONG;		*datatype_size = sizeof(unsigned long long); 		break; }
    case _XMP_N_TYPE_FLOAT:
      { *mpi_datatype = MPI_FLOAT;			*datatype_size = sizeof(float); 			break; }
    case _XMP_N_TYPE_DOUBLE:
      { *mpi_datatype = MPI_DOUBLE;			*datatype_size = sizeof(double); 			break; }
    case _XMP_N_TYPE_LONG_DOUBLE:
      { *mpi_datatype = MPI_LONG_DOUBLE;		*datatype_size = sizeof(long double); 			break; }
//  case _XMP_N_TYPE_FLOAT_IMAGINARY:
//    { *mpi_datatype = MPI_FLOAT;			*datatype_size = sizeof(float _Imaginary); 		break; }
//  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
//    { *mpi_datatype = MPI_DOUBLE;			*datatype_size = sizeof(double _Imaginary); 		break; }
//  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
//    { *mpi_datatype = MPI_LONG_DOUBLE;		*datatype_size = sizeof(long double _Imaginary);	break; }
//  case _XMP_N_TYPE_FLOAT_COMPLEX:
//    { *mpi_datatype = MPI_C_FLOAT_COMPLEX;		*datatype_size = sizeof(float _Complex); 		break; }
//  case _XMP_N_TYPE_DOUBLE_COMPLEX:
//    { *mpi_datatype = MPI_C_DOUBLE_COMPLEX;		*datatype_size = sizeof(double _Complex); 		break; }
//  case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
//    { *mpi_datatype = MPI_C_LONG_DOUBLE_COMPLEX;	*datatype_size = sizeof(long double _Complex); 		break; }
    default:
      _XMP_fatal("unknown data type for reduction");
  }
}

static void _XMP_setup_reduce_op(MPI_Op *mpi_op, int op) {
  switch (op) {
    case _XMP_N_REDUCE_SUM:
      *mpi_op = MPI_SUM;
      break;
    case _XMP_N_REDUCE_PROD:
      *mpi_op = MPI_PROD;
      break;
    case _XMP_N_REDUCE_BAND:
      *mpi_op = MPI_BAND;
      break;
    case _XMP_N_REDUCE_LAND:
      *mpi_op = MPI_LAND;
      break;
    case _XMP_N_REDUCE_BOR:
      *mpi_op = MPI_BOR;
      break;
    case _XMP_N_REDUCE_LOR:
      *mpi_op = MPI_LOR;
      break;
    case _XMP_N_REDUCE_BXOR:
      *mpi_op = MPI_BXOR;
      break;
    case _XMP_N_REDUCE_LXOR:
      *mpi_op = MPI_LXOR;
      break;
    case _XMP_N_REDUCE_MAX:
      *mpi_op = MPI_MAX;
      break;
    case _XMP_N_REDUCE_MIN:
      *mpi_op = MPI_MIN;
      break;
    case _XMP_N_REDUCE_FIRSTMAX:
      *mpi_op = MPI_MAX;
      break;
    case _XMP_N_REDUCE_FIRSTMIN:
      *mpi_op = MPI_MIN;
      break;
    case _XMP_N_REDUCE_LASTMAX:
      *mpi_op = MPI_MAX;
      break;
    case _XMP_N_REDUCE_LASTMIN:
      *mpi_op = MPI_MIN;
      break;
    case _XMP_N_REDUCE_EQV:
    case _XMP_N_REDUCE_NEQV:
    case _XMP_N_REDUCE_MINUS:
      _XMP_fatal("unsupported reduce operation");
    default:
      _XMP_fatal("unknown reduce operation");
  }
}
//end of copy

void cudaErrorCheck(cudaError_t e)
{
 if(e != cudaSuccess){
   _XMP_fatal((char*)cudaGetErrorString(e));
  }
}

void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *dev_addr, int count, int datatype, int op)
{
  if (count == 0) {
    return; // FIXME not good implementation
  }
  if (!nodes->is_member) {
    return;
  }

  // setup information
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  MPI_Op mpi_op;
  _XMP_setup_reduce_type(&mpi_datatype, &datatype_size, datatype);
  _XMP_setup_reduce_op(&mpi_op, op);

  size_t size = datatype_size * count;
  void *host_buf = _XMP_alloc(size);
  cudaError_t e;

  // copy dev to host
  e = cudaMemcpy(host_buf, dev_addr, size, cudaMemcpyDeviceToHost);
  cudaErrorCheck(e);

  MPI_Allreduce(MPI_IN_PLACE, host_buf, count, mpi_datatype, mpi_op, *((MPI_Comm *)nodes->comm));

  // copy host to dev
  e = cudaMemcpy(dev_addr, host_buf, size, cudaMemcpyHostToDevice);
  cudaErrorCheck(e);

  _XMP_free(host_buf);
}
  
void _XMP_reduce_gpu_CLAUSE(void *dev_addr, int count, int datatype, int op) {
  // setup information
  MPI_Datatype mpi_datatype;
  size_t datatype_size;
  MPI_Op mpi_op;
  _XMP_setup_reduce_type(&mpi_datatype, &datatype_size, datatype);
  _XMP_setup_reduce_op(&mpi_op, op);

  size_t size = datatype_size * count;
  void *host_buf = _XMP_alloc(size);
  cudaError_t e;

  // copy dev to host
  e = cudaMemcpy(host_buf, dev_addr, size, cudaMemcpyDeviceToHost);
  cudaErrorCheck(e);

  // reduce
  MPI_Allreduce(MPI_IN_PLACE, host_buf, count, mpi_datatype, mpi_op, *((MPI_Comm *)(_XMP_get_execution_nodes())->comm));

  // copy host to dev
  e = cudaMemcpy(dev_addr, host_buf, size, cudaMemcpyHostToDevice);
  cudaErrorCheck(e);

  _XMP_free(host_buf);
}
