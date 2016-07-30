#include <cuda_runtime.h>
#include <xmp_internal.h>
#include "tca-api.h"
#include <math.h>

#define _XMP_TCA_SYNC_MARK_SIZE sizeof(unsigned long)
#define _XMP_TCA_CACHE_ALIGNED_STRIDE 64
#define _XMP_TCA_PIO_SYNC_MARK 255
#define _XMP_TCA_COLL_MAX 64
#define _XMP_TCA_ALLREDUCE_TAG 0x100
#define _XMP_TCA_DEVICE_TO_HOST_WAIT_SLOT 0
#define _XMP_TCA_HOST_TO_DEVICE_WAIT_SLOT 1
#define _XMP_TCA_ALLREDUCE_TCACOPY_LIMIT 8

typedef struct _XMP_tca_coll_info_type {
  int tail_id;
  void *dev_addr[_XMP_TCA_COLL_MAX];
  int count[_XMP_TCA_COLL_MAX];
  int datatype[_XMP_TCA_COLL_MAX];
  int op[_XMP_TCA_COLL_MAX];
  MPI_Comm mpi_comm[_XMP_TCA_COLL_MAX];
  void *cpu_sendbuf[_XMP_TCA_COLL_MAX];
  void *cpu_recvbuf[_XMP_TCA_COLL_MAX];
  tcaHandle *recv_handles[_XMP_TCA_COLL_MAX];
  tcaHandle send_handles[_XMP_TCA_COLL_MAX];
  tcaHandle device_handles[_XMP_TCA_COLL_MAX];
  tcaPIOHandle *pio_handles[_XMP_TCA_COLL_MAX];
  tcaDesc *d2h_desc[_XMP_TCA_COLL_MAX];
  tcaDesc *h2d_desc[_XMP_TCA_COLL_MAX];
  _Bool flag[_XMP_TCA_COLL_MAX];
  size_t datasize[_XMP_TCA_COLL_MAX];
  int num_comms[_XMP_TCA_COLL_MAX];
  size_t recv_next_aligned_stride[_XMP_TCA_COLL_MAX];
  tcaOp tca_op[_XMP_TCA_COLL_MAX];
  tcaDataType tca_datatype[_XMP_TCA_COLL_MAX];
} _XMP_tca_coll_info_t;

_XMP_tca_coll_info_t coll_info;
int _XMP_tca_coll_info_flag = 0;

void _XMP_reduce_tca_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
void _XMP_reduce_tca_CLAUSE(void *data_addr, int count, int datatype, int op);

#define CUDA_CHECK(cuda_call) do {                                      \
    cudaError_t status = cuda_call;                                     \
    if(status != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",     \
	      __FILE__, __LINE__, cudaGetErrorString(status) );         \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

static void _XMP_setup_tca_reduce_type(tcaDataType *tca_datatype, size_t *datatype_size, int datatype) {
  switch (datatype) {
    //  case _XMP_N_TYPE_BOOL:
    //    { *tca_datatype = TCA_C_BOOL;                 *datatype_size = sizeof(_Bool);                         break; }
  case _XMP_N_TYPE_CHAR:
    { *tca_datatype = TCA_SIGNED_CHAR;                  *datatype_size = sizeof(char);                          break; }
  case _XMP_N_TYPE_UNSIGNED_CHAR:
    { *tca_datatype = TCA_UNSIGNED_CHAR;                *datatype_size = sizeof(unsigned char);                 break; }
  case _XMP_N_TYPE_SHORT:
    { *tca_datatype = TCA_SHORT;                        *datatype_size = sizeof(short);                         break; }
  case _XMP_N_TYPE_UNSIGNED_SHORT:
    { *tca_datatype = TCA_UNSIGNED_SHORT;               *datatype_size = sizeof(unsigned short);                break; }
  case _XMP_N_TYPE_INT:
    { *tca_datatype = TCA_INT;                          *datatype_size = sizeof(int);                           break; }
  case _XMP_N_TYPE_UNSIGNED_INT:
    { *tca_datatype = TCA_UNSIGNED;                     *datatype_size = sizeof(unsigned int);                  break; }
  case _XMP_N_TYPE_LONG:
    { *tca_datatype = TCA_LONG;                         *datatype_size = sizeof(long);                          break; }
  case _XMP_N_TYPE_UNSIGNED_LONG:
    { *tca_datatype = TCA_UNSIGNED_LONG;                *datatype_size = sizeof(unsigned long);                 break; }
  case _XMP_N_TYPE_LONGLONG:
    { *tca_datatype = TCA_LONG_LONG;                    *datatype_size = sizeof(long long);                     break; }
  case _XMP_N_TYPE_UNSIGNED_LONGLONG:
    { *tca_datatype = TCA_UNSIGNED_LONG_LONG;           *datatype_size = sizeof(unsigned long long);            break; }
  case _XMP_N_TYPE_FLOAT:
    { *tca_datatype = TCA_FLOAT;                        *datatype_size = sizeof(float);                         break; }
  case _XMP_N_TYPE_DOUBLE:
    { *tca_datatype = TCA_DOUBLE;                       *datatype_size = sizeof(double);                        break; }
  case _XMP_N_TYPE_LONG_DOUBLE:
    { *tca_datatype = TCA_LONG_DOUBLE;                  *datatype_size = sizeof(long double);                   break; }
#ifdef __STD_IEC_559_COMPLEX__
  case _XMP_N_TYPE_FLOAT_IMAGINARY:
    { *tca_datatype = TCA_FLOAT;                        *datatype_size = sizeof(float _Imaginary);              break; }
  case _XMP_N_TYPE_DOUBLE_IMAGINARY:
    { *tca_datatype = TCA_DOUBLE;                       *datatype_size = sizeof(double _Imaginary);             break; }
  case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
    { *tca_datatype = TCA_LONG_DOUBLE;                  *datatype_size = sizeof(long double _Imaginary);        break; }
#endif
    /* case _XMP_N_TYPE_FLOAT_COMPLEX: */
    /*   { *tca_datatype = TCA_C_FLOAT_COMPLEX;         *datatype_size = sizeof(float _Complex);                break; } */
    /* case _XMP_N_TYPE_DOUBLE_COMPLEX: */
    /*   { *tca_datatype = TCA_C_DOUBLE_COMPLEX;                *datatype_size = sizeof(double _Complex);               break; } */
    /* case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX: */
    /*   { *tca_datatype = TCA_C_LONG_DOUBLE_COMPLEX;   *datatype_size = sizeof(long double _Complex);          break; } */
  default:
    _XMP_fatal("unknown data type for reduction");
  }
}

static void _XMP_setup_tca_reduce_op(tcaOp *tca_op, int op) {
  switch (op) {
  case _XMP_N_REDUCE_SUM:
    *tca_op = TCA_OP_SUM;
    break;
  case _XMP_N_REDUCE_PROD:
    *tca_op = TCA_OP_PROD;
    break;
  case _XMP_N_REDUCE_BAND:
    *tca_op = TCA_OP_BAND;
    break;
  case _XMP_N_REDUCE_LAND:
    *tca_op = TCA_OP_LAND;
    break;
  case _XMP_N_REDUCE_BOR:
    *tca_op = TCA_OP_BOR;
    break;
  case _XMP_N_REDUCE_LOR:
    *tca_op = TCA_OP_LOR;
    break;
  case _XMP_N_REDUCE_BXOR:
    *tca_op = TCA_OP_BXOR;
    break;
  case _XMP_N_REDUCE_LXOR:
    *tca_op = TCA_OP_LXOR;
    break;
  case _XMP_N_REDUCE_MAX:
    *tca_op = TCA_OP_MAX;
    break;
  case _XMP_N_REDUCE_MIN:
    *tca_op = TCA_OP_MIN;
    break;
  case _XMP_N_REDUCE_FIRSTMAX:
    *tca_op = TCA_OP_MAX;
    break;
  case _XMP_N_REDUCE_FIRSTMIN:
    *tca_op = TCA_OP_MIN;
    break;
  case _XMP_N_REDUCE_LASTMAX:
    *tca_op = TCA_OP_MAX;
    break;
  case _XMP_N_REDUCE_LASTMIN:
    *tca_op = TCA_OP_MIN;
    break;
  case _XMP_N_REDUCE_EQV:
  case _XMP_N_REDUCE_NEQV:
  case _XMP_N_REDUCE_MINUS:
    _XMP_fatal("unsupported reduce operation");
  default:
    _XMP_fatal("unknown reduce operation");
  }
}

typedef void (*tca_op_func_3op_handler_t)(void *, void *, void *, int);
typedef tca_op_func_3op_handler_t tca_op_fn_3op_t;

#define OP_FUNC_3OP(name, op, type_name, type)				\
  static inline void tca_op_func_3op_##name##_##type_name(void *dst, void *src0, void *src1, int count) { \
    int i;								\
    type *s0 = (type *)src0;						\
    type *s1 = (type *)src1;						\
    type *d = (type *)dst;						\
    for(i = 0; i < count; i++) {					\
      *(d++) = *(s0++) op *(s1++);					\
    }									\
  }

OP_FUNC_3OP(sum, +, int8, int8_t)
OP_FUNC_3OP(sum, +, uint8, uint8_t)
OP_FUNC_3OP(sum, +, int16, int16_t)
OP_FUNC_3OP(sum, +, uint16, uint16_t)
OP_FUNC_3OP(sum, +, int32, int32_t)
OP_FUNC_3OP(sum, +, uint32, uint32_t)
OP_FUNC_3OP(sum, +, int64, int64_t)
OP_FUNC_3OP(sum, +, uint64, uint64_t)
OP_FUNC_3OP(sum, +, float, float)
OP_FUNC_3OP(sum, +, double, double)
OP_FUNC_3OP(sum, +, long_double, long double)

OP_FUNC_3OP(prod, *, int8, int8_t)
OP_FUNC_3OP(prod, *, uint8, uint8_t)
OP_FUNC_3OP(prod, *, int16, int16_t)
OP_FUNC_3OP(prod, *, uint16, uint16_t)
OP_FUNC_3OP(prod, *, int32, int32_t)
OP_FUNC_3OP(prod, *, uint32, uint32_t)
OP_FUNC_3OP(prod, *, int64, int64_t)
OP_FUNC_3OP(prod, *, uint64, uint64_t)
OP_FUNC_3OP(prod, *, float, float)
OP_FUNC_3OP(prod, *, double, double)
OP_FUNC_3OP(prod, *, long_double, long double)

enum {
  TCA_OP_INT8,
  TCA_OP_UINT8,
  TCA_OP_INT16,
  TCA_OP_UINT16,
  TCA_OP_INT32,
  TCA_OP_UINT32,
  TCA_OP_INT64,
  TCA_OP_UINT64,
  TCA_OP_FLOAT,
  TCA_OP_DOUBLE,
  TCA_OP_LONG_DOUBLE,
  TCA_OP_TYPE_MAX,
};

#define TCA_TYPE_FUNCTIONS(name, type)					\
  [TCA_OP_INT8]        = tca_op_func_##type##_##name##_int8,		\
    [TCA_OP_UINT8]       = tca_op_func_##type##_##name##_uint8,		\
    [TCA_OP_INT16]       = tca_op_func_##type##_##name##_int16,		\
    [TCA_OP_UINT16]      = tca_op_func_##type##_##name##_uint16,	\
    [TCA_OP_INT32]       = tca_op_func_##type##_##name##_int32,		\
    [TCA_OP_UINT32]      = tca_op_func_##type##_##name##_uint32,	\
    [TCA_OP_INT64]       = tca_op_func_##type##_##name##_int64,		\
    [TCA_OP_UINT64]      = tca_op_func_##type##_##name##_uint64,	\
    [TCA_OP_FLOAT]       = tca_op_func_##type##_##name##_float,		\
    [TCA_OP_DOUBLE]      = tca_op_func_##type##_##name##_double,	\
    [TCA_OP_LONG_DOUBLE] = tca_op_func_##type##_##name##_long_double	\

static tca_op_fn_3op_t tca_op_func_3op[TCA_OP_NOOP+1][TCA_OP_TYPE_MAX] =
  {
    [TCA_OP_SUM] = {
      TCA_TYPE_FUNCTIONS(sum, 3op),
    },
    [TCA_OP_PROD] = {
      TCA_TYPE_FUNCTIONS(prod, 3op),
    },
  };

static inline int get_tca_data_size(tcaDataType type) {
  return type >> 8;
}

static const int func_conv_table[(TCA_LONG_DOUBLE & 0xff) + 1] =
  {
    [TCA_CHAR & 0xff] = TCA_OP_UINT8,
    [TCA_SIGNED_CHAR & 0xff] = TCA_OP_INT8,
    [TCA_UNSIGNED_CHAR & 0xff] = TCA_OP_UINT8,
    [TCA_BYTE & 0xff] = TCA_OP_UINT8,
    [TCA_SHORT & 0xff] = TCA_OP_INT16,
    [TCA_UNSIGNED_SHORT & 0xff] = TCA_OP_UINT16,
    [TCA_INT & 0xff] = TCA_OP_INT32,
    [TCA_UNSIGNED & 0xff] = TCA_OP_UINT32,
    [TCA_LONG & 0xff] = TCA_OP_INT32,
    [TCA_UNSIGNED_LONG & 0xff] = TCA_OP_UINT32,
    [TCA_LONG_LONG & 0xff] = TCA_OP_INT64,
    [TCA_UNSIGNED_LONG_LONG & 0xff] = TCA_OP_UINT64,
    [TCA_FLOAT & 0xff] = TCA_OP_FLOAT,
    [TCA_DOUBLE & 0xff] = TCA_OP_DOUBLE,
    [TCA_LONG_DOUBLE & 0xff] = TCA_OP_LONG_DOUBLE,
  };

static inline int get_func_type_by_data_size(tcaDataType type) {
  return func_conv_table[type & 0xff];
}

static void init_coll_info()
{
  _XMP_tca_coll_info_flag = 1;
  coll_info.tail_id = 0;
  for (int i = 0; i < _XMP_TCA_COLL_MAX; i++) {
    coll_info.flag[i] = _XMP_N_INT_FALSE;
  }
}

static int get_coll_id(void *dev_addr, int count, int datatype, int op, MPI_Comm mpi_comm)
{
  for (int i = 0; i < coll_info.tail_id; i++) {
    if (coll_info.dev_addr[i] == dev_addr && coll_info.count[i] == count &&
	coll_info.datatype[i] == datatype && coll_info.op[i] == op && coll_info.mpi_comm[i] == mpi_comm) {
      return i;
    }
  }

  coll_info.dev_addr[coll_info.tail_id] = dev_addr;
  coll_info.count[coll_info.tail_id] = count;
  coll_info.datatype[coll_info.tail_id] = datatype;
  coll_info.op[coll_info.tail_id] = op;
  coll_info.mpi_comm[coll_info.tail_id] = mpi_comm;

  return coll_info.tail_id++;
}

static void _XMP_reduce_init_tca(void *dev_addr, int count, int datatype, int op, MPI_Comm mpi_comm, int id)
{
  int rank = _XMP_world_rank;
  int num_proc = _XMP_world_size;
  tcaDataType tca_datatype;
  tcaOp tca_op;
  size_t datatype_size;

  memset(&tca_datatype, 0x00, sizeof(tca_datatype));
  memset(&tca_op, 0x00, sizeof(tca_op));

  _XMP_setup_tca_reduce_type(&tca_datatype, &datatype_size, datatype);
  _XMP_setup_tca_reduce_op(&tca_op, op);

  const size_t datasize = count * datatype_size;
  const size_t sendsize = datasize + _XMP_TCA_SYNC_MARK_SIZE;
  const int num_comms = (int)log2(num_proc);
  const size_t recv_next_aligned_stride = (sendsize + _XMP_TCA_CACHE_ALIGNED_STRIDE - 1) & ~(_XMP_TCA_CACHE_ALIGNED_STRIDE - 1); // 64byte aligned
  const size_t recvsize = recv_next_aligned_stride * (num_comms + 1);
  
  TCA_CHECK(tcaMalloc(&coll_info.cpu_sendbuf[id], sendsize, tcaMemoryCPU));
  TCA_CHECK(tcaMalloc(&coll_info.cpu_recvbuf[id], recvsize, tcaMemoryCPU));
  coll_info.recv_handles[id] = (tcaHandle *)_XMP_alloc(sizeof(tcaHandle) * (num_comms + 1));
  coll_info.pio_handles[id] = (tcaPIOHandle *)_XMP_alloc(sizeof(tcaPIOHandle) * num_comms);

  void *cpu_sendbuf = coll_info.cpu_sendbuf[id];
  void *cpu_recvbuf = coll_info.cpu_recvbuf[id];
  tcaHandle *recv_h = (tcaHandle *)coll_info.recv_handles[id];
  tcaHandle *send_h = &coll_info.send_handles[id];
  tcaHandle *device_h = &coll_info.device_handles[id];
  tcaPIOHandle *pio_h = (tcaPIOHandle *)coll_info.pio_handles[id];

  *(unsigned long *)((unsigned long)cpu_sendbuf + datasize) = _XMP_TCA_PIO_SYNC_MARK;
  TCA_CHECK(tcaCreateHandle(&recv_h[0], cpu_recvbuf, recvsize, tcaMemoryCPU));
  TCA_CHECK(tcaCreateHandle(send_h, cpu_sendbuf, datasize, tcaMemoryCPU));
  TCA_CHECK(tcaCreateHandle(device_h, dev_addr, datasize, tcaMemoryGPU));

  // CPU to CPU
  int i, distance;
  for (distance = 1, i = 0; distance < num_proc; distance <<= 1, i++) {
    const int dest = (rank + distance) % num_proc;
    const int src = (rank + num_proc - distance) % num_proc;
    MPI_Sendrecv(&recv_h[0], sizeof(tcaHandle), MPI_BYTE, src, 0, &recv_h[i+1], sizeof(tcaHandle), MPI_BYTE, dest, 0, mpi_comm, MPI_STATUS_IGNORE);
    TCA_CHECK(tcaSetPIORegion(&pio_h[i], &recv_h[i+1], 0, recv_next_aligned_stride));
  }

  coll_info.d2h_desc[id] = tcaDescNew();
  coll_info.h2d_desc[id] = tcaDescNew();
  const int dma_flag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify;
  // Device to Host
  TCA_CHECK(tcaDescSetMemcpy(coll_info.d2h_desc[id], send_h, 0, device_h, 0, datasize, dma_flag, _XMP_TCA_DEVICE_TO_HOST_WAIT_SLOT, _XMP_TCA_ALLREDUCE_TAG));

  // Host to Device
  TCA_CHECK(tcaDescSetMemcpy(coll_info.h2d_desc[id], device_h, 0, send_h, 0, datasize, dma_flag, _XMP_TCA_HOST_TO_DEVICE_WAIT_SLOT, _XMP_TCA_ALLREDUCE_TAG));

  MPI_Barrier(mpi_comm);

  coll_info.flag[id] = _XMP_N_INT_TRUE;
  coll_info.datasize[id] = datasize;
  coll_info.num_comms[id] = num_comms;
  coll_info.recv_next_aligned_stride[id] = recv_next_aligned_stride;
  coll_info.tca_op[id] = tca_op;
  coll_info.tca_datatype[id] = tca_datatype;
}

static void _XMP_reduce_do_tca(void *dev_addr, int count, int datatype, int op, MPI_Comm mpi_comm, int id)
{
  void *cpu_sendbuf = coll_info.cpu_sendbuf[id];
  void *cpu_recvbuf = coll_info.cpu_recvbuf[id];
  tcaPIOHandle *pio_h = (tcaPIOHandle *)coll_info.pio_handles[id];
  tcaOp tca_op = coll_info.tca_op[id];
  tcaDataType tca_datatype = coll_info.tca_datatype[id];
  tcaHandle *h = (tcaHandle *)coll_info.recv_handles[id];
  tcaHandle *device_h = &coll_info.device_handles[id];
  tcaDesc *d2h_desc = coll_info.d2h_desc[id];
  tcaDesc *h2d_desc = coll_info.h2d_desc[id];

  tcaSendPIOCommit();
  volatile void *init_ptr_recv = (volatile void *)cpu_recvbuf;
  size_t recv_offset = 0;
  int i;
  const int num_comms = coll_info.num_comms[id];
  const size_t datasize = coll_info.datasize[id];
  const size_t recv_next_aligned_stride = coll_info.recv_next_aligned_stride[id];
  const size_t sendsize = datasize + _XMP_TCA_SYNC_MARK_SIZE;

  // copy device to host
  if (count <= _XMP_TCA_ALLREDUCE_TCACOPY_LIMIT) {
    TCA_CHECK(tcaDescSet(d2h_desc, 0));
    TCA_CHECK(tcaStartDMADesc(0));
    /* TCA_CHECK(tcaWaitDMAC(0)); */
    TCA_CHECK(tcaWaitDMARecvDesc(h, _XMP_TCA_DEVICE_TO_HOST_WAIT_SLOT, _XMP_TCA_ALLREDUCE_TAG));
  } else {
    CUDA_CHECK(cudaMemcpy(cpu_sendbuf, dev_addr, datasize, cudaMemcpyDeviceToHost));
  }

  // allreduce on CPU memory
  for (i = 0; i < num_comms; i++) {
    volatile unsigned long *pio_wait = (volatile unsigned long *)((unsigned long)cpu_recvbuf + datasize);
    TCA_CHECK(tcaSendPIO(&pio_h[i], recv_offset, (void *)cpu_sendbuf, sendsize));
    tcaSendPIOCommit();

    unsigned long j = 2147483647UL;
    while(*pio_wait != _XMP_TCA_PIO_SYNC_MARK && --j) {
      _mm_pause();
    }
    if (!j) {
      _XMP_fatal("pio_wait time out.");
    }

    *pio_wait = 0;
    if (i < num_comms - 1) {
      tca_op_func_3op[tca_op][get_func_type_by_data_size(tca_datatype)](cpu_sendbuf, cpu_sendbuf, cpu_recvbuf, count);
      recv_offset += recv_next_aligned_stride;
      cpu_recvbuf = (void *)((unsigned long)cpu_recvbuf + recv_next_aligned_stride);
    } else {
      tca_op_func_3op[tca_op][get_func_type_by_data_size(tca_datatype)](cpu_sendbuf, cpu_sendbuf, cpu_recvbuf, count);
    }
  }
  cpu_recvbuf = (void *)init_ptr_recv;

  // copy host to device
  if (count <= _XMP_TCA_ALLREDUCE_TCACOPY_LIMIT) {
    TCA_CHECK(tcaDescSet(h2d_desc, 0));
    TCA_CHECK(tcaStartDMADesc(0));
    /* TCA_CHECK(tcaWaitDMAC(0)); */
    TCA_CHECK(tcaWaitDMARecvDesc(device_h, _XMP_TCA_HOST_TO_DEVICE_WAIT_SLOT, _XMP_TCA_ALLREDUCE_TAG));
  } else {
    CUDA_CHECK(cudaMemcpy(dev_addr, cpu_sendbuf, datasize, cudaMemcpyHostToDevice));
  }
}

void _XMP_reduce_tca_NODES_ENTIRE(_XMP_nodes_t *nodes, void *dev_addr, int count, int datatype, int op)
{
  if (count == 0) {
    return; // FIXME not good implementation
  }
  if (!nodes->is_member) {
    return;
  }
  if (_XMP_tca_coll_info_flag) {
    init_coll_info();
  }

  MPI_Comm mpi_comm = *((MPI_Comm *)nodes->comm);

  int id = get_coll_id(dev_addr, count, datatype, op, mpi_comm);
  if (!coll_info.flag[id]) {
    _XMP_reduce_init_tca(dev_addr, count, datatype, op, mpi_comm, id);
  }

  _XMP_reduce_do_tca(dev_addr, count, datatype, op, mpi_comm, id);
}
  
void _XMP_reduce_tca_CLAUSE(void *dev_addr, int count, int datatype, int op)
{
  // Not implemented
  _XMP_fatal("_XMP_reduce_tca_CLAUSE is not implemented.");
}
