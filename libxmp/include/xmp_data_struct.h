#ifndef _XMP_DATA_STRUCT
#define _XMP_DATA_STRUCT
#include <mpi.h>
#include <stdint.h>
#include <stdbool.h>
#include "xmp_constant.h"
#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
#include <mpi-ext.h>
#endif

#define _XMP_comm_t void

// nodes descriptor
typedef struct _XMP_nodes_inherit_info_type {
  int shrink;
  // enable when shrink is false
  int lower;
  int upper;
  int stride;
  // ---------------------------

  int size;
} _XMP_nodes_inherit_info_t;

typedef struct _XMP_nodes_info_type {
  int size;

  // enable when is_member is true
  int rank;
  // -----------------------------
  int multiplier;
} _XMP_nodes_info_t;

typedef struct _XMP_nodes_type {
  unsigned long long on_ref_id;

  int is_member;
  int dim;
  int comm_size;
  int attr;

  // enable when is_member is true
  int comm_rank;
  _XMP_comm_t *comm;
  _XMP_comm_t *subcomm;  // enable when attr is _XMP_ENTIRE_NODES
  int use_subcomm;       // enable when attr is _XMP_ENTIRE_NODES

  struct _XMP_nodes_type *inherit_nodes;
  // enable when inherit_nodes is not NULL
  _XMP_nodes_inherit_info_t *inherit_info;
  _XMP_nodes_info_t info[1];
} _XMP_nodes_t;

typedef struct _XMP_nodes_ref_type {
  _XMP_nodes_t *nodes;
  int *ref;
  int shrink_nodes_size;
} _XMP_nodes_ref_t;

// template desciptor
typedef struct _XMP_template_info_type {
  // enable when is_fixed is true
  long long ser_lower;
  long long ser_upper;
  unsigned long long ser_size;
  // ----------------------------
} _XMP_template_info_t;

typedef struct _XMP_template_chunk_type {
  // enable when is_owner is true
  long long par_lower;
  long long par_upper;
  unsigned long long par_width;
  // ----------------------------

  int par_stride;
  unsigned long long par_chunk_width;
  int dist_manner;
  long long *mapping_array;
  _Bool is_regular_chunk;

  // enable when dist_manner is not _XMP_N_DIST_DUPLICATION
  int onto_nodes_index;
  // enable when onto_nodes_index is not _XMP_N_NO_ONTO_NODES
  _XMP_nodes_info_t *onto_nodes_info;
  // --------------------------------------------------------
} _XMP_template_chunk_t;

typedef struct _XMP_template_type {
  unsigned long long on_ref_id;

  _Bool is_fixed;
  _Bool is_distributed;
  _Bool is_owner;
  
  int   dim;

  // enable when is_distributed is true
  _XMP_nodes_t *onto_nodes;
  _XMP_template_chunk_t *chunk;
  // ----------------------------------

  _XMP_template_info_t info[1];
} _XMP_template_t;

// schedule of reflect comm.
typedef struct _XMP_reflect_sched_type {

  int lo_width, hi_width;
  int is_periodic;

  MPI_Datatype datatype_lo;
  MPI_Datatype datatype_hi;

  MPI_Request req[4];

  void *lo_send_buf, *lo_recv_buf;
  void *hi_send_buf, *hi_recv_buf;

  void *lo_send_array, *lo_recv_array;
  void *hi_send_array, *hi_recv_array;

  int count, blocklength;
  long long stride;

  int lo_rank, hi_rank;

#if defined(_XMP_XACC)
  void *lo_send_host_buf, *lo_recv_host_buf;
  void *hi_send_host_buf, *hi_recv_host_buf;
  void *lo_async_id;
  void *hi_async_id;
  void *event;
#endif
  
#if defined(_XMP_TCA)
  off_t lo_src_offset, lo_dst_offset;
  off_t hi_src_offset, hi_dst_offset;
#endif
} _XMP_reflect_sched_t;

// schedule of asynchronous reflect
typedef struct _XMP_async_reflect_type {

  int lwidth[_XMP_N_MAX_DIM], uwidth[_XMP_N_MAX_DIM];
  _Bool is_periodic[_XMP_N_MAX_DIM];

  MPI_Datatype *datatype;
  MPI_Request *reqs;
  int nreqs;

} _XMP_async_reflect_t;

// aligned array descriptor
typedef struct _XMP_array_info_type {
  _Bool is_shadow_comm_member;
  _Bool is_regular_chunk;
  int align_manner;

  int ser_lower;  // Lower bound of a global array, and output value of xmp_array_lbound().
  int ser_upper;  // Upper bound of a global array, and output value of xmp_array_lbound().
  int ser_size;   // Size of a global array.

  // enable when is_allocated is true
  int par_lower;  // Lower bound of a global array on a node.
  int par_upper;  // Upper bound of a global array on a node.
  int par_stride; // Stride of a global array on a node.
  int par_size;   // Size of a global array on a node.

  int local_lower;  // Lower bound of a local array
  int local_upper;  // Upper bound of a local array
  int local_stride; // Stride of a local array
  int alloc_size;   // Number of elements of a local array

  // ex1)
  // #pragma xmp template t(0:19)
  // #pragma xmp nodes p(4)
  // #pragma xmp distribute t(block) onto p
  // int a[20];
  // #pragma xmp align a[i] with t(i)
  //
  // ser_lower   -> 0, ser_upper   -> 19, ser_size -> 20
  // par_lower   -> 5*(xmp_node_num()-1),
  // par_upper   -> 5*xmp_node_num()-1, par_size -> 5
  // local_lower -> 0, local_upper -> 4, alloc_size -> 5

  // ex2)
  // #pragma xmp template t(0:19)
  // #pragma xmp nodes p(4)
  // #pragma xmp distribute t(block) onto p
  // int a[20];
  // #pragma xmp align a[i] with t(i)
  // #pragma xmp shadow a[1]
  //
  // Values of ser_lower, ser_upper, ser_size, par_lower, par_upper, par_size are
  // the same of ex1).
  //
  // local_lower -> 1, local_upper -> 5, alloc_size -> 7
  
  int *temp0;
  int temp0_v;

  unsigned long long dim_acc;
  unsigned long long dim_elmts;
  // --------------------------------

  long long align_subscript;

  int shadow_type;
  int shadow_size_lo;
  int shadow_size_hi;

  _XMP_reflect_sched_t *reflect_sched;
  _XMP_reflect_sched_t *reflect_acc_sched;
  // enable when is_shadow_comm_member is true
  _XMP_comm_t *shadow_comm;
  int shadow_comm_size;
  int shadow_comm_rank;
  // -----------------------------------------

  int align_template_index;

  unsigned long long *acc;

} _XMP_array_info_t;

typedef struct _XMP_array_type {
  _Bool is_allocated;
  _Bool is_align_comm_member;
  int dim;
  int type;
  size_t type_size;
  MPI_Datatype mpi_type;
  int order;

  // enable when is_allocated is true
  void *array_addr_p;
#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  uint64_t rdma_addr;
  int rdma_memid;
#endif
#if defined(_XMP_TCA)
  void* tca_handle;
  _Bool set_handle;  // If tca_handle has been set, set_handle = true
  int   dma_slot;
  int   wait_slot;
  int   wait_tag;
#endif
  unsigned long long total_elmts;
  // --------------------------------

  _XMP_async_reflect_t *async_reflect;

  // FIXME do not use these members
  // enable when is_align_comm_member is true
  _XMP_comm_t *align_comm;
  int align_comm_size;
  int align_comm_rank;
  // ----------------------------------------

  _Bool is_shrunk_template;
  _XMP_nodes_t *array_nodes;

#ifdef _XMP_MPI3_ONESIDED
  struct xmp_coarray *coarray;
#endif

  _XMP_template_t *align_template;
  _XMP_array_info_t info[1];
} _XMP_array_t;

typedef struct _XMP_task_desc_type {
  _XMP_nodes_t *nodes;
  int execute;

  unsigned long long on_ref_id;

  int ref_lower[_XMP_N_MAX_DIM];
  int ref_upper[_XMP_N_MAX_DIM];
  int ref_stride[_XMP_N_MAX_DIM];
} _XMP_task_desc_t;

// Note: When member is changed, _XMP_coarray_deallocate() may be changed.
typedef struct xmp_coarray{
  char **addr;      // Pointer to each node.
                    // e.g.) xmp_coarray.addr[2] is a pointer of an object on node 2.

#ifdef _XMP_FJRDMA
  uint64_t laddr;  // On the FJRDMA machines, xmp_coarray.addr[_XMP_world_rank] is not local address of a coarray,
                   // Thus, "laddr" is defined the local address.
#endif
  char *real_addr; // Pointer to local node.
                   // Note that xmp_coarray.addr[my_rank] may not be a pointer of an object.

  size_t elmt_size; // Element size of a coarray. A unit of it is Byte.
                    // e.g.) "int a[10]:[*]" is 4.

  int coarray_dims; // Number of dimensions of coarray.
                    // e.g.) "int a[10][20]:[4][2][*]" is 2.

  long *coarray_elmts; // Number of elements of each dimension of a coarray.
                       // e.g.) When "int a[10][20]:[*]", coarray_elmts[0] is 10, coarray_elmts[1] is 20.
                       //       If a coarray is scalar, coarray_elmts[0] is 1.

  long *distance_of_coarray_elmts; // Distance between each dimension of coarray. A unit of the distance is Byte.
                                   // e.g.) When "int a[10][20][30]:[*]", distance_of_coarray_elmts[0] is 2400 (20*30*sizeof(int)),
                                   //       distance_of_coarray_elmts[1] is 120 (30*sizeof(int)),
                                   //       distance_of_coarray_elmts[0] is 4 (sizeof(int)).

  int image_dims; // Number of dimensions of image set.
                  // e.g.) When "int a[10][20]:[4][2][*]" is 3.

  int *distance_of_image_elmts; // Distance between each dimension of image set.
                                // e.g.) When "int a[10][20]:[4][2][*]", distance_of_image_elmts[0] is 1,
                                //       distance_of_image_elmts[1] is 4, distance_of_image_elmts[2] is 8.
#ifdef _XMP_MPI3_ONESIDED
  MPI_Win win;
  //#ifdef _XMP_XACC
  char **addr_dev;
  char *real_addr_dev;
  MPI_Win win_acc;
  //#endif
#endif
}_XMP_coarray_t;

typedef struct _XMP_array_section{
  long start;
  long length;
  long stride;
  long elmts;    // Number of elements in each dimension
  long distance; // Distance between each dimension of an array.
                 // e.g.) When "int a[10][20][30]", _XMP_array_section_t[0].distance is 20*30*sizeof(int),
                 //       _XMP_array_section_t[1].distance is 30*sizeof(int), 
                 //       _XMP_array_section_t[0].distance is sizeof(int),
} _XMP_array_section_t;

typedef struct _XMP_gmv_desc_type
{
  _Bool is_global;
  int ndims;

  _XMP_array_t *a_desc;

  void *local_data;
  int *a_lb;
  int *a_ub;

  int *kind;
  int *lb;
  int *ub;
  int *st;

#if defined(_XMP_XACC)
  void *a_dev;
#endif
} _XMP_gmv_desc_t;

// Regular Section Descriptor (RSD)
// (l:u:s)
typedef struct _XMP_rsd_type {
  int l;
  int u;
  int s;
} _XMP_rsd_t;

// Basic Section Descriptor (BSD)
// (l:u:b:c)
// b = block width
// c = cyclic length
typedef struct _XMP_bsd_type {
  int l;
  int u;
  int b;
  int c;
} _XMP_bsd_t;

// Common Section Descriptor (CSD)
// ((l_1,...,l_n):(u_1,...,u_n):s)
// proposed in Gwan-Hwan Hwang , Jenq Kuen Lee, Communication set generations with CSD calculus and
// expression-rewriting framework, Parallel Computing, v.25 n.9, p.1105-1130, Sept. 1999 
typedef struct _XMP_csd_type {
  int *l;
  int *u;
  int n;
  int s;
} _XMP_csd_t;

// Communication Set
// ((l_1:u_1),(l_2:u_2),...)
typedef struct _XMP_comm_set_type {
  int l;
  int u;
  struct _XMP_comm_set_type *next;
} _XMP_comm_set_t;

//
// for asynchronous comms.
//

typedef struct _XMP_async_gmove {
  int mode;
  void *sendbuf;
  void *recvbuf;
  int recvbuf_size;
  _XMP_array_t *a;
  _XMP_comm_set_t *(*comm_set)[_XMP_N_MAX_DIM];
} _XMP_async_gmove_t;

typedef struct _XMP_async_comm {
  int   async_id;
  int   nreqs;
  int   nnodes;
  _Bool is_used;
  MPI_Request *reqs;
  _XMP_nodes_t **node;
  _XMP_async_gmove_t *gmove;
  struct _XMP_async_comm *next;
} _XMP_async_comm_t;

#define _XMP_ASYNC_COMM_SIZE 511
#define _XMP_MAX_ASYNC_REQS  (4 * _XMP_N_MAX_DIM * 10)
#define _XMP_MAX_ASYNC_NODES (20)

typedef struct _XMP_gpu_array_type {
  int gtol;
  unsigned long long acc;
} _XMP_gpu_array_t;

typedef struct _XMP_gpu_data_type {
  _Bool is_aligned_array;
  void *host_addr;
  void *device_addr;
  _XMP_array_t *host_array_desc;
  _XMP_gpu_array_t *device_array_desc;
  size_t size;
} _XMP_gpu_data_t;

#endif // _XMP_DATA_STRUCT
