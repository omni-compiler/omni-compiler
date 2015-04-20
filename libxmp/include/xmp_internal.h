#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#ifndef _XMP_INTERNAL
#define _XMP_INTERNAL

extern int _XMPC_running;
extern int _XMPF_running;

#ifndef MIN
#define MIN(a,b)  ( (a)<(b) ? (a) : (b) )
#endif

#ifndef MAX
#define MAX(a,b)  ( (a)>(b) ? (a) : (b) )
#endif
// --------------- including headers  --------------------------------
#include <mpi.h>
#include <stdio.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
// --------------- macro functions -----------------------------------
#ifdef DEBUG
#define _XMP_ASSERT(_flag) \
{ \
  if (!(_flag)) { \
    _XMP_unexpected_error(); \
  } \
}
#else
#define _XMP_ASSERT(_flag)
#endif

#define _XMP_RETURN_IF_SINGLE \
{ \
  if (_XMP_world_size == 1) { \
    return; \
  } \
}

#define _XMP_IS_SINGLE \
(_XMP_world_size == 1)

// --------------- structures ----------------------------------------
#include "xmp_data_struct.h"

#ifdef __cplusplus
extern "C" {
#define restrict __restrict__
#define template template_
#endif

// ----- libxmp ------------------------------------------------------
// xmp_align.c
extern void _XMP_calc_array_dim_elmts(_XMP_array_t *array, int array_index);
extern void _XMP_init_array_desc(_XMP_array_t **array, _XMP_template_t *template, int dim,
				 int type, size_t type_size, ...);
extern void _XMP_finalize_array_desc(_XMP_array_t *array);
extern void _XMP_align_array_NOT_ALIGNED(_XMP_array_t *array, int array_index);
extern void _XMP_align_array_DUPLICATION(_XMP_array_t *array, int array_index, int template_index,
                                  long long align_subscript);
extern void _XMP_align_array_BLOCK(_XMP_array_t *array, int array_index, int template_index,
                            long long align_subscript, int *temp0);
extern void _XMP_align_array_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                             long long align_subscript, int *temp0);
extern void _XMP_align_array_BLOCK_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                                   long long align_subscript, int *temp0);
extern void _XMP_align_array_GBLOCK(_XMP_array_t *array, int array_index, int template_index,
				    long long align_subscript, int *temp0);
extern void _XMP_init_array_nodes(_XMP_array_t *array);
extern void _XMP_init_array_comm(_XMP_array_t *array, ...);
extern void _XMP_init_array_comm2(_XMP_array_t *array, int args[]);
extern void _XMP_alloc_array(void **array_addr, _XMP_array_t *array_desc, ...);
extern void _XMP_alloc_array2(void **array_addr, _XMP_array_t *array_desc, unsigned long long *acc[]);
extern void _XMP_dealloc_array(_XMP_array_t *array_desc);

// xmp_array_section.c
extern void _XMP_normalize_array_section(_XMP_gmv_desc_t *gmv_desc, int idim, int *lower, int *upper, int *stride);
// FIXME make these static
extern void _XMP_pack_array_BASIC(void *buffer, void *src, int array_type,
                                         int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XMP_pack_array_GENERAL(void *buffer, void *src, size_t array_type_size,
                                           int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XMP_unpack_array_BASIC(void *dst, void *buffer, int array_type,
                                           int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XMP_unpack_array_GENERAL(void *dst, void *buffer, size_t array_type_size,
                                             int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XMP_pack_array(void *buffer, void *src, int array_type, size_t array_type_size,
                            int array_dim, int *l, int *u, int *s, unsigned long long *d);
extern void _XMP_unpack_array(void *dst, void *buffer, int array_type, size_t array_type_size,
                              int array_dim, int *l, int *u, int *s, unsigned long long *d);

// xmp_async.c
_XMP_async_comm_t *_XMP_get_or_create_async(int async_id);

// xmp_barrier.c
extern void _XMP_barrier_NODES_ENTIRE(_XMP_nodes_t *nodes);
extern void _XMP_barrier_EXEC(void);

// xmp_coarray.c
typedef struct _XMP_coarray_list_type {
  _XMP_coarray_t *coarray;
  struct _XMP_coarray_list_type *next;
} _XMP_coarray_list_t;

extern _XMP_coarray_list_t *_XMP_coarray_list_head;
extern _XMP_coarray_list_t *_XMP_coarray_list_tail;

extern void _XMP_onesided_initialize(int, char **);
extern void _XMP_onesided_finalize(const int);
extern void _XMP_build_coarray_queue();
extern void _XMP_coarray_lastly_deallocate();

// xmp_coarray_utils.c
extern void _XMP_local_put(const int, const int, const int, const int, _XMP_array_section_t *, _XMP_array_section_t *,
			   _XMP_coarray_t *, void *, const size_t, const size_t);
extern void _XMP_local_get(const int, const int, const int, const int, _XMP_array_section_t *, _XMP_array_section_t *,
			   _XMP_coarray_t *, void *, const size_t, const size_t);

// xmp_intrinsic.c
extern void xmpf_transpose(void *dst_p, void *src_p, int opt);
extern void xmpf_matmul(void *x_p, void *a_p, void *b_p);
extern void xmpf_pack_mask(void *v_p, void *a_p, void *m_p);
extern void xmpf_pack_nomask(void *v_p, void *a_p);
extern void xmpf_pack(void *v_p, void *a_p, void *m_p);
extern void xmpf_unpack_mask(void *a_p, void *v_p, void *m_p);
extern void xmpf_unpack_nomask(void *a_p, void *v_p);
extern void xmpf_unpack(void *a_p, void *v_p, void *m_p);

// xmp_gmove.c
extern void _XMP_gtol_array_ref_triplet(_XMP_array_t *array,
					int dim_index, int *lower, int *upper, int *stride);
extern int _XMP_calc_gmove_array_owner_linear_rank_SCALAR(_XMP_array_t *array, int *ref_index);
extern void _XMP_gmove_bcast_SCALAR(void *dst_addr, void *src_addr,
				    size_t type_size, int root_rank);
extern unsigned long long _XMP_gmove_bcast_ARRAY(void *dst_addr, int dst_dim,
					  int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
					  void *src_addr, int src_dim,
					  int *src_l, int *src_u, int *src_s, unsigned long long *src_d,
					  int type, size_t type_size, int root_rank);
extern int _XMP_check_gmove_array_ref_inclusion_SCALAR(_XMP_array_t *array, int array_index,
						       int ref_index);
extern void _XMP_gmove_localcopy_ARRAY(int type, int type_size,
                                       void *dst_addr, int dst_dim,
                                       int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
                                       void *src_addr, int src_dim,
                                       int *src_l, int *src_u, int *src_s, unsigned long long *src_d);
extern int _XMP_calc_global_index_HOMECOPY(_XMP_array_t *dst_array, int dst_dim_index,
					   int *dst_l, int *dst_u, int *dst_s,
					   int *src_l, int *src_u, int *src_s);
extern int _XMP_calc_global_index_BCAST(int dst_dim, int *dst_l, int *dst_u, int *dst_s,
					_XMP_array_t *src_array, int *src_array_nodes_ref,
					int *src_l, int *src_u, int *src_s);
extern void _XMP_gmove_SENDRECV_ARRAY(_XMP_array_t *dst_array, _XMP_array_t *src_array,
				      int type, size_t type_size, ...);
extern void _XMP_gmove_array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long *src_d);

// xmp_loop.c
extern int _XMP_sched_loop_template_width_1(int ser_init, int ser_cond, int ser_step,
                                            int *par_init, int *par_cond, int *par_step,
                                            int template_lower, int template_upper, int template_stride);
extern int _XMP_sched_loop_template_width_N(int ser_init, int ser_cond, int ser_step,
                                            int *par_init, int *par_cond, int *par_step,
                                            int template_lower, int template_upper, int template_stride,
                                            int width, int template_ser_lower, int template_ser_upper);
extern void _XMP_sched_loop_template_DUPLICATION(int ser_init, int ser_cond, int ser_step,
						 int *par_init, int *par_cond, int *par_step,
						 _XMP_template_t *template, int template_index);
extern void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
					   int *par_init, int *par_cond, int *par_step,
					   _XMP_template_t *template, int template_index);
extern void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
					    int *par_init, int *par_cond, int *par_step,
					    _XMP_template_t *template, int template_index);
extern void _XMP_sched_loop_template_BLOCK_CYCLIC(int ser_init, int ser_cond, int ser_step,
						  int *par_init, int *par_cond, int *par_step,
						  _XMP_template_t *template, int template_index);
extern void _XMP_sched_loop_template_GBLOCK(int ser_init, int ser_cond, int ser_step,
					    int *par_init, int *par_cond, int *par_step,
					    _XMP_template_t *template, int template_index);

// xmp_nodes.c
extern _XMP_nodes_t *_XMP_create_temporary_nodes(_XMP_nodes_t *n);
extern _XMP_nodes_t *_XMP_init_nodes_struct_GLOBAL(int dim, int *dim_size, int is_static);
extern _XMP_nodes_t *_XMP_init_nodes_struct_EXEC(int dim, int *dim_size, int is_static);
extern _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NUMBER(int dim, int ref_lower, int ref_upper, int ref_stride,
                                                         int *dim_size, int is_static);
extern _XMP_nodes_t *_XMP_init_nodes_struct_NODES_NAMED(int dim, _XMP_nodes_t *ref_nodes,
                                                        int *shrink, int *ref_lower, int *ref_upper, int *ref_stride,
                                                        int *dim_size, int is_static);
extern void _XMP_finalize_nodes(_XMP_nodes_t *nodes);
extern _XMP_nodes_t *_XMP_create_nodes_by_comm(int is_member, _XMP_comm_t *comm);
extern int _XMP_calc_linear_rank(_XMP_nodes_t *n, int *rank_array);
extern int _XMP_calc_linear_rank_on_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes);
extern _XMP_nodes_ref_t *_XMP_init_nodes_ref(_XMP_nodes_t *n, int *rank_array);
extern void _XMP_finalize_nodes_ref(_XMP_nodes_ref_t *nodes_ref);
extern _XMP_nodes_ref_t *_XMP_create_nodes_ref_for_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes);
extern void _XMP_translate_nodes_rank_array_to_ranks(_XMP_nodes_t *nodes, int *ranks, int *rank_array, int shrink_nodes_size);
extern int _XMP_get_next_rank(_XMP_nodes_t *nodes, int *rank_array);
extern int _XMP_calc_nodes_index_from_inherit_nodes_index(_XMP_nodes_t *nodes, int inherit_nodes_index);

// xmp_nodes_stack.c
extern void _XMP_push_nodes(_XMP_nodes_t *nodes);
extern void _XMP_pop_nodes(void);
extern void _XMP_pop_n_free_nodes(void);
extern void _XMP_pop_n_free_nodes_wo_finalize_comm(void);
extern _XMP_nodes_t *_XMP_get_execution_nodes(void);
extern int _XMP_get_execution_nodes_rank(void);
extern void _XMP_push_comm(_XMP_comm_t *comm);
extern void _XMP_finalize_comm(_XMP_comm_t *comm);

/* xmpf_pack_vector.c */
void _XMP_pack_vector(char * restrict dst, char * restrict src,
		      int count, int blocklength, long stride);
void _XMP_pack_vector2(char * restrict dst, char * restrict src,
                       int count, int blocklength,
                       int nnodes, int type_size, int src_block_dim);
void _XMP_unpack_vector(char * restrict dst, char * restrict src,
			int count, int blocklength, long stride);
void _XMPF_unpack_transpose_vector(char * restrict dst, char * restrict src,
                                   int dst_stride, int src_stride,
                                   int type_size, int dst_block_dim);
void _XMP_check_reflect_type(void);

// xmp_reduce.c
extern void _XMP_reduce_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
extern void _XMPF_reduce_FLMM_NODES_ENTIRE(_XMP_nodes_t *nodes,
					   void *addr, int count, int datatype, int op,
					   int num_locs, void **loc_vars, int *loc_types);

// xmp_reflect.c
extern void _XMP_set_reflect__(_XMP_array_t *a, int dim, int lwidth, int uwidth,
			       int is_periodic);
extern void _XMP_reflect__(_XMP_array_t *a);
extern void _XMP_wait_async__(int async_id);
extern void _XMP_reflect_async__(_XMP_array_t *a, int async_id);

// xmp_runtime.c
extern void _XMP_init(int argc, char** argv);
extern void _XMP_finalize(int return_val);

// xmp_shadow.c
extern void _XMP_create_shadow_comm(_XMP_array_t *array, int array_index);
extern void _XMP_reflect_shadow_FULL(void *array_addr, _XMP_array_t *array_desc, int array_index);
extern void _XMP_init_shadow(_XMP_array_t *array, ...);

// xmp_sort.c
extern void _XMP_sort(_XMP_array_t *a_desc, _XMP_array_t *b_desc, int is_up);

// xmp_template.c
extern _XMP_template_t *_XMP_create_template_desc(int dim, _Bool is_fixed);
extern int _XMP_check_template_ref_inclusion(int ref_lower, int ref_upper, int ref_stride,
                                             _XMP_template_t *t, int index);
extern void _XMP_init_template_FIXED(_XMP_template_t **template, int dim, ...);
extern _XMP_nodes_t *_XMP_create_nodes_by_template_ref(_XMP_template_t *ref_template, int *shrink,
                                                       long long *ref_lower, long long *ref_upper, long long *ref_stride);

extern void _XMP_calc_template_size(_XMP_template_t *t);
extern void _XMP_init_template_chunk(_XMP_template_t *template, _XMP_nodes_t *nodes);
extern void _XMP_finalize_template(_XMP_template_t *template);
extern int _XMP_calc_template_owner_SCALAR(_XMP_template_t *ref_template, int dim_index, long long ref_index);
extern int _XMP_calc_template_par_triplet(_XMP_template_t *template, int template_index, int nodes_rank,
                                          int *template_lower, int *template_upper, int *template_stride);

void _XMP_dist_template_DUPLICATION(_XMP_template_t *template, int template_index);
void _XMP_dist_template_BLOCK(_XMP_template_t *template, int template_index, int nodes_index);
void _XMP_dist_template_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index) ;
void _XMP_dist_template_BLOCK_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index, unsigned long long width);
void _XMP_dist_template_GBLOCK(_XMP_template_t *template, int template_index, int nodes_index,
			       int *mapping_array, int *temp0);

// xmp_util.c
extern unsigned long long _XMP_get_on_ref_id(void);
extern void *_XMP_alloc(size_t size);
extern void _XMP_free(void *p);
extern void _XMP_fatal(char *msg);
extern void _XMP_fatal_nomsg();
extern void _XMP_unexpected_error(void);

// xmp_world.c
extern int _XMP_world_size;
extern int _XMP_world_rank;
extern void *_XMP_world_nodes;

extern void _XMP_init_world(int *argc, char ***argv);
extern void _XMP_finalize_world(void);
extern int _XMP_split_world_by_color(int color);

#ifdef _XMP_XACC
extern void _XMP_reflect_do_gpu(_XMP_array_t *array_desc);
extern void _XMP_reflect_init_gpu(void *acc_addr, _XMP_array_t *array_desc);
extern int _XMP_get_owner_pos(_XMP_array_t *a, int dim, int index);
extern void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
extern void _XMP_reduce_gpu_CLAUSE(void *data_addr, int count, int datatype, int op);
#endif

// ----- libxmp_threads ----------------------------------------------
// xmp_threads_runtime.c
extern void _XMP_threads_init(void);
extern void _XMP_threads_finalize(void);

#ifdef __cplusplus
}
#endif

// ----- for coarray & post/wait -------------------
#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA)
#define _XMP_DEFAULT_ONESIDED_HEAP_SIZE   "27M"
#define _XMP_DEFAULT_ONESIDED_STRIDE_SIZE "5M"
/* Momo:
   Each process allocates 32MByte (27M+5M), and the test program uses up to 16 process
   on a single node. Therefore the node needs 512MByte (32M*16) for coarray operation. 
*/

#define _XMP_COARRAY_QUEUE_INITIAL_SIZE 32         /**< This value is trial */
#define _XMP_COARRAY_QUEUE_INCREMENT_RAITO (1.5)   /**< This value is trial */
#define _XMP_GASNET_COARRAY_SHIFT_QUEUE_INITIAL_SIZE _XMP_COARRAY_QUEUE_INITIAL_SIZE        /** The same vaule may be good. */
#define _XMP_GASNET_COARRAY_SHIFT_QUEUE_INCREMENT_RAITO _XMP_COARRAY_QUEUE_INCREMENT_RAITO  /** The same vaule may be good. */

#define _XMP_POSTREQ_TABLE_INITIAL_SIZE 32         /**< This value is trial */
#define _XMP_POSTREQ_TABLE_INCREMENT_RATIO (1.5)   /**< This value is trial */
extern size_t _XMP_get_offset(const _XMP_array_section_t *, const int);
extern void _XMP_post_wait_initialize();
#define _XMP_PACK   0
#define _XMP_UNPACK 1
#define _XMP_MPUT   2
extern void _XMP_stride_memcpy_1dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_2dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_3dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_4dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_5dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_6dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
extern void _XMP_stride_memcpy_7dim(char *, const char *, const _XMP_array_section_t *, size_t, int);
#endif

#ifdef _XMP_GASNET
#include <gasnet.h>
#define _XMP_GASNET_STRIDE_INIT_SIZE 16
#define _XMP_GASNET_STRIDE_BLK       16
#define _XMP_GASNET_ALIGNMENT        8

#define GASNET_BARRIER() do {  \
	gasnet_barrier_notify(0,GASNET_BARRIERFLAG_ANONYMOUS); \
	gasnet_barrier_wait(0,GASNET_BARRIERFLAG_ANONYMOUS);   \
  } while (0)

extern void _XMP_gasnet_malloc_do(_XMP_coarray_t *, void **, const size_t);
extern void _XMP_gasnet_initialize(int, char**, const size_t, const size_t);
extern void _XMP_gasnet_finalize(const int);
extern void _XMP_gasnet_put(const int, const int, const int, const int, const int, const _XMP_array_section_t*, const _XMP_array_section_t*, 
			    const _XMP_coarray_t*, const void*, const size_t, const size_t);
extern void _XMP_gasnet_get(const int, const int, const int, const int, const int, const _XMP_array_section_t*, const _XMP_array_section_t*,
                            const _XMP_coarray_t*, const void*, const size_t, const size_t);
extern void _XMP_gasnet_sync_all();
extern void _XMP_gasnet_sync_memory();
extern void _xmp_gasnet_post_wait_initialize();
extern void _xmp_gasnet_post(const int, const int);
extern void _xmp_gasnet_wait_noargs();
extern void _xmp_gasnet_wait_node(const int);
extern void _xmp_gasnet_wait(const int, const int);
extern void _XMP_gasnet_coarray_lastly_deallocate();
#endif

#ifdef _XMP_FJRDMA
#define _XMP_FLAG_NIC (FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC1 | FJMPI_RDMA_IMMEDIATE_RETURN)
#define _XMP_SEND_NIC FJMPI_RDMA_LOCAL_NIC0
#define _XMP_POSTREQ_SEND_NIC FJMPI_RDMA_LOCAL_NIC2
#define _XMP_POSTREQ_RECV_NIC FJMPI_RDMA_LOCAL_NIC3
#define _XMP_POSTREQ_NIC_FLAG (FJMPI_RDMA_LOCAL_NIC2 | FJMPI_RDMA_REMOTE_NIC3 | FJMPI_RDMA_REMOTE_NOTICE)
#define _XMP_TEMP_MEMID 0
#define _XMP_POSTREQ_ID 1
#define _XMP_FJRDMA_INTERVAL 8192
#define _XMP_FJRDMA_MAX_PROCS 82944

#include <mpi-ext.h>
extern void _XMP_fjrdma_initialize(int, char**);
extern void _XMP_fjrdma_finalize();
extern void _XMP_fjrdma_sync_memory();
extern void _XMP_fjrdma_sync_all();
extern void _XMP_fjrdma_malloc_do(_XMP_coarray_t *, void **, const size_t);
extern void _XMP_fjrdma_put(const int, const int, const int, const int, const int, const _XMP_array_section_t *,  
			    const _XMP_array_section_t *, const _XMP_coarray_t *, void *, const _XMP_coarray_t *, const int, const int);
extern void _XMP_fjrdma_get(const int, const int, const int, const int, const int, const _XMP_array_section_t *, 
			    const _XMP_array_section_t *, const _XMP_coarray_t *, void *, const _XMP_coarray_t *, const int, const int);
extern void _XMP_fjrdma_shortcut_put(const int, const uint64_t, const uint64_t, const _XMP_coarray_t *, const _XMP_coarray_t *, const size_t);
extern void _XMP_fjrdma_shortcut_get(const int, const uint64_t, const uint64_t, const _XMP_coarray_t *, const _XMP_coarray_t *, const size_t);
extern void _xmp_fjrdma_post_wait_initialize();
extern void _xmp_fjrdma_post(const int, const int);
extern void _xmp_fjrdma_wait_noargs();
extern void _xmp_fjrdma_wait_node(const int);
extern void _xmp_fjrdma_wait(const int, const int);
extern void _XMP_fjrdma_coarray_lastly_deallocate();
#endif

#ifdef _XMP_TIMING
extern double t0, t1;
#define _XMP_TSTART(t0)  ((t0) = MPI_Wtime())
#define _XMP_TEND(t, t0) ((t) = (t) + MPI_Wtime() - (t0))
#define _XMP_TEND2(t, tt, t0) { double _XMP_TMP = MPI_Wtime(); \
                                (t) = (t) + _XMP_TMP - (t0); \
                                (tt) = (tt) + _XMP_TMP - (t0); }

struct _XMPTIMING
{ double t_mem, t_copy, t_comm, t_sched, t_wait, t_misc,
    tdim[_XMP_N_MAX_DIM],
    tdim_mem[_XMP_N_MAX_DIM],
    tdim_copy[_XMP_N_MAX_DIM],
    tdim_comm[_XMP_N_MAX_DIM],
    tdim_sched[_XMP_N_MAX_DIM],
    tdim_wait[_XMP_N_MAX_DIM],
    tdim_misc[_XMP_N_MAX_DIM];
} xmptiming_;

#else

#define _XMP_TSTART(t0)
#define _XMP_TEND(t, t0)
#define _XMP_TEND2(t, tt, t0)

#endif

#ifdef _XMP_TCA
#define TCA_CHECK(tca_call) do { \
  int status = tca_call;         \
  if(status != TCA_SUCCESS) {    \
  if(status == TCA_ERROR_INVALID_VALUE) {                 \
  fprintf(stderr,"(TCA) error TCA API, INVALID_VALUE\n"); \
  exit(-1);                                               \
  }else if(status == TCA_ERROR_OUT_OF_MEMORY){            \
  fprintf(stderr,"(TCA) error TCA API, OUT_OF_MEMORY\n"); \
  exit(-1);                                               \
  }else if(status == TCA_ERROR_NOT_SUPPORTED){            \
  fprintf(stderr,"(TCA) error TCA API, NOT_SUPPORTED\n"); \
  exit(-1);                                               \
  }else{                                                  \
  fprintf(stderr,"(TCA) error TCA API, UNKWON\n");        \
  exit(-1); \
  }         \
  }         \
  }while (0)
#endif

#ifdef _XMP_GASNET
#define _XMP_LOCK_CHUNK 8       // for lock

typedef struct xmp_lock{
  int islocked;
  gasnet_hsl_t  hsl;
  int  wait_size;   /* How many elements in wait_list */
  int  wait_head;   /* Index for next dequeue */
  int  wait_tail;   /* Index for next enqueue */
  int *wait_list;   /* Circular queue of waiting threads */
} xmp_gasnet_lock_t;

typedef enum {
  _XMP_LOCKSTATE_WAITING = 300,    /* waiting for a reply */
  _XMP_LOCKSTATE_GRANTED,    /* lock attempt granted */
  _XMP_LOCKSTATE_FAILED,     /* lock attempt failed */
  _XMP_LOCKSTATE_HANDOFF,    /* unlock op complete--handoff in progress */
  _XMP_LOCKSTATE_DONE        /* unlock op complete */
} xmp_gasnet_lock_state_t;

extern void _xmp_gasnet_lock(_XMP_coarray_t*, int, int);
extern void _xmp_gasnet_unlock(_XMP_coarray_t*, int, int);
extern void _xmp_gasnet_do_lock(int, xmp_gasnet_lock_t*, int*);
extern void _xmp_gasnet_lock_initialize(xmp_gasnet_lock_t*, int);
extern void _xmp_gasnet_do_unlock(int, xmp_gasnet_lock_t*, int*, int*);
extern void _xmp_gasnet_do_lockhandoff(int);
extern void _xmp_gasnet_unpack(gasnet_token_t, const char*, const size_t, 
			       const int, const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_using_buf(gasnet_token_t, const int, const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_reply(gasnet_token_t, const int);
extern void _xmp_gasnet_pack(gasnet_token_t, const char*, const size_t, 
			     const int, const int, const int, const size_t, const int, const int);
extern void _xmp_gasnet_unpack_get_reply(gasnet_token_t, char *, size_t, const int, const int);
extern void _XMP_pack_coarray(char*, const char*, const int, const _XMP_array_section_t*);
extern void _XMP_unpack_coarray(char*, const int, const char*, const _XMP_array_section_t*, const int);

/* Every handler function needs a uniqe number between 200-255.   
 * The Active Message library reserves ID's 1-199 for itself: client libs must
 * use IDs between 200-255. 
 */
#define _XMP_GASNET_LOCK_REQUEST               200
#define _XMP_GASNET_SETLOCKSTATE               201
#define _XMP_GASNET_UNLOCK_REQUEST             202
#define _XMP_GASNET_LOCKHANDOFF                203
#define _XMP_GASNET_POSTREQ                    204
#define _XMP_GASNET_UNPACK                     205
#define _XMP_GASNET_UNPACK_USING_BUF           206
#define _XMP_GASNET_UNPACK_REPLY               207
#define _XMP_GASNET_PACK                       208
#define _XMP_GASNET_UNPACK_GET_REPLY           209
#define _XMP_GASNET_PACK_USGIN_BUF             210
#define _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF 211
#define _XMP_GASNET_PACK_GET_HANDLER           212
#define _XMP_GASNET_UNPACK_GET_REPLY_NONC      213
extern void _xmp_gasnet_lock_request(gasnet_token_t, int, uint32_t, uint32_t);
extern void _xmp_gasnet_setlockstate(gasnet_token_t, int);
extern void _xmp_gasnet_do_setlockstate(int);
extern void _xmp_gasnet_unlock_request(gasnet_token_t, int, uint32_t, uint32_t);
extern void _xmp_gasnet_lockhandoff(gasnet_token_t, int);
extern void _xmp_gasnet_postreq(gasnet_token_t, const int, const int);
extern void _xmp_gasnet_pack_using_buf(gasnet_token_t, const char*, const size_t,
				       const int, const int, const int, const int);
extern void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t);
extern void _xmp_gasnet_pack_get(gasnet_token_t, const char*, const size_t, const int,
				 const int, const int, const int, const size_t, const int, const int);
extern void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t, char *, size_t, const int, const int, const int);

/*  Macros for splitting and reassembling 64-bit quantities  */
#define HIWORD(arg)     ((uint32_t) (((uint64_t)(arg)) >> 32))
#if PLATFORM_COMPILER_CRAY || PLATFORM_COMPILER_INTEL
/* workaround irritating warning #69: Integer conversion resulted in truncation.                                 
   which happens whenever Cray C or Intel C sees address-of passed to SEND_PTR                                   
*/
#define LOWORD(arg)     ((uint32_t) (((uint64_t)(arg)) & 0xFFFFFFFF))
#else
#define LOWORD(arg)     ((uint32_t) ((uint64_t)(arg)))
#endif
#define UPCRI_MAKEWORD(hi,lo) (   (((uint64_t)(hi)) << 32) \
                                  | (((uint64_t)(lo)) & 0xffffffff) )

/* These macros are referred from upcr.h of Berkeley UPC */
/*                                                                                                                 
 * Network polling                                                                                                 
 * ===============                                                                                                 
 *                                                                                                                 
 * The upcr_poll() function explicitly causes the runtime to attempt to make                                       
 * progress on any network requests that may be pending.  While many other                                         
 * runtime functions implicitly do this as well (i.e. most of those which call                                     
 * the network layer) this function may be useful in cases where a large amount                                    
 * of time has elapsed since the last runtime call (e.g. if a great deal of                                        
 * application-level calculation is taking place).  This function may also be                                      
 * indirectly when a upc_fence is used.                                                                            
 *                                                                                                                 
 * upcr_poll() also provides a null strict reference, corresponding to upc_fence in the                            
 * UPC memory model.                                                                                               
 * DOB: we should really rename upcr_poll to upcr_fence, but this would break                                      
 * compatibility between old runtimes and new translators, so until the next                                       
 * major runtime interface upgrade, (b)upc_poll expands to upcr_poll_nofence,                                      
 * which polls without the overhead of strict memory fences.                                                       
 */

/* Bug 2996 - upcr_poll_nofence should also yield in polite mode to get                                          
 * resonable performance from a spin-loop constructed according to our                                           
 * recommendations.                                                                                              
 * The bug was first seen w/ smp-conduit, but when using a network we                                            
 * cannot claim to know if gasnet_AMPoll() is going to yield or not.                                             
 * With an RMDA-capable transport one actually could expect that it                                              
 * would NOT.                                                                                                    
 */
#define upcr_poll_nofence() do {        \
    gasnet_AMPoll();			\
  } while (0)
#if GASNET_CONDUIT_SMP && !UPCRI_UPC_PTHREADS && !GASNET_PSHM
/* in the special case of exactly one UPC thread, nothing is required for                                        
 * correctness of fence (poll is likely a no-op as well, included solely                                         
 * for tracing purposes)                                                                                         
 */
#define upcr_poll() upcr_poll_nofence()
#else
/* in all other cases, a fence needs to act as a null strict reference,                                          
 * which means we need an architectural membar & optimization barrier to                                         
 * ensure that surrounding relaxed shared and local operations are not                                           
 * reordered in any way across this point (which could be visible if other                                       
 * CPU's or an RDMA enabled NIC are modifying memory via strict operations).                                     
 * We need both an WMB and RMB within the fence, but it doesn't actually matter                                  
 * whether they come before or after the optional poll (which is added as                                        
 * a performance optimization, to help ensure progress in spin-loops using fence).                               
 * We combine them in a call to gasnett_local_mb(), which on some architectures                                  
 * can be slightly more efficient than WMB and RMB called in sequence.                                           
 */
#define upcr_poll() do {              \
    gasnett_local_mb();               \
    upcr_poll_nofence();              \
  } while (0)
#endif

#define _xmp_lock_t xmp_gasnet_lock_t

extern void _xmp_lock(_XMP_coarray_t*, int, int);
extern void _xmp_unlock(_XMP_coarray_t*, int, int);
extern void _xmp_lock_initialize(_xmp_lock_t*, int);

#endif // _XMP_GASNET

#endif // _XMP_INTERNAL
