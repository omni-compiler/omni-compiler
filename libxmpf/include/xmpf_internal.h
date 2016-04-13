#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif 

#include "mpi.h"
#include "xmp.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* #define LBOUND 0 */
/* #define UBOUND 1 */
/* #define STRIDE 2 */

#define XMP_OBJ_REF_NODES 1
#define XMP_OBJ_REF_TEMPL 2

#define REF_OFFSET arg0
#define REF_LBOUND arg0
#define REF_INDEX  arg1
#define REF_UBOUND arg1
#define REF_STRIDE arg2

#define SUBSCRIPT_ASTERISK 0
#define SUBSCRIPT_SCALAR   1
#define SUBSCRIPT_TRIPLET  2
#define SUBSCRIPT_NOLB     3
#define SUBSCRIPT_NOUB     4
#define SUBSCRIPT_NOLBUB   5
#define SUBSCRIPT_NONE     6

#define MAX_RANK 31

typedef struct _XMP_object_ref_type {
  int ref_kind; 
  _XMP_template_t *t_desc;
  _XMP_nodes_t *n_desc;
    
  int ndims;
/*   int *offset; */
/*   int *index; */
  int *arg0;
  int *arg1;
  int *arg2;
  int *subscript_type;
} _XMP_object_ref_t;


/* typedef struct _XMP_object_ref_type2 { */
/*   int ref_kind;  */
/*   _XMP_template_t *t_desc; */
/*   _XMP_nodes_t *n_desc; */
    
/*   int ndims; */
/*   int *lb; */
/*   int *ub; */
/*   int *st; */
/* } _XMP_object_ref_t2; */


/* From xmpf_index.c */
void _XMP_L2G(int local_idx, long long int *global_idx,
	      _XMP_template_t *template, int template_index);
void _XMP_G2L(long long int global_idx,int *local_idx,
	      _XMP_template_t *template, int template_index);

/* From xmpf_misc.c */
void xmpf_dbg_printf(char *fmt, ...);
size_t _XMP_get_datatype_size(int datatype);
void xmpf_finalize_all__(void);
void xmpf_finalize_each__(void);


/* From xmpf_gcomm.c */
int _XMPF_get_owner_pos_BLOCK(_XMP_array_t *a, int dim, int index);
_Bool _XMP_is_entire(_XMP_object_ref_t *rp);

// From xmp_world.c
extern int _XMP_world_size;
extern int _XMP_world_rank;

/* From xmp_align.c */
void xmpf_array_alloc__(_XMP_array_t **a_desc, int *n_dim, int *type, _XMP_template_t **t_desc);
void xmpf_array_dealloc__(_XMP_array_t **a_desc);
void xmpf_align_info__(_XMP_array_t **a_desc, int *a_idx,
		       int *lower, int *upper, int *t_idx, int *off);
void xmpf_array_set_local_array__(_XMP_array_t **a_desc, void *array_addr, int is_coarray);
void _XMP_finalize_array_desc(_XMP_array_t *array);
void _XMP_align_array_NOT_ALIGNED(_XMP_array_t *array, int array_index);
void _XMP_align_array_DUPLICATION(_XMP_array_t *array, int array_index, int template_index,
                                  long long align_subscript);
void _XMP_align_array_BLOCK(_XMP_array_t *array, int array_index, int template_index,
                            long long align_subscript, int *temp0);
void _XMP_align_array_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                             long long align_subscript, int *temp0);
void _XMP_align_array_BLOCK_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                                   long long align_subscript, int *temp0);
void _XMP_align_array_GBLOCK(_XMP_array_t *array, int array_index, int template_index,
			     long long align_subscript, int *temp0);
void _XMP_init_array_nodes(_XMP_array_t *array);
void _XMP_calc_array_dim_elmts(_XMP_array_t *array, int array_index);

/* From xmp_lib.c */
void xmp_barrier(void);


/* From xmp_loop.c */
void _XMP_sched_loop_template_DUPLICATION(int ser_init, int ser_cond, int ser_step,
                                          int *par_init, int *par_cond, int *par_step,
                                          _XMP_template_t *template, int template_index);

void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
                                    int *par_init, int *par_cond, int *par_step,
                                    _XMP_template_t *template, int template_index);

void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                     int *par_init, int *par_cond, int *par_step,
                                     _XMP_template_t *template, int template_index);

void _XMP_sched_loop_template_BLOCK_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                           int *par_init, int *par_cond, int *par_step,
                                           _XMP_template_t *template, int template_index);

void _XMP_sched_loop_template_GBLOCK(int ser_init, int ser_cond, int ser_step,
				     int *par_init, int *par_cond, int *par_step,
				     _XMP_template_t *template, int template_index);

/* From xmp_reduce.c */
void _XMP_reduce_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);

/* From xmp_shadow.c */
void _XMP_create_shadow_comm(_XMP_array_t *array, int array_index);
void _XMP_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr,
                             _XMP_array_t *array_desc, int array_index);
void _XMP_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
                                 void *lo_send_buffer, void *hi_send_buffer,
                                 _XMP_array_t *array_desc, int array_index);
void _XMP_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr,
                               _XMP_array_t *array_desc, int array_index);


/* From xmp_barrier.c */
void _XMP_barrier_NODES_ENTIRE(_XMP_nodes_t *nodes);

/* From xmp_gmove.c */
int _XMP_calc_gmove_array_owner_linear_rank_SCALAR(_XMP_array_t *array, int *ref_index);
void _XMP_gmove_bcast_SCALAR(void *dst_addr, void *src_addr,
			     size_t type_size, int root_rank);
unsigned long long _XMP_gmove_bcast_ARRAY(void *dst_addr, int dst_dim,
					  int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
					  void *src_addr, int src_dim,
					  int *src_l, int *src_u, int *src_s, unsigned long long *src_d,
					  int type, size_t type_size, int root_rank);
void _XMP_gtol_array_ref_triplet(_XMP_array_t *array,
				 int dim_index, int *lower, int *upper, int *stride);
int _XMP_check_gmove_array_ref_inclusion_SCALAR(_XMP_array_t *array, int array_index,
						int ref_index);
void _XMP_gmove_localcopy_ARRAY(int type, int type_size,
				void *dst_addr, int dst_dim,
				int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
				void *src_addr, int src_dim,
				int *src_l, int *src_u, int *src_s, unsigned long long *src_d);
int _XMP_calc_global_index_HOMECOPY(_XMP_array_t *dst_array, int dst_dim_index,
				    int *dst_l, int *dst_u, int *dst_s,
				    int *src_l, int *src_u, int *src_s);
int _XMP_calc_global_index_BCAST(int dst_dim, int *dst_l, int *dst_u, int *dst_s,
				 _XMP_array_t *src_array, int *src_array_nodes_ref, int *src_l, int *src_u, int *src_s);
void _XMP_sendrecv_ARRAY(int type, int type_size, MPI_Datatype *mpi_datatype,
                         _XMP_array_t *dst_array, int *dst_array_nodes_ref,
                         int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
                         _XMP_array_t *src_array, int *src_array_nodes_ref,
                         int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc);
void _XMP_gmove_BCAST_GSCALAR(void *dst_addr, void *src_addr, _XMP_array_t *array, int ref_index[]);
void _XMP_gmove_SENDRECV_GSCALAR(void *dst_addr, void *src_addr,
				 _XMP_array_t *dst_array, _XMP_array_t *src_array,
				 int dst_ref_index[], int src_ref_index[]);

extern void _XMP_gmove_array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long *src_d, int mode);
extern void _XMP_gmove_inout_scalar(void *scalar, _XMP_gmv_desc_t *gmv_desc, int rdma_type);

/* From xmp_runtime.c */
void _XMP_init(int argc, char** argv);
void _XMP_finalize(int);

/* From xmp_reflect.c */
extern void _XMP_set_reflect__(_XMP_array_t *a_desc, int dim, int lwidth, int uwidth, int is_periodic);
extern void _XMP_reflect__(_XMP_array_t *a_desc);
extern void _XMP_wait_async__(int async_id);
extern void _XMP_reflect_async__(_XMP_array_t *a_desc, int async_id);
extern _XMP_async_comm_t* _XMP_get_current_async();
extern _XMP_async_comm_t* _XMP_get_async(int);
extern void xmpc_init_async(int);
extern void xmpc_start_async();

/* From xmpf_pack.c */
void _XMPF_pack_array(void *buffer, void *src, int array_type, size_t array_type_size,
		      int array_dim, int *l, int *u, int *s, unsigned long long *d);
void _XMPF_unpack_array(void *dst, void *buffer, int array_type, size_t array_type_size,
			int array_dim, int *l, int *u, int *s, unsigned long long *d);

/* From xmp_coarray.c */
extern void _XMP_gasnet_not_continuous_put();
extern void _XMP_gasnet_continuous_put();
extern void _XMP_gasnet_not_continuous_get();
extern void _XMP_gasnet_continuous_get();
extern void _XMP_coarray_malloc_info_1(const long, const size_t);
//extern void _XMP_coarray_malloc_info_2(const int, const int, const size_t);
//extern void _XMP_coarray_malloc_info_3(const int, const int, const int, const size_t);
//extern void _XMP_coarray_malloc_info_4(const int, const int, const int, const int, const size_t);
//extern void _XMP_coarray_malloc_info_5(const int, const int, const int, const int, const int, const size_t);
//extern void _XMP_coarray_malloc_info_6(const int, const int, const int, const int, const int, const int, const size_t);
//extern void _XMP_coarray_malloc_info_7(const int, const int, const int, const int, const int, const int, const int, const size_t);

extern void _XMP_coarray_malloc_image_info_1();
//extern void _XMP_coarray_malloc_image_info_2(const int);
//extern void _XMP_coarray_malloc_image_info_3(const int, const int);
//extern void _XMP_coarray_malloc_image_info_4(const int, const int, const int);
//extern void _XMP_coarray_malloc_image_info_5(const int, const int, const int, const int);
//extern void _XMP_coarray_malloc_image_info_6(const int, const int, const int, const int, const int);
//extern void _XMP_coarray_malloc_image_info_7(const int, const int, const int, const int, const int, const int);

//extern void _XMP_coarray_malloc_do_f(void **, void *);
extern void _XMP_coarray_malloc_do(void **, void *);
extern void _XMP_coarray_regmem_do(void **, void *);

extern void _XMP_coarray_rdma_coarray_set_1(const long, const long, const long);
//extern void _XMP_coarray_rdma_coarray_set_2(const int, const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_coarray_set_3(const int, const int, const int, const int, const int, const int,
//					    const int, const int, const int);
//extern void _XMP_coarray_rdma_coarray_set_4(const int, const int, const int, const int, const int, const int,
//                                            const int, const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_coarray_set_5(const int, const int, const int, const int, const int, const int,
//                                            const int, const int, const int, const int, const int, const int,
//					    const int, const int, const int);
//extern void _XMP_coarray_rdma_coarray_set_6(const int, const int, const int, const int, const int, const int,
//                                            const int, const int, const int, const int, const int, const int,
//					    const int, const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_coarray_set_7(const int, const int, const int, const int, const int, const int,
//                                            const int, const int, const int, const int, const int, const int,
//                                            const int, const int, const int, const int, const int, const int,
//					    const int, const int, const int);

extern void _XMP_coarray_rdma_array_set_1(const long, const long, const long, const long, const long);
//extern void _XMP_coarray_rdma_array_set_2(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_array_set_3(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_array_set_4(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_array_set_5(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_array_set_6(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_array_set_7(const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int,
//					  const int, const int, const int, const int, const int);

extern void _XMP_coarray_rdma_image_set_1(const int);
//extern void _XMP_coarray_rdma_image_set_2(const int, const int);
//extern void _XMP_coarray_rdma_image_set_3(const int, const int, const int);
//extern void _XMP_coarray_rdma_image_set_4(const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_image_set_5(const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_image_set_6(const int, const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_image_set_7(const int, const int, const int, const int, const int, const int, const int);

//extern void _XMP_coarray_rdma_do_f(const int*, const void*, const void*, const void*);
extern void _XMP_coarray_rdma_do(const int, void*, void*, void *);
//extern size_t get_offset(const void *, const int);

/* libxmp/include/xmp_func_decl.h */
extern void _XMP_coarray_shortcut_put(const int, void*, const void*, const long, const long, const long, const long);
//extern void _XMP_coarray_shortcut_put_f(const int*, void*, const void*, const long*, const long*, const long*, const long*);
extern void _XMP_coarray_shortcut_get(const int, void*, const void*, const long, const long, const long, const long);
//extern void _XMP_coarray_shortcut_get_f(const int*, void*, const void*, const long*, const long*, const long*, const long*);


/******************************************\
    COARRAY
\******************************************/
#include "xmpf_internal_coarray.h"
