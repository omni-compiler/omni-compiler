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


/* From xmpf_gcomm.c */
int _XMPF_get_owner_pos_BLOCK(_XMP_array_t *a, int dim, int index);
_Bool _XMP_is_entire(_XMP_object_ref_t *rp);

// From xmp_world.c
extern int _XMP_world_size;
extern int _XMP_world_rank;

/* From xmp_align.c */
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

extern void _XMP_gmove_array_array_common(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp, int *dst_l, int *dst_u, int *dst_s, unsigned long long  *dst_d, int *src_l, int *src_u, int *src_s, unsigned long long *src_d);

/* From xmp_runtime.c */
void _XMP_init(int argc, char** argv);
void _XMP_finalize(int);

/* From xmp_reflect.c */
void _XMP_set_reflect__(_XMP_array_t *a_desc, int dim, int lwidth, int uwidth, int is_periodic);
void _XMP_reflect__(_XMP_array_t *a_desc);
void _XMP_wait_async__(int async_id);
void _XMP_reflect_async__(_XMP_array_t *a_desc, int async_id);

/* From xmp_coarray.c */
extern int xmpf_get_descr_id_(char *baseAddr);
extern void _XMP_gasnet_not_continuous_put();
extern void _XMP_gasnet_continuous_put();
extern void _XMP_gasnet_not_continuous_get();
extern void _XMP_gasnet_continuous_get();
extern void _XMP_coarray_malloc_info_1(const int, const size_t);
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

extern void _XMP_coarray_rdma_coarray_set_1(const int, const int, const int);
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

extern void _XMP_coarray_rdma_array_set_1(const int, const int, const int, const int, const int);
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

extern void _XMP_coarray_rdma_node_set_1(const int);
//extern void _XMP_coarray_rdma_node_set_2(const int, const int);
//extern void _XMP_coarray_rdma_node_set_3(const int, const int, const int);
//extern void _XMP_coarray_rdma_node_set_4(const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_node_set_5(const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_node_set_6(const int, const int, const int, const int, const int, const int);
//extern void _XMP_coarray_rdma_node_set_7(const int, const int, const int, const int, const int, const int, const int);

//extern void _XMP_coarray_rdma_do_f(const int*, const void*, const void*, const void*);
extern void _XMP_coarray_rdma_do(const int, const void*, const void*, const void *);
//extern void _XMP_coarray_sync_all();
//extern void _XMP_coarray_sync_memory();
//extern size_t get_offset(const void *, const int);
//extern void _XMP_coarray_shortcut_put(const int, const void*, const void*, const size_t, const size_t, const size_t);
//extern void _XMP_coarray_shortcut_put_f(const int*, const void*, const void*, const size_t*, const size_t*, const size_t*);
//extern void _XMP_coarray_shortcut_get(const int, const void*, const void*, const size_t, const size_t, const size_t);
//extern void _XMP_coarray_shortcut_get_f(const int*, const void*, const void*, const size_t*, const size_t*, const size_t*);


/******************************************\
    COARRAY
\******************************************/
#define BOOL   int
#define TRUE   1
#define FALSE  0

//////////////////////////// TEST TEST TEST
//#define _XMP_COARRAY_FJRDMA
//////////////////////////// TEST TEST TEST

#ifdef _XMP_COARRAY_FJRDMA
#  define BOUNDARY_BYTE 4
#else
#  define BOUNDARY_BYTE 1
#endif

#define ROUND_UP(n,p)         ((((n)-1)/(p)+1)*(p))
#define ROUND_UP_BOUNDARY(n)  ROUND_UP((n),BOUNDARY_BYTE)

/*-- parameters --*/
#define DESCR_ID_MAX   250
#define SMALL_WORK_SIZE_KB  10
extern int _XMP_boundaryByte;     // communication boundary (bytes)

/*-- codes --*/
#define COARRAY_GET_CODE  700
#define COARRAY_PUT_CODE  701

/* xmpf_coarray.c */
extern int _XMPF_coarrayMsg;          // default: debug message off
extern void xmpf_coarray_msg_(int *sw);

extern char *_XMPF_errmsg;   // to answer ERRMSG argument in Fortran
extern void xmpf_copy_errmsg_(char *errmsg, int *msglen);

extern int _XMPF_nowInTask(void);   // for restriction check
extern void _XMPF_checkIfInTask(char *msgopt);   // restriction check
extern void _XMPF_coarrayDebugPrint(char *format, ...);

extern int _XMPF_get_coarrayElement(int serno);
extern char *_XMPF_get_coarrayDesc(int serno);
extern int _XMPF_get_coarrayStart(int serno, char *baseAddr);

extern void xmpf_coarray_malloc_(int *serno, char **pointer, int *count, int *element);

extern void xmpf_sync_all_nostat_(void);
extern void xmpf_sync_all_stat_(int *stat, char *msg, int *msglen);

extern void xmpf_sync_memory_nostat_(void);
extern void xmpf_sync_memory_stat_(int *stat, char *msg, int *msglen);

extern void xmpf_sync_image_nostat_(int *image);
extern void xmpf_sync_image_stat_(int *image,
                                  int *stat, char *msg, int *msglen);
extern void xmpf_sync_images_nostat_(int *images, int *size);
extern void xmpf_sync_images_stat_(int *images, int *size,
                                   int *stat, char *msg, int *msglen);
extern void xmpf_sync_allimages_nostat_(void);
extern void xmpf_sync_allimages_stat_(int *stat, char *msg, int *msglen);

extern int xmpf_num_nodes_(void);
extern int xmpf_node_num_(void);

/* xmpf_coarray_put.c */
extern void xmpf_coarray_put_scalar_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition);
extern void xmpf_coarray_put_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *rhs, int *condition,
                                    int *rank, ...);
extern void xmpf_coarray_put_spread_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *rhs, int *condition,
                                     int *rank, ...);

/* xmpf_coarray_get.c */
extern void xmpf_coarray_get_scalar_(int *serno, char *baseAddr, int *element,
                                     int *coindex, char *result);
extern void xmpf_coarray_get_array_(int *serno, char *baseAddr, int *element,
                                    int *coindex, char *result, int *rank, ...);

