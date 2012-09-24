#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define XMPF_MAX_DIM  7

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

typedef struct _XMP_gmv_desc_type {

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

} _XMP_gmv_desc_t;


/* From xmpf_index.c */
void _XMP_L2G(int local_idx, long long int *global_idx,
	      _XMP_template_t *template, int template_index);
void _XMP_G2L(long long int global_idx,int *local_idx,
	      _XMP_template_t *template, int template_index);

/* From xmpf_misc.c */
void xmpf_dbg_printf(char *fmt, ...);
size_t _XMP_get_datatype_size(int datatype);

/* From xmp_align.c */
_XMP_template_t *_XMP_create_template_desc(int dim, _Bool is_fixed);
void _XMP_calc_template_size(_XMP_template_t *t);
void _XMP_dist_template_DUPLICATION(_XMP_template_t *template, int template_index);
void _XMP_dist_template_BLOCK(_XMP_template_t *template, int template_index, int nodes_index);
void _XMP_dist_template_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index);
void _XMP_dist_template_BLOCK_CYCLIC(_XMP_template_t *template, int template_index, int nodes_index, unsigned long long width);
void _XMP_init_template_chunk(_XMP_template_t *template, _XMP_nodes_t *nodes);

void _XMP_align_array_NOT_ALIGNED(_XMP_array_t *array, int array_index);
void _XMP_align_array_DUPLICATION(_XMP_array_t *array, int array_index, int template_index,
                                  long long align_subscript);
void _XMP_align_array_BLOCK(_XMP_array_t *array, int array_index, int template_index,
                            long long align_subscript, int *temp0);
void _XMP_align_array_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                             long long align_subscript, int *temp0);
void _XMP_align_array_BLOCK_CYCLIC(_XMP_array_t *array, int array_index, int template_index,
                                   long long align_subscript, int *temp0);
void _XMP_init_array_nodes(_XMP_array_t *array);
void _XMP_calc_array_dim_elmts(_XMP_array_t *array, int array_index);


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


/* From xmp_reduce.c */
void _XMP_reduce_CLAUSE(void *data_addr, int count, int datatype, int op);

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
void _XMP_sendrecv_ARRAY(unsigned long long gmove_total_elmts,
			 int type, int type_size, MPI_Datatype *mpi_datatype,
			 _XMP_array_t *dst_array, int *dst_array_nodes_ref,
			 int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
			 _XMP_array_t *src_array, int *src_array_nodes_ref,
			 int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc);

/* From xmp_runtime.c */
void _XMP_init(int argc, char** argv);
void _XMP_finalize(void);
