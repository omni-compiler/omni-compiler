/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_RUNTIME_FUNC_DECL
#define _XMP_RUNTIME_FUNC_DECL

#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>

// ----- libxml
// xmp_align.c
extern void _XMP_init_array_desc(void **array, void *template, int dim, int type, size_t type_size, ...);
extern void _XMP_finalize_array_desc(void *array);
extern void _XMP_align_array_NOT_ALIGNED(void *array, int array_index);
extern void _XMP_align_array_DUPLICATION(void *array, int array_index, int template_index, long long align_subscript);
extern void _XMP_align_array_BLOCK(void *array, int array_index, int template_index, long long align_subscript, int *temp0);
extern void _XMP_align_array_CYCLIC(void *array, int array_index, int template_index, long long align_subscript, int *temp0);
extern void _XMP_alloc_array(void **array_addr, void *array_desc, ...);
extern void _XMP_init_array_alloc_params(void **array_addr, void *array_desc, ...);
extern void _XMP_init_array_addr(void **array_addr, void *init_addr, void *array_desc, ...);
extern void _XMP_init_array_comm(void *array, ...);
extern unsigned long long _XMP_get_array_total_elmts(void *array);

// xmp_array_section.c
extern void _XMP_normalize_array_section(int *lower, int *upper, int *stride);
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

// xmp_barrier.c
extern void _XMP_barrier_NODES_ENTIRE(void *nodes);
extern void _XMP_barrier_EXEC(void);

// xmp_bcast.c
extern void _XMP_bcast_NODES_ENTIRE_OMITTED(void *bcast_nodes, void *addr, int count, size_t datatype_size);
extern void _XMP_bcast_NODES_ENTIRE_GLOBAL(void *bcast_nodes, void *addr, int count, size_t datatype_size,
                                           int from_lower, int from_upper, int from_stride);
extern void _XMP_bcast_NODES_ENTIRE_NODES(void *bcast_nodes, void *addr, int count, size_t datatype_size, void *from_nodes, ...);

// xmp_coarray.c
extern void _XMP_init_coarray_STATIC(void **coarray, void *addr, int type, size_t type_size, int coarray_size, int dim, ...);
extern void _XMP_init_coarray_DYNAMIC(void **coarray, void *addr, int type, size_t type_size, int dim, ...);
extern void _XMP_finalize_coarray(void *coarray);

// xmp_coarray_RMA.c
extern void _XMP_coarray_get(void *coarray, void *addr);
extern void _XMP_coarray_put(void *addr, void *coarray);

// xmp_gmove.c
extern void _XMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, void *array, ...);
extern _Bool _XMP_gmove_HOMECOPY_SCALAR(void *array, ...);
extern void _XMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, void *dst_array, void *src_array, ...);
extern void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...);
extern void _XMP_gmove_BCAST_ARRAY(void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_HOMECOPY_ARRAY(void *dst_array, int type, size_t type_size, ...);
extern void _XMP_gmove_SENDRECV_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);

// xmp_loop.c
#define _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const template, const int template_index)

#define _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(_type) \
(const _type ser_init, _type ser_cond, const _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const template, const int template_index)

extern void _XMP_sched_loop_template_BLOCK_CHAR               _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(char);
extern void _XMP_sched_loop_template_BLOCK_UNSIGNED_CHAR      _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned char);
extern void _XMP_sched_loop_template_BLOCK_SHORT              _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(short);
extern void _XMP_sched_loop_template_BLOCK_UNSIGNED_SHORT     _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned short);
extern void _XMP_sched_loop_template_BLOCK_INT                _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(int);
extern void _XMP_sched_loop_template_BLOCK_UNSIGNED_INT       _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned int);
extern void _XMP_sched_loop_template_BLOCK_LONG               _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(long);
extern void _XMP_sched_loop_template_BLOCK_UNSIGNED_LONG      _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned long);
extern void _XMP_sched_loop_template_BLOCK_LONGLONG           _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_S(long long);
extern void _XMP_sched_loop_template_BLOCK_UNSIGNED_LONGLONG  _XMP_SM_SCHED_LOOP_TEMPLATE_BLOCK_U(unsigned long long);

#define _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const template, const int template_index)

#define _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(_type) \
(const _type ser_init, _type ser_cond, const _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const template, const int template_index)

extern void _XMP_sched_loop_template_CYCLIC_CHAR               _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(char);
extern void _XMP_sched_loop_template_CYCLIC_UNSIGNED_CHAR      _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned char);
extern void _XMP_sched_loop_template_CYCLIC_SHORT              _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(short);
extern void _XMP_sched_loop_template_CYCLIC_UNSIGNED_SHORT     _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned short);
extern void _XMP_sched_loop_template_CYCLIC_INT                _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(int);
extern void _XMP_sched_loop_template_CYCLIC_UNSIGNED_INT       _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned int);
extern void _XMP_sched_loop_template_CYCLIC_LONG               _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(long);
extern void _XMP_sched_loop_template_CYCLIC_UNSIGNED_LONG      _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned long);
extern void _XMP_sched_loop_template_CYCLIC_LONGLONG           _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_S(long long);
extern void _XMP_sched_loop_template_CYCLIC_UNSIGNED_LONGLONG  _XMP_SM_SCHED_LOOP_TEMPLATE_CYCLIC_U(unsigned long long);

#define _XMP_SM_SCHED_LOOP_NODES_S(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const nodes, const int nodes_index)

#define _XMP_SM_SCHED_LOOP_NODES_U(_type) \
(_type ser_init, _type ser_cond, _type ser_step, \
 _type *const par_init, _type *const par_cond, _type *const par_step, \
 const void *const nodes, const int nodes_index)

extern void _XMP_sched_loop_nodes_CHAR               _XMP_SM_SCHED_LOOP_NODES_S(char);
extern void _XMP_sched_loop_nodes_UNSIGNED_CHAR      _XMP_SM_SCHED_LOOP_NODES_U(unsigned char);
extern void _XMP_sched_loop_nodes_SHORT              _XMP_SM_SCHED_LOOP_NODES_S(short);
extern void _XMP_sched_loop_nodes_UNSIGNED_SHORT     _XMP_SM_SCHED_LOOP_NODES_U(unsigned short);
extern void _XMP_sched_loop_nodes_INT                _XMP_SM_SCHED_LOOP_NODES_S(int);
extern void _XMP_sched_loop_nodes_UNSIGNED_INT       _XMP_SM_SCHED_LOOP_NODES_U(unsigned int);
extern void _XMP_sched_loop_nodes_LONG               _XMP_SM_SCHED_LOOP_NODES_S(long);
extern void _XMP_sched_loop_nodes_UNSIGNED_LONG      _XMP_SM_SCHED_LOOP_NODES_U(unsigned long);
extern void _XMP_sched_loop_nodes_LONGLONG           _XMP_SM_SCHED_LOOP_NODES_S(long long);
extern void _XMP_sched_loop_nodes_UNSIGNED_LONGLONG  _XMP_SM_SCHED_LOOP_NODES_U(unsigned long long);

// xmp_math_function.c
extern int _XMP_modi_ll_i(long long value, int cycle);
extern int _XMP_modi_i_i(int value, int cycle);

// xmp_nodes.c
extern void _XMP_validate_nodes_ref(int *lower, int *upper, int *stride, int size);
extern void _XMP_init_nodes_STATIC_GLOBAL(int map_type, void **nodes, int dim, ...);
extern void _XMP_init_nodes_DYNAMIC_GLOBAL(int map_type, void **nodes, int dim, ...);
extern void _XMP_init_nodes_STATIC_EXEC(int map_type, void **nodes, int dim, ...);
extern void _XMP_init_nodes_DYNAMIC_EXEC(int map_type, void **nodes, int dim, ...);
extern void _XMP_init_nodes_STATIC_NODES_NUMBER(int map_type, void **nodes, int dim,
                                                int ref_lower, int ref_upper, int ref_stride, ...);
extern void _XMP_init_nodes_DYNAMIC_NODES_NUMBER(int map_type, void **nodes, int dim,
                                                 int ref_lower, int ref_upper, int ref_stride, ...);
extern void _XMP_init_nodes_STATIC_NODES_NAMED(int get_upper, int map_type, void **nodes, int dim, void *ref_nodes, ...);
extern void _XMP_init_nodes_DYNAMIC_NODES_NAMED(int get_upper, int map_type, void **nodes, int dim, void *ref_nodes, ...);
extern void _XMP_finalize_nodes(void *nodes);
extern _Bool _XMP_exec_task_GLOBAL_PART(int ref_lower, int ref_upper, int ref_stride);
extern _Bool _XMP_exec_task_NODES_ENTIRE(void *ref_nodes);
extern _Bool _XMP_exec_task_NODES_PART(int get_upper, void *ref_nodes, ...);
extern void *_XMP_create_nodes_by_comm(void *comm);

// xmp_nodes_stack.c
extern void _XMP_push_nodes(void *nodes);
extern void _XMP_pop_nodes(void);
extern void _XMP_pop_n_free_nodes(void);
extern void _XMP_pop_n_free_nodes_wo_finalize_comm(void);
extern void *_XMP_get_execution_nodes(void);
extern int _XMP_get_execution_nodes_rank(void);
extern void _XMP_push_comm(void *comm);
extern void _XMP_finalize_comm(void *comm);

// xmp_reduce.c
extern void _XMP_reduce_NODES_ENTIRE(void *nodes, void *addr, int count, int datatype, int op);
extern void _XMP_reduce_FLMM_NODES_ENTIRE(void *nodes, void *addr, int count, int datatype, int op, int num_locs, ...);
extern void _XMP_reduce_CLAUSE(void *data_addr, int count, int datatype, int op);
extern void _XMP_reduce_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...);
extern void _XMP_init_reduce_comm_NODES(void *nodes, ...);
extern void _XMP_init_reduce_comm_TEMPLATE(void *template, ...);

// xmp_runtime.c
extern void _XMP_init(void);
extern void _XMP_finalize(void);

// xmp_shadow.c
extern void _XMP_init_shadow(void *array, ...);
extern void _XMP_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr, void *array_desc, int array_index);
extern void _XMP_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr, void *array_desc, int array_index);
extern void _XMP_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
                                        void *lo_send_buffer, void *hi_send_buffer,
                                        void *array_desc, int array_index);
extern void _XMP_reflect_shadow_FULL(void *array_addr, void *array_desc, int array_index);

// xmp_template.c
extern void _XMP_init_template_FIXED(void **template, int dim, ...);
extern void _XMP_init_template_UNFIXED(void **template, int dim, ...);
extern void _XMP_init_template_chunk(void *template, void *nodes);
extern void _XMP_finalize_template(void *template);
extern void _XMP_dist_template_DUPLICATION(void *template, int template_index);
extern void _XMP_dist_template_BLOCK(void *template, int template_index, int nodes_index);
extern void _XMP_dist_template_CYCLIC(void *template, int template_index, int nodes_index);
extern _Bool _XMP_exec_task_TEMPLATE_PART(int get_upper, void *ref_template, ...);

// xmp_util.c
extern void *_XMP_alloc(size_t size);
extern void _XMP_free(void *p);
extern void _XMP_fatal(char *msg);
extern void _XMP_unexpected_error(void);

// xmp_world.c
extern void _XMP_init_world(int *argc, char ***argv);
extern void _XMP_finalize_world(void);

// ----- libxmp_threads
// xmp_threads_runtime.c
extern void _XMP_threads_init(int argc, char *argv[]);
extern void _XMP_threads_finalize(int ret);

// xmp_gpu_runtime.cu
extern void _XMP_gpu_init(void);
extern void _XMP_gpu_finalize(void);

// xmp_gpu_data.cu
extern void _XMP_gpu_init_data_NOT_ALIGNED(void **host_desc, void **device_desc, void **device_addr, void *addr, size_t size);
extern void _XMP_gpu_init_data_ALIGNED(void **host_data_desc, void **device_data_desc, void **device_addr, void *addr, void *array_desc);
extern void _XMP_gpu_finalize_data(void *desc);

// xmp_gpu_sync.cu
extern void _XMP_gpu_sync(void *desc, int direction);

#endif // _XMP_RUNTIME_FUNC_DECL
