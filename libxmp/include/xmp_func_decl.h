/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_RUNTIME_FUNC_DECL
#define _XMP_RUNTIME_FUNC_DECL

#ifndef _XMP_CRAY
#include <stddef.h>
#endif

// ----- libxml
// xmp_align.c
extern void _XMP_init_array_desc(void **array, void *template, int dim, int type, size_t type_size, ...);
extern void _XMP_finalize_array_desc(void *array);
extern void _XMP_align_array_NOT_ALIGNED(void *array, int array_index);
extern void _XMP_align_array_DUPLICATION(void *array, int array_index, int template_index, long long align_subscript);
extern void _XMP_align_array_BLOCK(void *array, int array_index, int template_index, long long align_subscript, int *temp0);
extern void _XMP_align_array_CYCLIC(void *array, int array_index, int template_index, long long align_subscript, int *temp0);
extern void _XMP_align_array_BLOCK_CYCLIC(void *array, int array_index, int template_index, long long align_subscript, int *temp0);
extern void _XMP_align_array_GBLOCK(void *array, int array_index, int template_index,
				    long long align_subscript, int *temp0);
extern void _XMP_alloc_array(void **array_addr, void *array_desc, ...);
extern void _XMP_dealloc_array(void *array_desc);
extern void _XMP_alloc_array_EXTERN(void **array_addr, void *array_desc, ...);
extern void _XMP_init_array_addr(void **array_addr, void *init_addr, void *array_desc, ...);
extern void _XMP_init_array_comm(void *array, ...);
extern void _XMP_init_array_nodes(void *array);
extern unsigned long long _XMP_get_array_total_elmts(void *array);
extern void _XMP_align_array_noalloc(void *a, int adim, int tdim, long long align_subscript, int *temp0, unsigned long long *acc0);

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
extern void _XMP_gasnet_not_continuous_put();
extern void _XMP_gasnet_continuous_put();
extern void _XMP_gasnet_not_continuous_get();
extern void _XMP_gasnet_continuous_get();
extern void _XMP_coarray_malloc_set_f(int *elmt_size, int *coarray_dims, int *image_dims);
extern void _XMP_coarray_malloc_set(int elmt_size, int coarray_dims, int image_dims);
extern void _XMP_coarray_malloc_array_info_f(int *dim, long long *size);
extern void _XMP_coarray_malloc_array_info(int dim, long long size);
extern void _XMP_coarray_malloc_image_info_f(int *dim, int *image_size);
extern void _XMP_coarray_malloc_image_info(int dim, int image_size);
extern void _XMP_coarray_malloc_do_f(void **coarray, void *addr);
extern void _XMP_coarray_malloc_do(void **coarray, void *addr);
extern void _XMP_coarray_rdma_set_f(int *coarray_dims, int *array_dims, int *image_dims);
extern void _XMP_coarray_rdma_set(int coarray_dims, int array_dims, int image_dimss);
extern void _XMP_coarray_rdma_coarray_set_f(int *dim, long long *start, long long *length, long long *stride);
extern void _XMP_coarray_rdma_coarray_set(int dim, long long start, long long length, long long stride);
extern void _XMP_coarray_rdma_array_set_f(int *dim, long long *start, long long *length, long long *stride, long long *size, long long *distance);
extern void _XMP_coarray_rdma_array_set(int dim, long long start, long long length, long long stride, long long size, long long distance);
extern void _XMP_coarray_rdma_node_set_f(int *dim, int *image_num);
extern void _XMP_coarray_rdma_node_set(int dim, int image_num);
extern void _XMP_coarray_rdma_do_f(int *rdma_code, void *coarray, void *array);
extern void _XMP_coarray_rdma_do(int rdma_code, void *coarray, void *array);
extern void _XMP_coarray_sync_all();
extern void _XMP_coarray_sync_memory();
extern void xmp_sync_memory(int* status);
extern void xmp_sync_all(int* status);
extern void xmp_sync_image(int image, int* status);
extern void xmp_sync_image_f(int *image, int* status);
extern void xmp_sync_images(int num, int* image_set, int* status);
extern void xmp_sync_images_f(int *num, int* image_set, int* status);
extern void xmp_sync_images_all(int* status);
extern long long get_offset(void *, int);

// xmp_gmove.c
extern void _XMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, void *array, ...);
extern int _XMP_gmove_HOMECOPY_SCALAR(void *array, ...);
extern void _XMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, void *dst_array, void *src_array, ...);
extern void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...);
extern void _XMP_gmove_BCAST_ARRAY(void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_HOMECOPY_ARRAY(void *dst_array, int type, size_t type_size, ...);
extern void _XMP_gmove_SENDRECV_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_BCAST_TO_NOTALIGNED_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);

// xmp_loop.c
extern void _XMP_sched_loop_template_DUPLICATION(int ser_init, int ser_cond, int ser_step,
                                                 int *par_init, int *par_cond, int *par_step,
                                                 void *template, int template_index);
extern void _XMP_sched_loop_template_BLOCK(int ser_init, int ser_cond, int ser_step,
                                           int *par_init, int *par_cond, int *par_step,
                                           void *template, int template_index);
extern void _XMP_sched_loop_template_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                            int *par_init, int *par_cond, int *par_step,
                                            void *template, int template_index);
extern void _XMP_sched_loop_template_BLOCK_CYCLIC(int ser_init, int ser_cond, int ser_step,
                                           int *par_init, int *par_cond, int *par_step,
                                           void *template, int template_index);
extern void _XMP_sched_loop_nodes(int ser_init, int ser_cond, int ser_step,
                                  int *par_init, int *par_cond, int *par_step,
                                  void *nodes, int nodes_index);
extern void _XMP_sched_loop_template_GBLOCK(int ser_init, int ser_cond, int ser_step,
					    int *par_init, int *par_cond, int *par_step,
					    void *template, int template_index);

// xmp_math_function.c
extern int _XMP_modi_ll_i(long long value, int cycle);
extern int _XMP_modi_i_i(int value, int cycle);

// xmp_nodes.c
extern void _XMP_init_nodes_STATIC_GLOBAL(void **nodes, int dim, ...);
extern void _XMP_init_nodes_DYNAMIC_GLOBAL(void **nodes, int dim, ...);
extern void _XMP_init_nodes_STATIC_EXEC(void **nodes, int dim, ...);
extern void _XMP_init_nodes_DYNAMIC_EXEC(void **nodes, int dim, ...);
extern void _XMP_init_nodes_STATIC_NODES_NUMBER(void **nodes, int dim,
                                                int ref_lower, int ref_upper, int ref_stride, ...);
extern void _XMP_init_nodes_DYNAMIC_NODES_NUMBER(void **nodes, int dim,
                                                 int ref_lower, int ref_upper, int ref_stride, ...);
extern void _XMP_init_nodes_STATIC_NODES_NAMED(void **nodes, int dim, void *ref_nodes, ...);
extern void _XMP_init_nodes_DYNAMIC_NODES_NAMED(void **nodes, int dim, void *ref_nodes, ...);
extern void _XMP_finalize_nodes(void *nodes);
extern int _XMP_exec_task_GLOBAL_PART(void **task_desc, int ref_lower, int ref_upper, int ref_stride);
extern int _XMP_exec_task_NODES_ENTIRE(void **task_desc, void *ref_nodes);
extern int _XMP_exec_task_NODES_PART(void **task_desc, void *ref_nodes, ...);
extern void _XMP_exec_task_NODES_FINALIZE(void *task_desc);

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
extern int _XMP_init_reduce_comm_NODES(void *nodes, ...);
extern int _XMP_init_reduce_comm_TEMPLATE(void *template, ...);

// xmp_reflect.c
extern void _XMP_set_reflect__(void *a, int dim, int lwidth, int uwidth, int is_periodic);
extern void _XMP_reflect__(char *a);
extern void _XMP_wait_async__(int async_id);
extern void _XMP_reflect_async__(void *a, int async_id);

// xmp_runtime.c
//extern void _XMP_init(void);
extern void _XMP_init(int, char**); 
extern void _XMP_finalize(int);
extern char *_XMP_desc_of(void *p);

// xmp_shadow.c
extern void _XMP_init_shadow(void *array, ...);
extern void _XMP_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr, void *array_desc, int array_index);
extern void _XMP_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr, void *array_desc, int array_index);
extern void _XMP_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
                                        void *lo_send_buffer, void *hi_send_buffer,
                                        void *array_desc, int array_index);
extern void _XMP_reflect_shadow_FULL(void *array_addr, void *array_desc, int array_index);
extern void _XMP_init_shadow_noalloc(void *a, int shadow_type, int lshadow, int ushadow);

// xmp_template.c
extern void _XMP_init_template_FIXED(void **template, int dim, ...);
extern void _XMP_init_template_UNFIXED(void **template, int dim, ...);
extern void _XMP_set_template_size(void **template, int dim, ...);
extern void _XMP_init_template_chunk(void *template, void *nodes);
extern void _XMP_finalize_template(void *template);
extern void _XMP_dist_template_DUPLICATION(void *template, int template_index);
extern void _XMP_dist_template_BLOCK(void *template, int template_index, int nodes_index);
extern void _XMP_dist_template_CYCLIC(void *template, int template_index, int nodes_index);
extern void _XMP_dist_template_BLOCK_CYCLIC(void *template, int template_index, int nodes_index, unsigned long long width);
extern void _XMP_dist_template_GBLOCK(void *template, int template_index, int nodes_index,
				      int *mapping_array);

extern int _XMP_exec_task_TEMPLATE_PART(void **task_desc, void *ref_template, ...);
extern long long int _XMP_L2G_GBLOCK(int local_idx, void *template, int template_index);

// xmp_util.c
extern void *_XMP_alloc(size_t size);
extern void _XMP_free(void *p);
extern void _XMP_fatal(char *msg);
extern void _XMP_unexpected_error(void);

// xmp_world.c
extern void _XMP_init_world(int *argc, char ***argv);
extern void _XMP_finalize_world(void);

// xmp_post.c
#ifdef _XMP_FJRDMA_COARRAY
extern void _XMP_post(xmp_desc_t, int, int node, int tag);
extern void _XMP_wait(int dummy, int target_node, int tag);
extern void _XMP_post_initialize(void);
#else
extern void _XMP_post(void *, int num, ...);
extern void _XMP_wait(int num, ...);
extern void _XMP_post_initialize(void);
#endif

// xmp_gasnet_post.c
#ifndef _XMP_FJRDMA_COARRAY
extern void _xmp_gasnet_post(int node, int tag);
extern void _xmp_gasnet_wait(int num, ...);
extern void _xmp_gasnet_post_initialize(void);
#endif

// ----- libxmp_threads
// xmp_threads_runtime.c
extern void _XMP_threads_init(void);
extern void _XMP_threads_finalize(void);

// xmp_gpu_runtime.cu
extern void _XMP_gpu_init(void);
extern void _XMP_gpu_finalize(void);

// xmp_gpu_data.cu
extern void _XMP_gpu_init_data_NOT_ALIGNED(void **host_data_desc,
                                           void **device_addr, void *addr, size_t size);
extern void _XMP_gpu_init_data_ALIGNED(void **host_data_desc, void ** device_array_desc,
                                       void **device_addr, void *addr, void *array_desc);
extern void _XMP_gpu_finalize_data(void *desc);

// xmp_gpu_sync.cu
extern void _XMP_gpu_sync(void *desc, int direction);

// xmp_gpu_shadow.c
extern void _XMP_gpu_pack_shadow_NORMAL(void *desc, void **lo_buffer, void **hi_buffer, int array_index);
extern void _XMP_gpu_unpack_shadow_NORMAL(void *desc, void *lo_buffer, void *hi_buffer, int array_index);

#endif // _XMP_RUNTIME_FUNC_DECL
