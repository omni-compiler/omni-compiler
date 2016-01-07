#ifndef _XMP_RUNTIME_FUNC_DECL
#define _XMP_RUNTIME_FUNC_DECL

#if !defined(_XMP_CRAY)
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
extern int _XMP_lidx_GBLOCK(void *a, int i_dim, int global_idx);

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

// xmp_async.c
extern void xmpc_init_async(int async_id);
extern void xmpc_start_async(int async_id);

// xmp_barrier.c
extern void _XMP_barrier_NODES_ENTIRE(void *nodes);
extern void _XMP_barrier_EXEC(void);

// xmp_bcast.c
extern void _XMP_bcast_NODES_ENTIRE_OMITTED(void *bcast_nodes, void *addr, int count, size_t datatype_size);
extern void _XMP_bcast_NODES_ENTIRE_GLOBAL(void *bcast_nodes, void *addr, int count, size_t datatype_size,
                                           int from_lower, int from_upper, int from_stride);
extern void _XMP_bcast_NODES_ENTIRE_NODES(void *bcast_nodes, void *addr, int count, size_t datatype_size, void *from_nodes, ...);

// xmp_bcast_acc.c
extern void _XMP_bcast_acc_NODES_ENTIRE_OMITTED(void *bcast_nodes, void *addr, int count, size_t datatype_size);
extern void _XMP_bcast_acc_NODES_ENTIRE_GLOBAL(void *bcast_nodes, void *addr, int count, size_t datatype_size,
                                           int from_lower, int from_upper, int from_stride);
extern void _XMP_bcast_acc_NODES_ENTIRE_NODES(void *bcast_nodes, void *addr, int count, size_t datatype_size, void *from_nodes, ...);

// xmp_coarray.c
extern void _XMP_gasnet_not_continuous_put();
extern void _XMP_gasnet_continuous_put();
extern void _XMP_gasnet_not_continuous_get();
extern void _XMP_gasnet_continuous_get();
extern void _XMP_coarray_malloc_info_1(const long, const size_t);
extern void _XMP_coarray_malloc_info_2(const long, const long, const size_t);
extern void _XMP_coarray_malloc_info_3(const long, const long, const long, const size_t);
extern void _XMP_coarray_malloc_info_4(const long, const long, const long, const long, 
				       const size_t);
extern void _XMP_coarray_malloc_info_5(const long, const long, const long, const long, 
				       const long, const size_t);
extern void _XMP_coarray_malloc_info_6(const long, const long, const long, const long, 
				       const long, const long, const size_t);
extern void _XMP_coarray_malloc_info_7(const long, const long, const long, const long, 
				       const long, const long, const long, const size_t);

extern void _XMP_coarray_malloc_image_info_1();
extern void _XMP_coarray_malloc_image_info_2(const int);
extern void _XMP_coarray_malloc_image_info_3(const int, const int);
extern void _XMP_coarray_malloc_image_info_4(const int, const int, const int);
extern void _XMP_coarray_malloc_image_info_5(const int, const int, const int, const int);
extern void _XMP_coarray_malloc_image_info_6(const int, const int, const int, const int, const int);
extern void _XMP_coarray_malloc_image_info_7(const int, const int, const int, const int, const int, const int);

extern void _XMP_coarray_malloc_do_f(void **, void *);
extern void _XMP_coarray_malloc_do(void **, void *);
extern void _XMP_coarray_attach(void **, void *, const size_t);
extern void _XMP_coarray_detach(void **);
extern void _XMP_coarray_lastly_deallocate();

extern void _XMP_coarray_rdma_coarray_set_1(const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_2(const long, const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_3(const long, const long, const long, const long, const long, const long,
					    const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_4(const long, const long, const long, const long, const long, const long,
                                            const long, const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_5(const long, const long, const long, const long, const long, const long,
                                            const long, const long, const long, const long, const long, const long,
					    const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_6(const long, const long, const long, const long, const long, const long,
                                            const long, const long, const long, const long, const long, const long,
					    const long, const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_coarray_set_7(const long, const long, const long, const long, const long, const long,
                                            const long, const long, const long, const long, const long, const long,
                                            const long, const long, const long, const long, const long, const long,
					    const long, const long, const long);

extern void _XMP_coarray_rdma_array_set_1(const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_2(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_3(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_4(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_5(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_6(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);
extern void _XMP_coarray_rdma_array_set_7(const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long,
					  const long, const long, const long, const long, const long);

extern void _XMP_coarray_rdma_image_set_1(const int);
extern void _XMP_coarray_rdma_image_set_2(const int, const int);
extern void _XMP_coarray_rdma_image_set_3(const int, const int, const int);
extern void _XMP_coarray_rdma_image_set_4(const int, const int, const int, const int);
extern void _XMP_coarray_rdma_image_set_5(const int, const int, const int, const int, const int);
extern void _XMP_coarray_rdma_image_set_6(const int, const int, const int, const int, const int, const int);
extern void _XMP_coarray_rdma_image_set_7(const int, const int, const int, const int, const int, const int, const int);

extern void _XMP_coarray_rdma_do_f(const int*, void*, void*, void*);
extern void _XMP_coarray_rdma_do(const int, void*, void*, void *);
extern void _XMP_coarray_sync_all();
extern void _XMP_coarray_sync_memory();
extern void xmp_sync_memory(const int* status);
extern void xmp_sync_all(const int* status);
extern void xmp_sync_image(int image, int* status);
extern void xmp_sync_image_f(int *image, int* status);
extern void xmp_sync_images(const int num, int* image_set, int* status);
extern void xmp_sync_images_f(const int *num, int* image_set, int* status);
extern void xmp_sync_images_all(int* status);
extern void _XMP_coarray_shortcut_put(const int, void*, const void*, const long, const long, const long, const long);
extern void _XMP_coarray_shortcut_put_f(const int*, void*, const void*, const long*, const long*, const long*, const long*);
extern void _XMP_coarray_shortcut_get(const int, void*, const void*, const long, const long, const long, const long);
extern void _XMP_coarray_shortcut_get_f(const int*, void*, const void*, const long*, const long*, const long*, const long*);

// xmp_coarray_acc.c
int _XMP_coarray_get_total_elmts(void *coarray_desc);
void _XMP_coarray_malloc_do_acc(void **coarray_desc, void *addr);
void _XMP_coarray_shortcut_put_acc(const int target_image, const void *dst_desc, const void *src_desc, 
				   const size_t dst_offset, const size_t src_offset, 
				   const size_t dst_elmts, const size_t src_elmts,
				   const int is_dst_on_acc, const int is_src_on_acc);
void _XMP_coarray_shortcut_get_acc(const int target_image, const void *dst_desc, const void *src_desc, 
				   const size_t dst_offset, const size_t src_offset, 
				   const size_t dst_elmts, const size_t src_elmts,
				   const int is_dst_on_acc, const int is_src_on_acc);
extern void _XMP_coarray_rdma_do_acc(const int, void*, void*, void *, const int, const int);

// xmp_reflect_acc.c
extern void _XMP_reflect_init_acc(void *, void *);
extern void _XMP_reflect_do_acc(void *);
extern void _XMP_reflect_acc(void *);

#ifdef _XMP_TCA
// xmp_tca.c
extern void _XMP_init_tca();
extern void _XMP_alloc_tca(void *);
extern void _XMP_create_TCA_handle(void *, void *);
extern void _XMP_create_TCA_desc(void *);
extern void _XMP_reflect_do_tca(_XMP_array_t *);
#endif

// xmp_gmove.c
extern void _XMP_gmove_BCAST_SCALAR(void *dst_addr, void *src_addr, void *array, ...);
extern int _XMP_gmove_HOMECOPY_SCALAR(void *array, ...);
extern void _XMP_gmove_SENDRECV_SCALAR(void *dst_addr, void *src_addr, void *dst_array, void *src_array, ...);
extern void _XMP_gmove_LOCALCOPY_ARRAY(int type, size_t type_size, ...);
extern void _XMP_gmove_BCAST_ARRAY(void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_HOMECOPY_ARRAY(void *dst_array, int type, size_t type_size, ...);
extern void _XMP_gmove_SENDRECV_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_BCAST_TO_NOTALIGNED_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);

// xmp_gmove_acc.c
extern void _XMP_gmove_acc_BCAST_SCALAR(void *dst_addr, void *src_addr, void *array, ...);
extern int _XMP_gmove_acc_HOMECOPY_SCALAR(void *array, ...);
extern void _XMP_gmove_acc_SENDRECV_SCALAR(void *dst_addr, void *src_addr, void *dst_array, void *src_array, ...);
extern void _XMP_gmove_acc_LOCALCOPY_ARRAY(int type, size_t type_size, ...);
extern void _XMP_gmove_acc_BCAST_ARRAY(void *src_array, int type, size_t type_size, ...);
extern void _XMP_gmove_acc_HOMECOPY_ARRAY(void *dst_array, int type, size_t type_size, ...);
extern void _XMP_gmove_acc_SENDRECV_ARRAY(void *dst_array, void *src_array, void *dst_array_dev, void *src_array_dev, int type, size_t type_size, ...);
extern void _XMP_gmove_acc_BCAST_TO_NOTALIGNED_ARRAY(void *dst_array, void *src_array, int type, size_t type_size, ...);

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
//extern int _XMP_exec_task_GLOBAL_PART(void **task_desc, int ref_lower, int ref_upper, int ref_stride);
extern int _XMP_exec_task_NODES_ENTIRE(void **task_desc, void *ref_nodes);
extern int _XMP_exec_task_NODES_ENTIRE_nocomm(void **task_desc, void *ref_nodes);
extern int _XMP_exec_task_NODES_PART(void **task_desc, void *ref_nodes, ...);
extern int _XMP_exec_task_NODES_PART_nocomm(void **task_desc, void *ref_nodes, ...);
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

// xmp_reduce_acc.c
extern void _XMP_reduce_acc_NODES_ENTIRE(void *nodes, void *dev_addr, int count, int datatype, int op);
extern void _XMP_reduce_acc_FLMM_NODES_ENTIRE(void *nodes, void *addr, int count, int datatype, int op, int num_locs, ...);
extern void _XMP_reduce_acc_CLAUSE(void *dev_addr, int count, int datatype, int op);
extern void _XMP_reduce_acc_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...);


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
//extern void _XMP_init_shadow_noalloc(void *a, int shadow_type, int lshadow, int ushadow);
extern void _XMP_init_shadow_noalloc(void *a, ...);

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
				      int *mapping_array, int *temp0);

extern int _XMP_exec_task_TEMPLATE_PART(void **task_desc, void *ref_template, ...);
extern int _XMP_exec_task_TEMPLATE_PART_nocomm(void **task_desc, void *ref_template, ...);
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
extern void _XMP_post_1(const void*, const int, const int);
extern void _XMP_post_2(const void*, const int, const int, const int);
extern void _XMP_post_3(const void*, const int, const int, const int, const int);
extern void _XMP_post_4(const void*, const int, const int, const int, const int, const int);
extern void _XMP_post_5(const void*, const int, const int, const int, const int, const int, const int);
extern void _XMP_post_6(const void*, const int, const int, const int, const int, const int, const int, const int);
extern void _XMP_post_7(const void*, const int, const int, const int, const int, const int, const int, const int, const int);

extern void _XMP_wait_noargs();
extern void _XMP_wait_1(const void*, const int, const int);
extern void _XMP_wait_2(const void*, const int, const int, const int);
extern void _XMP_wait_3(const void*, const int, const int, const int, const int);
extern void _XMP_wait_4(const void*, const int, const int, const int, const int, const int);
extern void _XMP_wait_5(const void*, const int, const int, const int, const int, const int, const int);
extern void _XMP_wait_6(const void*, const int, const int, const int, const int, const int, const int, const int);
extern void _XMP_wait_7(const void*, const int, const int, const int, const int, const int, const int, const int, const int);

extern void _XMP_wait_node_1(const void*, const int);
extern void _XMP_wait_node_2(const void*, const int, const int);
extern void _XMP_wait_node_3(const void*, const int, const int, const int);
extern void _XMP_wait_node_4(const void*, const int, const int, const int, const int);
extern void _XMP_wait_node_5(const void*, const int, const int, const int, const int, const int);
extern void _XMP_wait_node_6(const void*, const int, const int, const int, const int, const int, const int);
extern void _XMP_wait_node_7(const void*, const int, const int, const int, const int, const int, const int, const int);

// xmp_lock.c
extern void _XMP_lock_0(const void*, const unsigned int);
extern void _XMP_lock_1(const void*, const unsigned int, const int);
extern void _XMP_lock_2(const void*, const unsigned int, const int, const int);
extern void _XMP_lock_3(const void*, const unsigned int, const int, const int, const int);
extern void _XMP_lock_4(const void*, const unsigned int, const int, const int, const int, const int);
extern void _XMP_lock_5(const void*, const unsigned int, const int, const int, const int, const int, const int);
extern void _XMP_lock_6(const void*, const unsigned int, const int, const int, const int, const int, const int, const int);
extern void _XMP_lock_7(const void*, const unsigned int, const int, const int, const int, const int, const int, const int, const int);

extern void _XMP_unlock_0(const void*, const unsigned int);
extern void _XMP_unlock_1(const void*, const unsigned int, const int);
extern void _XMP_unlock_2(const void*, const unsigned int, const int, const int);
extern void _XMP_unlock_3(const void*, const unsigned int, const int, const int, const int);
extern void _XMP_unlock_4(const void*, const unsigned int, const int, const int, const int, const int);
extern void _XMP_unlock_5(const void*, const unsigned int, const int, const int, const int, const int, const int);
extern void _XMP_unlock_6(const void*, const unsigned int, const int, const int, const int, const int, const int, const int);
extern void _XMP_unlock_7(const void*, const unsigned int, const int, const int, const int, const int, const int, const int, const int);

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

// xmp_intrinsic.c
extern void xmp_transpose(void *dst_d, void *src_d, int opt);
extern void xmp_matmul(void *x_p, void *a_p, void *b_p);
extern void xmp_pack_mask(void *v_p, void *a_p, void *m_p);
extern void xmp_pack_nomask(void *v_p, void *a_p);
extern void xmp_pack(void *v_p, void *a_p, void *m_p);
extern void xmp_unpack_mask(void *a_p, void *v_p, void *m_p);
extern void xmp_unpack_nomask(void *a_p, void *v_p);
extern void xmp_unpack(void *a_p, void *v_p, void *m_p);
extern void xmp_gather(void *, void *, ...);
extern void xmp_scatter(void *, void *, ...);

// xmp_lock_unlock.c
extern void _XMP_lock_initialize_1(void*, const unsigned int);
extern void _XMP_lock_initialize_2(void*, const unsigned int, const unsigned int);
extern void _XMP_lock_initialize_3(void*, const unsigned int, const unsigned int, const unsigned int);
extern void _XMP_lock_initialize_4(void*, const unsigned int, const unsigned int, const unsigned int,
				   const unsigned int);
extern void _XMP_lock_initialize_5(void*, const unsigned int, const unsigned int, const unsigned int,
				   const unsigned int, const unsigned int);
extern void _XMP_lock_initialize_6(void*, const unsigned int, const unsigned int, const unsigned int,
				   const unsigned int, const unsigned int, const unsigned int);
extern void _XMP_lock_initialize_7(void*, const unsigned int, const unsigned int, const unsigned int,
				   const unsigned int, const unsigned int, const unsigned int,
				   const unsigned int);
#endif // _XMP_RUNTIME_FUNC_DECL
