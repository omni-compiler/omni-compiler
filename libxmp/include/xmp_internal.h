/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_INTERNAL
#define _XMP_INTERNAL

// --------------- including headers  --------------------------------
#include <stddef.h>

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

// --------------- structures ----------------------------------------
#define _XMP_comm void
#include "xmp_data_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

// ----- libxmp ------------------------------------------------------
// xmp_array_section.c
extern void _XMP_normalize_array_section(int *lower, int *upper, int *stride);
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

// xmp_barrier.c
extern void _XMP_barrier_EXEC(void);

// xmp_nodes.c
extern void _XMP_init_nodes_STATIC_NODES_NAMED_MAIN(_XMP_nodes_t **nodes, int dim,
                                                    _XMP_nodes_t *ref_nodes,
                                                    int *shrink, int *ref_lower, int *ref_upper, int *ref_stride,
                                                    int *dim_size);
extern void _XMP_finalize_nodes(_XMP_nodes_t *nodes);
extern _XMP_nodes_t *_XMP_create_nodes_by_comm(int is_member, _XMP_comm *comm);
extern int _XMP_calc_linear_rank(_XMP_nodes_t *n, int *rank_array);
extern int _XMP_calc_linear_rank_on_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes);
extern _XMP_nodes_ref_t *_XMP_init_nodes_ref(_XMP_nodes_t *n, int *rank_array);
extern void _XMP_finalize_nodes_ref(_XMP_nodes_ref_t *nodes_ref);
extern _XMP_nodes_ref_t *_XMP_create_nodes_ref_for_target_nodes(_XMP_nodes_t *n, int *rank_array, _XMP_nodes_t *target_nodes);

// xmp_nodes_stack.c
extern void _XMP_push_nodes(_XMP_nodes_t *nodes);
extern void _XMP_pop_nodes(void);
extern void _XMP_pop_n_free_nodes(void);
extern void _XMP_pop_n_free_nodes_wo_finalize_comm(void);
extern _XMP_nodes_t *_XMP_get_execution_nodes(void);
extern int _XMP_get_execution_nodes_rank(void);
extern void _XMP_push_comm(_XMP_comm *comm);
extern void _XMP_finalize_comm(_XMP_comm *comm);

// xmp_template.c
extern int _XMP_check_template_ref_inclusion(int ref_lower, int ref_upper, int ref_stride,
                                             _XMP_template_t *t, int index);
extern _XMP_nodes_t *_XMP_create_nodes_by_template_ref(_XMP_template_t *template, int *shrink,
                                                       long long *lower, long long *upper, long long *stride);
extern int _XMP_calc_template_owner_SCALAR(_XMP_template_t *template, int dim_index, long long ref_index);

// xmp_util.c
extern void *_XMP_alloc(size_t size);
extern void _XMP_free(void *p);
extern void _XMP_fatal(char *msg);
extern void _XMP_unexpected_error(void);

// xmp_world.c
extern int _XMP_world_size;
extern int _XMP_world_rank;
extern void *_XMP_world_nodes;

extern void _XMP_init_world(int *argc, char ***argv);
extern void _XMP_finalize_world(void);
extern int _XMP_split_world_by_color(int color);

// xmp_runtime.c
extern void _XMP_init(void);
extern void _XMP_finalize(void);

// ----- libxmp_threads ----------------------------------------------
// xmp_threads_runtime.c
extern void _XMP_threads_init(void);
extern void _XMP_threads_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // _XMP_INTERNAL
