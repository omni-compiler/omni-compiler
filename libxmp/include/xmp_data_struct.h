/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XMP_DATA_STRUCT
#define _XMP_DATA_STRUCT

#include <stdbool.h>
#include "xmp_constant.h"

#define _XMP_comm_t void
#define _XMP_coarray_comm_t void
#define _XMP_data_type_t void

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
} _XMP_nodes_info_t;

typedef struct _XMP_nodes_type {
  unsigned long long on_ref_id;

  int is_member;
  int dim;
  int comm_size;

  // enable when is_member is true
  int comm_rank;
  _XMP_comm_t *comm;
  // -----------------------------

  struct _XMP_nodes_type *inherit_nodes;
  // enable when inherit_nodes is not NULL
  _XMP_nodes_inherit_info_t *inherit_info;
  // -------------------------------------
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

// aligned array descriptor
typedef struct _XMP_array_info_type {
  _Bool is_shadow_comm_member;
  _Bool is_regular_chunk;
  int align_manner;

  int ser_lower;
  int ser_upper;
  int ser_size;

  // enable when is_allocated is true
  int par_lower;
  int par_upper;
  int par_stride;
  int par_size;

  int local_lower;
  int local_upper;
  int local_stride;
  int alloc_size;

  int *temp0;
  int temp0_v;

  unsigned long long dim_acc;
  unsigned long long dim_elmts;
  // --------------------------------

  long long align_subscript;

  int shadow_type;
  int shadow_size_lo;
  int shadow_size_hi;

  // enable when is_shadow_comm_member is true
  _XMP_comm_t *shadow_comm;
  int shadow_comm_size;
  int shadow_comm_rank;
  // -----------------------------------------

  int align_template_index;
} _XMP_array_info_t;

typedef struct _XMP_array_type {
  _Bool is_allocated;
  _Bool is_align_comm_member;
  int dim;
  int type;
  size_t type_size;

  // enable when is_allocated is true
  void **array_addr_p;
  unsigned long long total_elmts;
  // --------------------------------

  // FIXME do not use these members
  // enable when is_align_comm_member is true
  _XMP_comm_t *align_comm;
  int align_comm_size;
  int align_comm_rank;
  // ----------------------------------------

  _XMP_nodes_t *array_nodes;

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

// coarray descriptor
typedef struct _XMP_coarray_type {
  void *addr;
  int type;
  size_t type_size;

  _XMP_nodes_t *nodes;
  _XMP_coarray_comm_t *comm;
  _XMP_data_type_t *data_type;
} _XMP_coarray_t;

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
