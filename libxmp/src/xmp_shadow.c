#include <stdarg.h>
#include "xmp_array_section.h"
#include "xmp_constant.h"
#include "xmp_internal.h"

//FIXME delete this include
#include <stdio.h>

static void _XCALABLEMP_create_shadow_comm(_XCALABLEMP_array_t *array, int array_index) {
  _XCALABLEMP_array_info_t *ai = &(array->info[array_index]);
  _XCALABLEMP_template_chunk_t *chunk = ai->align_template_chunk;
  _XCALABLEMP_nodes_t *onto_nodes = array->align_template->onto_nodes;

  int onto_nodes_index = chunk->onto_nodes_index;

  int array_dim = array->dim;

//  int color = 1;
//  for (int i = 0; i < array_dim; i++) {
//    ;
//  }
}

static void _XCALABLEMP_pack_shadow_buffer(void *buffer, int array_type, int array_dim,
                                           int *lower, int *upper, int *stride, unsigned long long *dim_acc);
static void _XCALABLEMP_unpack_shadow_buffer(void *buffer, int array_type, int array_dim,
                                             int *lower, int *upper, int *stride, unsigned long long *dim_acc);

// FIXME implement
static void _XCALABLEMP_pack_shadow_buffer(void *buffer, int array_type, int array_dim,
                                           int *lower, int *upper, int *stride, unsigned long long *dim_acc) {
  printf("[%d] pack ", _XCALABLEMP_world_rank);
  for (int i = 0; i < array_dim; i++) {
    printf("[%d:%d:%d](%ld)", lower[i], upper[i], stride[i], dim_acc[i]);
  }
  printf("\n");
}

// FIXME implement
static void _XCALABLEMP_unpack_shadow_buffer(void *buffer, int array_type, int array_dim,
                                             int *lower, int *upper, int *stride, unsigned long long *dim_acc) {
  printf("[%d] unpack ", _XCALABLEMP_world_rank);
  for (int i = 0; i < array_dim; i++) {
    printf("[%d:%d:%d](%ld)", lower[i], upper[i], stride[i], dim_acc[i]);
  }
  printf("\n");
}

void _XCALABLEMP_init_shadow(_XCALABLEMP_array_t *array, ...) {
  int dim = array->dim;
  va_list args;
  va_start(args, array);
  for (int i = 0; i < dim; i++) {
    int type = va_arg(args, int);
    _XCALABLEMP_array_info_t *ai = &(array->info[i]);
    ai->shadow_type = type;

    switch (type) {
      case _XCALABLEMP_N_SHADOW_NONE:
        break;
      case _XCALABLEMP_N_SHADOW_NORMAL:
        {
          int lo = va_arg(args, int);
          if (lo < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          int hi = va_arg(args, int);
          if (hi < 0) _XCALABLEMP_fatal("<shadow-width> should be a nonnegative integer");

          if ((lo != 0) || (hi != 0)) {
            ai->shadow_size_lo = lo;
            ai->shadow_size_hi = hi;

            if (array->is_allocated) {
              ai->local_lower += lo;
              ai->local_upper += lo;
           // ai->local_stride is not changed
              ai->alloc_size += lo + hi;
            }
          }
        } break;
      case _XCALABLEMP_N_SHADOW_FULL:
        {
          // FIXME calc shadow_size_{lo/hi} size

          if (array->is_allocated) {
            ai->local_lower = ai->par_lower;
            ai->local_upper = ai->par_upper;
            ai->local_stride = ai->par_stride;
            ai->alloc_size = ai->ser_size;
          }
        } break;
      default:
        _XCALABLEMP_fatal("unknown shadow type");
    }
  }
}

// FIXME consider full shadow in other dimensions
void _XCALABLEMP_pack_shadow_NORMAL_BASIC(void **lo_buffer, void **hi_buffer, void *array_addr,
                                          _XCALABLEMP_array_t *array_desc, int array_index, int array_type) {
  int array_dim = array_desc->dim;
  _XCALABLEMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XCALABLEMP_template_chunk_t *ti = ai->align_template_chunk;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // pack lo shadow
  if (ai->shadow_size_lo > 0) {
    // FIXME strict condition
    if (ai->shadow_size_lo > ai->par_size) {
      _XCALABLEMP_fatal("shadow size is too big");
    }

    // alloc buffer
    *lo_buffer = _XCALABLEMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts));

    // calc index
    for (int i = 0; i < array_dim; i++) {
      if (i == array_index) {
        // FIXME shadow is allowed in BLOCK distribution
        lower[i] = array_desc->info[i].local_lower;
        upper[i] = lower[i] + array_desc->info[i].shadow_size_lo - 1;
        stride[i] = 1;
      }
      else {
        lower[i] = array_desc->info[i].local_lower;
        upper[i] = array_desc->info[i].local_upper;
        stride[i] = array_desc->info[i].local_stride;
      }

      dim_acc[i] = array_desc->info[i].dim_acc;
    }

    // pack data
    _XCALABLEMP_pack_shadow_buffer(*lo_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
  }

  // pack hi shadow
  if (ai->shadow_size_hi > 0) {
    // FIXME strict condition
    if (ai->shadow_size_hi > ai->par_size) {
      _XCALABLEMP_fatal("shadow size is too big");
    }

    // alloc buffer
    *hi_buffer = _XCALABLEMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts));

    // calc index
    for (int i = 0; i < array_dim; i++) {
      if (i == array_index) {
        // XXX shadow is allowed in BLOCK distribution
        lower[i] = array_desc->info[i].local_upper - array_desc->info[i].shadow_size_hi + 1;
        upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
        stride[i] = 1;
      }
      else {
        lower[i] = array_desc->info[i].local_lower;
        upper[i] = array_desc->info[i].local_upper;
        stride[i] = array_desc->info[i].local_stride;
      }

      dim_acc[i] = array_desc->info[i].dim_acc;
    }

    // pack data
    _XCALABLEMP_pack_shadow_buffer(*hi_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
  }
}

// FIXME not consider full shadow
void _XCALABLEMP_unpack_shadow_NORMAL_BASIC(void *lo_buffer, void *hi_buffer, void *array_addr,
                                            _XCALABLEMP_array_t *array_desc, int array_index, int array_type) {
  int array_dim = array_desc->dim;
  _XCALABLEMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XCALABLEMP_template_chunk_t *ti = ai->align_template_chunk;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // unpack lo shadow
  if (ai->shadow_size_lo > 0) {
    // FIXME strict condition
    if (ai->shadow_size_lo > ai->par_size) {
      _XCALABLEMP_fatal("shadow size is too big");
    }

    // calc index
    for (int i = 0; i < array_dim; i++) {
      if (i == array_index) {
        // XXX shadow is allowed in BLOCK distribution
        lower[i] = 0;
        upper[i] = array_desc->info[i].shadow_size_lo - 1;
        stride[i] = 1;
      }
      else {
        lower[i] = array_desc->info[i].local_lower;
        upper[i] = array_desc->info[i].local_upper;
        stride[i] = array_desc->info[i].local_stride;
      }

      dim_acc[i] = array_desc->info[i].dim_acc;
    }

    // unpack data
    _XCALABLEMP_unpack_shadow_buffer(lo_buffer, array_type, array_dim, lower, upper, stride, dim_acc);

    // free buffer
    _XCALABLEMP_free(lo_buffer);
  }

  // unpack hi shadow
  if (ai->shadow_size_hi > 0) {
    // FIXME strict condition
    if (ai->shadow_size_hi > ai->par_size) {
      _XCALABLEMP_fatal("shadow size is too big");
    }

    // calc index
    for (int i = 0; i < array_dim; i++) {
      if (array_index == 0) {
        // XXX shadow is allowed in BLOCK distribution
        lower[i] = array_desc->info[i].shadow_size_lo + array_desc->info[i].par_size;
        upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
        stride[i] = 1;
      }
      else {
        lower[i] = array_desc->info[i].local_lower;
        upper[i] = array_desc->info[i].local_upper;
        stride[i] = array_desc->info[i].local_stride;
      }

      dim_acc[i] = array_desc->info[i].dim_acc;
    }

    // unpack data
    _XCALABLEMP_unpack_shadow_buffer(hi_buffer, array_type, array_dim, lower, upper, stride, dim_acc);

    // free buffer
    _XCALABLEMP_free(hi_buffer);
  }
}
