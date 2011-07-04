/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_constant.h"
#include "xmp_internal.h"

void _XMP_gpu_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr,
                                 _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  int array_type = array_desc->type;
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // pack lo shadow
  if (rank != (size - 1)) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // alloc buffer
      *lo_buffer = _XMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts) * (array_desc->type_size));

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_upper - array_desc->info[i].shadow_size_lo + 1;
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
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(*lo_buffer, array_addr, array_desc->type_size,
                                       array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_pack_array_BASIC(*lo_buffer, array_addr, array_type, array_dim, lower, upper, stride, dim_acc);
      }
    }
  }

  // pack hi shadow
  if (rank != 0) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // alloc buffer
      *hi_buffer = _XMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts) * (array_desc->type_size));

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_lower;
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
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_pack_array_GENERAL(*hi_buffer, array_addr, array_desc->type_size,
                                       array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_pack_array_BASIC(*hi_buffer, array_addr, array_type, array_dim, lower, upper, stride, dim_acc);
      }
    }
  }
}

void _XMP_gpu_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr,
                                   _XMP_array_t *array_desc, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  int array_type = array_desc->type;
  int array_dim = array_desc->dim;
  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  int rank = ai->shadow_comm_rank;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // unpack lo shadow
  if (rank != 0) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
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
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(array_addr, lo_buffer, array_desc->type_size,
                                         array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_unpack_array_BASIC(array_addr, lo_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
      }

      // free buffer
      _XMP_free(lo_buffer);
    }
  }

  // unpack hi shadow
  if (rank != (size - 1)) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_upper + 1;
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
      if (array_type == _XMP_N_TYPE_NONBASIC) {
        _XMP_unpack_array_GENERAL(array_addr, hi_buffer, array_desc->type_size,
                                         array_dim, lower, upper, stride, dim_acc);
      }
      else {
        _XMP_unpack_array_BASIC(array_addr, hi_buffer, array_type, array_dim, lower, upper, stride, dim_acc);
      }

      // free buffer
      _XMP_free(hi_buffer);
    }
  }
}
