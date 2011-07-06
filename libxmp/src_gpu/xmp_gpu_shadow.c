/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmp_constant.h"
#include "xmp_internal.h"

extern void _XMP_gpu_pack_array(_XMP_gpu_array_t *device_desc, void *host_shadow_buffer, void *gpu_array_addr,
                                size_t type_size, size_t alloc_size, int array_dim,
                                int *lower, int *upper, int *stride);
extern void _XMP_gpu_unpack_array(_XMP_gpu_array_t *device_desc, void *gpu_array_addr, void *host_shadow_buffer,
                                  size_t type_size, size_t alloc_size, int array_dim,
                                  int *lower, int *upper, int *stride);

void _XMP_gpu_pack_shadow_NORMAL(_XMP_gpu_data_t *desc, void **lo_buffer, void **hi_buffer, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  _XMP_array_t *array_desc = desc->host_array_desc;
  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  void *device_array_addr = desc->device_addr;
  int array_dim = array_desc->dim;
  size_t array_type_size = array_desc->type_size;

  int lower[array_dim], upper[array_dim], stride[array_dim];

  // pack lo shadow
  if (rank != (size - 1)) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // alloc buffer
      size_t alloc_size = (ai->shadow_size_lo) * (ai->dim_elmts) * (array_type_size);
      *lo_buffer = _XMP_alloc(alloc_size);

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
      }

      // pack data
      _XMP_gpu_pack_array(desc->device_array_desc, *lo_buffer, device_array_addr,
                          array_type_size, alloc_size, array_dim,
                          lower, upper, stride);
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
      size_t alloc_size = (ai->shadow_size_hi) * (ai->dim_elmts) * (array_type_size);
      *hi_buffer = _XMP_alloc(alloc_size);

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
      }

      // pack data
      _XMP_gpu_pack_array(desc->device_array_desc, *hi_buffer, device_array_addr,
                          array_type_size, alloc_size, array_dim,
                          lower, upper, stride);
    }
  }
}

void _XMP_gpu_unpack_shadow_NORMAL(_XMP_gpu_data_t *desc, void *lo_buffer, void *hi_buffer, int array_index) {
  _XMP_RETURN_IF_SINGLE;

  _XMP_array_t *array_desc = desc->host_array_desc;
  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  void *device_array_addr = desc->device_addr;
  int array_dim = array_desc->dim;
  size_t array_type_size = array_desc->type_size;

  int lower[array_dim], upper[array_dim], stride[array_dim];

  // unpack lo shadow
  if (rank != 0) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      size_t alloc_size = (ai->shadow_size_lo) * (ai->dim_elmts) * (array_type_size);

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
      }

      // unpack data
      _XMP_gpu_unpack_array(desc->device_array_desc, device_array_addr, lo_buffer,
                            array_type_size, alloc_size, array_dim,
                            lower, upper, stride);

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

      size_t alloc_size = (ai->shadow_size_hi) * (ai->dim_elmts) * (array_type_size);

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
      }

      // unpack data
      _XMP_gpu_unpack_array(desc->device_array_desc, device_array_addr, hi_buffer,
                            array_type_size, alloc_size, array_dim,
                            lower, upper, stride);

      // free buffer
      _XMP_free(hi_buffer);
    }
  }
}
