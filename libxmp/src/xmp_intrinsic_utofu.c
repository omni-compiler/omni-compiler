#include "xmp_internal.h"

static uint64_t _XMP_utofu_sync_memory_armw( const uint64_t edata,
                                             const unsigned long int post_flags,
                                             const uintptr_t cbvalue )
{
  int ret;
  uint64_t rmt_value = 0;
  if( post_flags & UTOFU_ONESIDED_FLAG_TCQ_NOTICE ) {
    void *cbdata;
    ret = utofu_poll_tcq(_xmp_utofu_vcq_hdl, 0, &cbdata);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_fatal("_XMP_utofu_sync_memory_armw : utofu_poll_tcq not success");
    }
  }

  if( post_flags & UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE ) {
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_fatal("_XMP_utofu_sync_memory_armw : utofu_poll_mrq not success");
    }
    if( ret == UTOFU_SUCCESS ) {
      rmt_value = _XMP_utofu_check_mrq_notice( &notice );
    }
  }
  return rmt_value;
}


void _XMP_utofu_atomic_define(int target_rank,
                              _XMP_coarray_t *dst_desc, size_t dst_offset, int value,
                              size_t elmt_size)
{
  if( elmt_size != 4 ) {
      _XMP_fatal("_XMP_utofu_atomic_define : invalid elmt_size");
  }

  int ident_val = _xmp_utofu_num_of_puts;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_PUT_POST_FLAGS;
  uint64_t edata = _xmp_utofu_edata_flag_armw_puts;
  uintptr_t cbvalue = ident_val;

  utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];
  enum utofu_armw_op armw_op = UTOFU_ARMW_OP_SWAP;
  utofu_stadd_t rmt_stadd = dst_desc->stadds[target_rank] + elmt_size * dst_offset;

  uint32_t op_value = value;

  _xmp_utofu_num_of_puts++;
  utofu_armw4(_xmp_utofu_vcq_hdl, rmt_vcq_id, armw_op, op_value, rmt_stadd,
              edata, post_flags, (void *)cbvalue);
  while( _xmp_utofu_num_of_puts != 0 )
    _XMP_utofu_sync_memory_armw(edata, post_flags, cbvalue);
}

void _XMP_utofu_atomic_ref(int target_rank,
                           _XMP_coarray_t *dst_desc, size_t dst_offset, int *value,
                           size_t elmt_size)
{
  if( elmt_size != 4 ) {
      _XMP_fatal("_XMP_utofu_atomic_ref : invalid elmt_size");
  }

  int ident_val = _xmp_utofu_num_of_gets;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_GET_POST_FLAGS;
  uint64_t edata = _xmp_utofu_edata_flag_armw_gets;
  uintptr_t cbvalue = ident_val;

  utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];
  enum utofu_armw_op armw_op = UTOFU_ARMW_OP_OR;
  utofu_stadd_t rmt_stadd = dst_desc->stadds[target_rank] + elmt_size * dst_offset;

  uint32_t op_value = 0;

  _xmp_utofu_num_of_gets++;
  utofu_armw4(_xmp_utofu_vcq_hdl, rmt_vcq_id, armw_op, op_value, rmt_stadd,
              edata, post_flags, (void *)cbvalue);
  while( _xmp_utofu_num_of_gets != 0 )
    *value = _XMP_utofu_sync_memory_armw(edata, post_flags, cbvalue);
}
