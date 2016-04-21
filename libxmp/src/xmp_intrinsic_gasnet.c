#include "xmp_internal.h"
gasnet_hsl_t _hsl;

void _XMP_gasnet_intrinsic_initialize()
{
  gasnet_hsl_init(&_hsl);
}

void _xmp_gasnet_atomic_define_do(gasnet_token_t token, const char *src_addr, const size_t elmt_size,
				  gasnet_handlerarg_t addr_hi, gasnet_handlerarg_t addr_lo,
				  gasnet_handlerarg_t doneflag_hi, gasnet_handlerarg_t doneflag_lo)
{
  int *dst_addr = (int *)UPCRI_MAKEWORD(addr_hi, addr_lo);

  gasnet_hsl_lock(&_hsl);
  memcpy(dst_addr, src_addr, elmt_size);
  gasnet_hsl_unlock(&_hsl);
  
  gasnet_AMReplyShort2(token, _XMP_GASNET_ATOMIC_DEFINE_REPLY_DO, doneflag_hi, doneflag_lo);
}

void _xmp_gasnet_atomic_define_reply_do(gasnet_token_t token, gasnet_handlerarg_t doneflag_hi,
					gasnet_handlerarg_t doneflag_lo)
{
  int *done_flag = (int *)UPCRI_MAKEWORD(doneflag_hi, doneflag_lo);
  *done_flag = 1;
}

void _XMP_gasnet_atomic_define(int target_rank, _XMP_coarray_t *dst_desc, size_t dst_offset, int value,
			       _XMP_coarray_t *src_desc, size_t src_offset, size_t elmt_size)
{
  char *dst_addr = dst_desc->addr[target_rank] + elmt_size * dst_offset;
  char *src_addr = (src_desc == NULL)? (char*)&value : src_desc->addr[_XMP_world_rank] + elmt_size * src_offset;
  int doneflag = 0;
  gasnet_AMRequestMedium4(target_rank, _XMP_GASNET_ATOMIC_DEFINE_DO, src_addr, elmt_size,
			  HIWORD(dst_addr), LOWORD(dst_addr), HIWORD(&doneflag), LOWORD(&doneflag));
  GASNET_BLOCKUNTIL(doneflag == 1);
}

void _xmp_gasnet_atomic_ref_do(gasnet_token_t token, const size_t elmt_size,
			       gasnet_handlerarg_t src_addr_hi, gasnet_handlerarg_t src_addr_lo,
			       gasnet_handlerarg_t dst_addr_hi, gasnet_handlerarg_t dst_addr_lo,
			       gasnet_handlerarg_t doneflag_hi, gasnet_handlerarg_t doneflag_lo)
{
  int *dst_addr = (int *)UPCRI_MAKEWORD(dst_addr_hi, dst_addr_lo);
  gasnet_AMReplyMedium4(token, _XMP_GASNET_ATOMIC_REF_REPLY_DO, dst_addr, elmt_size, src_addr_hi,
			src_addr_lo, doneflag_hi, doneflag_lo);
}

void _xmp_gasnet_atomic_ref_reply_do(gasnet_token_t token, int *dst_addr, size_t elmt_size,
				     gasnet_handlerarg_t src_addr_hi, gasnet_handlerarg_t src_addr_lo,
				     gasnet_handlerarg_t doneflag_hi, gasnet_handlerarg_t doneflag_lo)
{
  int *src_addr  = (int *)UPCRI_MAKEWORD(src_addr_hi, src_addr_lo);
  gasnet_hsl_lock(&_hsl);
  memcpy(src_addr, dst_addr, elmt_size);
  gasnet_hsl_unlock(&_hsl);

  int *doneflag = (int *)UPCRI_MAKEWORD(doneflag_hi, doneflag_lo);
  *doneflag = 1;
}

void _XMP_gasnet_atomic_ref(int target_rank ,_XMP_coarray_t *dst_desc, size_t dst_offset, int* value,
			    size_t elmt_size)
{
  gasnet_AMPoll();
  char *dst_addr = dst_desc->addr[target_rank] + elmt_size * dst_offset;
  int doneflag = 0;
  gasnet_AMRequestShort7(target_rank, _XMP_GASNET_ATOMIC_REF_DO, elmt_size,
			 HIWORD(value), LOWORD(value), HIWORD(dst_addr), LOWORD(dst_addr),
			 HIWORD(&doneflag), LOWORD(&doneflag));
  GASNET_BLOCKUNTIL(doneflag == 1);
}
