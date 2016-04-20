#include "xmp_internal.h"
gasnet_hsl_t _hsl;
volatile static int _doneflag_define, _doneflag_ref;

void _XMP_gasnet_intrinsic_initialize()
{
  gasnet_hsl_init(&_hsl);
}

void _xmp_gasnet_atomic_define_do(gasnet_token_t token, const char *src_addr, const size_t elmt_size,
				  const int addr_hi, const int addr_lo)
{
  gasnet_hsl_lock(&_hsl);
  int *dst_addr = (int *)UPCRI_MAKEWORD(addr_hi, addr_lo);
  memcpy(dst_addr, src_addr, elmt_size);
  gasnet_hsl_unlock(&_hsl);
  gasnet_AMReplyShort0(token, _XMP_GASNET_ATOMIC_DEFINE_REPLY_DO);
}

void _xmp_gasnet_atomic_define_reply_do(gasnet_token_t token)
{
  gasnet_hsl_lock(&_hsl);
  _doneflag_define = 1;
  gasnet_hsl_unlock(&_hsl);
}

void _XMP_gasnet_atomic_define(int target_rank, _XMP_coarray_t *c, size_t offset, int value, size_t elmt_size)
{
  char *dst_addr = c->addr[target_rank] + elmt_size * offset;
  if(target_rank == _XMP_world_rank){
    gasnet_hsl_lock(&_hsl);
    memcpy(dst_addr, &value, elmt_size);
    gasnet_hsl_unlock(&_hsl);
  }
  else{
    _doneflag_define = 0;
    gasnet_AMRequestMedium2(target_rank, _XMP_GASNET_ATOMIC_DEFINE_DO, &value, elmt_size,
			    HIWORD(dst_addr), LOWORD(dst_addr));
    GASNET_BLOCKUNTIL(_doneflag_define == 1);
  }
}

void _xmp_gasnet_atomic_ref_do(gasnet_token_t token, const size_t elmt_size,
			       const int src_addr_hi, const int src_addr_lo,
			       const int dst_addr_hi, const int dst_addr_lo)
{
  int *dst_addr = (int *)UPCRI_MAKEWORD(dst_addr_hi, dst_addr_lo);
  gasnet_AMReplyMedium2(token, _XMP_GASNET_ATOMIC_REF_REPLY_DO, dst_addr, elmt_size, src_addr_hi,
			src_addr_lo);
}

void _xmp_gasnet_atomic_ref_reply_do(gasnet_token_t token, int *dst_addr, size_t elmt_size,
				     const int src_addr_hi, const int src_addr_lo)
{
  gasnet_hsl_lock(&_hsl);
  int *src_addr = (int *)UPCRI_MAKEWORD(src_addr_hi, src_addr_lo);
  memcpy(src_addr, dst_addr, elmt_size);
  _doneflag_ref = 1;
  gasnet_hsl_unlock(&_hsl);
}

void _XMP_gasnet_atomic_ref(int target_rank, _XMP_coarray_t *c, size_t offset, int *value, size_t elmt_size)
{
  gasnet_AMPoll();
  char *dst_addr = c->addr[target_rank]+elmt_size*offset;
  _doneflag_ref = 0;
  gasnet_AMRequestShort5(target_rank, _XMP_GASNET_ATOMIC_REF_DO, elmt_size,
			 HIWORD(value), LOWORD(value), HIWORD(dst_addr), LOWORD(dst_addr));
  GASNET_BLOCKUNTIL(_doneflag_ref == 1);
}
