#include "xmp_internal.h"
gasnet_hsl_t _hsl;
static uint32_t _atomic_operations = 0;
static int *_xmp_gasnet_atomic_queue;
static int _xmp_gasnet_atomic_queue_size = _XMP_GASNET_ATOMIC_INIT_SIZE;
#define _XMP_ATOMIC_REG  0
#define _XMP_ATOMIC_DONE 1

/**
   Create new atomic queue
*/
static void _extend_atomic_queue()
{
  if(_atomic_operations >= _xmp_gasnet_atomic_queue_size){
    _xmp_gasnet_atomic_queue_size *= _XMP_GASNET_ATOMIC_INCREMENT_RATIO;
    int *tmp;
    int next_size = _xmp_gasnet_atomic_queue_size * sizeof(int);
    if((tmp = realloc(_xmp_gasnet_atomic_queue, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _xmp_gasnet_atomic_queue = tmp;
  }
}

void XMP_gasnet_atomic_sync_memory()
{
  for(int i=0;i<_atomic_operations;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_atomic_queue[i] == _XMP_ATOMIC_DONE);
  
  _atomic_operations = 0;
}

void _XMP_gasnet_intrinsic_initialize()
{
  gasnet_hsl_init(&_hsl);
  _xmp_gasnet_atomic_queue = malloc(sizeof(int) * _XMP_GASNET_ATOMIC_INIT_SIZE);
}

void _xmp_gasnet_atomic_define_do(gasnet_token_t token, const char *src_addr, const size_t elmt_size,
				  gasnet_handlerarg_t addr_hi, gasnet_handlerarg_t addr_lo,
				  gasnet_handlerarg_t local_atomic_operations)
{
  int *dst_addr = (int *)UPCRI_MAKEWORD(addr_hi, addr_lo);

  gasnet_hsl_lock(&_hsl);
  memcpy(dst_addr, src_addr, elmt_size);
  gasnet_hsl_unlock(&_hsl);

  gasnet_AMReplyShort1(token, _XMP_GASNET_ATOMIC_DEFINE_REPLY_DO, local_atomic_operations);
}

void _xmp_gasnet_atomic_define_reply_do(gasnet_token_t token, gasnet_handlerarg_t local_atomic_operations)
{
  _xmp_gasnet_atomic_queue[local_atomic_operations] = _XMP_ATOMIC_DONE;
}

void _XMP_gasnet_atomic_define(int target_rank, _XMP_coarray_t *dst_desc, size_t dst_offset, int value,
			       _XMP_coarray_t *src_desc, size_t src_offset, size_t elmt_size)
{
  char *dst_addr = dst_desc->addr[target_rank] + elmt_size * dst_offset;
  char *src_addr = (src_desc == NULL)? (char*)&value : src_desc->addr[_XMP_world_rank] + elmt_size * src_offset;
  if(target_rank == _XMP_world_rank){
    gasnet_hsl_lock(&_hsl);
    memcpy(dst_addr, src_addr, elmt_size);
    gasnet_hsl_unlock(&_hsl);
  }
  else{
    _extend_atomic_queue();
    _xmp_gasnet_atomic_queue[_atomic_operations] = _XMP_ATOMIC_REG;
    gasnet_AMRequestMedium3(target_rank, _XMP_GASNET_ATOMIC_DEFINE_DO, src_addr, elmt_size, HIWORD(dst_addr), LOWORD(dst_addr), _atomic_operations);
    _atomic_operations++;
  }
}

void _xmp_gasnet_atomic_ref_do(gasnet_token_t token, const size_t elmt_size,
			       gasnet_handlerarg_t src_addr_hi, gasnet_handlerarg_t src_addr_lo,
			       gasnet_handlerarg_t dst_addr_hi, gasnet_handlerarg_t dst_addr_lo,
			       gasnet_handlerarg_t local_atomic_operations)
{
  int *dst_addr = (int *)UPCRI_MAKEWORD(dst_addr_hi, dst_addr_lo);
  gasnet_AMReplyMedium3(token, _XMP_GASNET_ATOMIC_REF_REPLY_DO, dst_addr, elmt_size, src_addr_hi,
			src_addr_lo, local_atomic_operations);
}

void _xmp_gasnet_atomic_ref_reply_do(gasnet_token_t token, int *dst_addr, size_t elmt_size, gasnet_handlerarg_t src_addr_hi,
				     gasnet_handlerarg_t src_addr_lo, gasnet_handlerarg_t local_atomic_operations)
{
  int *src_addr  = (int *)UPCRI_MAKEWORD(src_addr_hi, src_addr_lo);
  gasnet_hsl_lock(&_hsl);
  memcpy(src_addr, dst_addr, elmt_size);
  gasnet_hsl_unlock(&_hsl);
  _xmp_gasnet_atomic_queue[local_atomic_operations] = _XMP_ATOMIC_DONE;
}

void _XMP_gasnet_atomic_ref(int target_rank ,_XMP_coarray_t *dst_desc, size_t dst_offset, int* value,
			    size_t elmt_size)
{
  gasnet_AMPoll();
  char *dst_addr = dst_desc->addr[target_rank] + elmt_size * dst_offset;
  if(target_rank == _XMP_world_rank){
    gasnet_hsl_lock(&_hsl);
    memcpy(value, dst_addr, elmt_size);
    gasnet_hsl_unlock(&_hsl);
  }
  else{
    _extend_atomic_queue();
    _xmp_gasnet_atomic_queue[_atomic_operations] = _XMP_ATOMIC_REG;
    gasnet_AMRequestShort6(target_rank, _XMP_GASNET_ATOMIC_REF_DO, elmt_size,
			   HIWORD(value), LOWORD(value), HIWORD(dst_addr), LOWORD(dst_addr), _atomic_operations);
    _atomic_operations++;
  }
}
