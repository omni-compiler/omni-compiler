#include "xmp_internal.h"
extern char ** _xmp_gasnet_buf;
extern int *_xmp_gasnet_stride_queue;
extern size_t _xmp_gasnet_coarray_shift, _xmp_gasnet_stride_size, _xmp_gasnet_heap_size;
static int _xmp_gasnet_stride_wait_size = 0;
static int _xmp_gasnet_stride_queue_size = _XMP_GASNET_STRIDE_INIT_SIZE;
volatile static int done_get_flag;
struct _shift_queue_t{
  unsigned int max_size;   /**< Max size of queue */
  unsigned int      num;   /**< How many shifts are in this queue */
  size_t        *shifts;   /**< shifts array */
};
static struct _shift_queue_t _shift_queue;

void _XMP_gasnet_build_shift_queue()
{
  _shift_queue.max_size = _XMP_GASNET_COARRAY_SHIFT_QUEUE_INITIAL_SIZE;
  _shift_queue.num      = 0;
  _shift_queue.shifts   = malloc(sizeof(size_t*) * _shift_queue.max_size);
}

static void _rebuild_shift_queue()
{
  _shift_queue.max_size *= _XMP_GASNET_COARRAY_SHIFT_QUEUE_INCREMENT_RAITO;
  size_t *tmp;
  size_t next_size = _shift_queue.max_size * sizeof(size_t*);
  if((tmp = realloc(_shift_queue.shifts, next_size)) == NULL)
    _XMP_fatal("cannot allocate memory");
  else
    _shift_queue.shifts = tmp;
}

static void _push_shift_queue(size_t s)
{
  if(_shift_queue.num >= _shift_queue.max_size)
    _rebuild_shift_queue();

  _shift_queue.shifts[_shift_queue.num++] = s;
}

static size_t _pop_shift_queue()
{
  if(_shift_queue.num == 0)  return 0;

  _shift_queue.num--;
  return _shift_queue.shifts[_shift_queue.num];
}

void _XMP_gasnet_coarray_lastly_deallocate(){
  _xmp_gasnet_coarray_shift -= _pop_shift_queue();
}

void _XMP_gasnet_malloc_do(_XMP_coarray_t *coarray, void **addr, const size_t coarray_size)
{
  char **each_addr;  // head address of a local array on each node
  size_t tmp_shift;

  each_addr = _XMP_alloc(sizeof(char *) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++)
    each_addr[i] = (char *)_xmp_gasnet_buf[i] + _xmp_gasnet_coarray_shift;

  if(coarray_size % _XMP_GASNET_ALIGNMENT == 0)
    tmp_shift = coarray_size;
  else{
    tmp_shift = ((coarray_size / _XMP_GASNET_ALIGNMENT) + 1) * _XMP_GASNET_ALIGNMENT;
  }
  _xmp_gasnet_coarray_shift += tmp_shift;
  _push_shift_queue(tmp_shift);

  if(_xmp_gasnet_coarray_shift > _xmp_gasnet_heap_size){
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Cannot allocate coarray. Heap memory size of corray is too small.\n");
      fprintf(stderr, "        Please set the environmental variable \"XMP_ONESIDED_HEAP_SIZE\".\n");
      fprintf(stderr, "        e.g.) export XMP_ONESIDED_HEAP_SIZE=%zuM (or more).\n", (_xmp_gasnet_coarray_shift/1024/1024)+1);
    }
    _XMP_fatal_nomsg();
  }

  coarray->addr = each_addr;
  coarray->real_addr = each_addr[_XMP_world_rank];
  *addr = each_addr[_XMP_world_rank];
}

void _XMP_gasnet_sync_memory()
{
  for(int i=0;i<_xmp_gasnet_stride_wait_size;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[i] == 1);

  _xmp_gasnet_stride_wait_size = 0;

  gasnet_wait_syncnbi_puts();
}

void _XMP_gasnet_sync_all()
{
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

static void _gasnet_c_to_c_put(const int target, const size_t dst_point, 
			       const size_t src_point, const _XMP_coarray_t *dst, 
			       const void *src, const size_t transfer_size)
{
  gasnet_put_nbi_bulk(target, dst->addr[target]+dst_point, ((char *)src)+src_point, 
		      transfer_size);

}

static void _gasnet_nonc_to_c_put(const int target, const size_t dst_point, const int src_dims, 
				  const _XMP_array_section_t *src, const _XMP_coarray_t *dst, const void *src_ptr, 
				  const size_t transfer_size)
{
  char archive[transfer_size];
  _XMP_pack_coarray(archive, src_ptr, src_dims, src);
  _gasnet_c_to_c_put(target, dst_point, 0, dst, archive, transfer_size);
}

void _xmp_gasnet_unpack_reply(gasnet_token_t t, const int ith)
{
  _xmp_gasnet_stride_queue[ith] = 1;
}

static void _extend_stride_queue()
{
  if(_xmp_gasnet_stride_wait_size >= _xmp_gasnet_stride_queue_size){
    int old_size = _xmp_gasnet_stride_wait_size;
    int new_size = old_size + _XMP_GASNET_STRIDE_BLK;
    int *new_list = malloc(sizeof(int) * new_size);
    int *old_list = _xmp_gasnet_stride_queue;
    memcpy(new_list, old_list, sizeof(int) * old_size);
    _xmp_gasnet_stride_queue = new_list;
    _xmp_gasnet_stride_queue_size = new_size;
    free(old_list);
  }
}

void _xmp_gasnet_unpack_using_buf(gasnet_token_t t, const int addr_hi, const int addr_lo, 
				  const int dst_dims, const int ith, const int flag)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  char* src_addr = _xmp_gasnet_buf[_XMP_world_rank];
  memcpy(dst, src_addr, dst_info_size);
  _XMP_unpack_coarray((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst, flag);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

void _xmp_gasnet_unpack(gasnet_token_t t, const char* src_addr, const size_t nbytes, 
			const int addr_hi, const int addr_lo, const int dst_dims, const int ith, const int flag)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  memcpy(dst, src_addr, dst_info_size);
  _XMP_unpack_coarray((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst, flag);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

static void _stride_size_error(size_t request_size){
  if(_XMP_world_rank == 0){
    fprintf(stderr, "[ERROR] Memory size for coarray stride transfer is too small.\n");
    fprintf(stderr, "        Please set the environmental variable \"XMP_COARRAY_STRIDE_SIZE\".\n");
    fprintf(stderr, "        e.g.) export XMP_COARRAY_STRIDE_SIZE=%zuM (or more).\n", (request_size/1024/1024)+1);
  }
  _XMP_fatal_nomsg();
}

static void _gasnet_c_to_nonc_put(const int target, const size_t src_point, const int dst_dims, 
				  const _XMP_array_section_t *dst_info, 
				  const _XMP_coarray_t *dst, const void *src, size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, (char *)src+src_point, transfer_size - dst_info_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target, _XMP_GASNET_UNPACK, archive, (size_t)transfer_size,
			    HIWORD(dst->addr[target]), LOWORD(dst->addr[target]), dst_dims, 
			    _xmp_gasnet_stride_wait_size, 0);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put(target, _xmp_gasnet_buf[target], archive, (size_t)transfer_size);
    gasnet_AMRequestShort5(target, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target]), 
			   LOWORD(dst->addr[target]), dst_dims, _xmp_gasnet_stride_wait_size, 0);
  }
  else{
    _stride_size_error(transfer_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

static void _gasnet_nonc_to_nonc_put(const int target, const int dst_dims, const int src_dims,
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src,
				     const _XMP_coarray_t *dst, const void *src_ptr, size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  _XMP_pack_coarray(archive + dst_info_size, src_ptr, src_dims, src);
  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;

  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target, _XMP_GASNET_UNPACK, archive, transfer_size,
    			    HIWORD(dst->addr[target]), LOWORD(dst->addr[target]), dst_dims,
    			    _xmp_gasnet_stride_wait_size, 0);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put_bulk(target, _xmp_gasnet_buf[target], archive, transfer_size);
    gasnet_AMRequestShort5(target, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target]),
                           LOWORD(dst->addr[target]), dst_dims, _xmp_gasnet_stride_wait_size, 0);
  }
  else{
    _stride_size_error(transfer_size);
  }

  GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] == 1);
}

static void _gasnet_scalar_mput(const int target, const size_t src_point, const int dst_dims,
				const _XMP_array_section_t *dst_info, const _XMP_coarray_t *dst, 
				const void *src, const size_t elmt_size, const size_t num_elmts)
{
  size_t transfer_size = elmt_size;
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, (char *)src+src_point, elmt_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target, _XMP_GASNET_UNPACK, archive, transfer_size,
                            HIWORD(dst->addr[target]), LOWORD(dst->addr[target]), dst_dims,
                            _xmp_gasnet_stride_wait_size, 1);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put(target, _xmp_gasnet_buf[target], archive, transfer_size);
    gasnet_AMRequestShort5(target, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target]),
                           LOWORD(dst->addr[target]), dst_dims, _xmp_gasnet_stride_wait_size, 1);
  }
  else{
    _stride_size_error(transfer_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

/* e.g. dst[:]:[2] = src[:]; The dst must be a coarray. */
void _XMP_gasnet_put(const int dst_continuous, const int src_continuous, const int target, const int dst_dims, 
		     const int src_dims, const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, 
		     const _XMP_coarray_t *dst, const void *src, const size_t dst_elmts, const size_t src_elmts)
{
  if(dst_elmts == src_elmts){
    size_t transfer_size = dst->elmt_size*dst_elmts;
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      size_t dst_point = _XMP_get_offset(dst_info, dst_dims);
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_c_put(target, dst_point, src_point, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
      size_t dst_point = _XMP_get_offset(dst_info, dst_dims);
      _gasnet_nonc_to_c_put(target, dst_point, src_dims, src_info, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_nonc_put(target, src_point, dst_dims, dst_info, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
      _gasnet_nonc_to_nonc_put(target, dst_dims, src_dims, dst_info, src_info, 
			       dst, src, transfer_size);
    }
  }
  else{
    if(src_elmts == 1){
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_scalar_mput(target, src_point, dst_dims, dst_info, dst, src, dst->elmt_size, 
			  dst_elmts);
    }
    else{
      _XMP_fatal("Unkown shape of coarray");
    }
  }
}

static void _gasnet_c_to_c_get(const int target, const size_t dst_point, const size_t src_point, 
			       const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  gasnet_get_bulk(((char *)dst)+dst_point, target, ((char *)src->addr[target])+src_point,
		  transfer_size);

}

static void _gasnet_c_to_nonc_get(const int target, const size_t src_point, const int dst_dims, const _XMP_array_section_t *dst_info, 
				  const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  if(transfer_size < _xmp_gasnet_stride_size){
    char* src_addr = (char *)_xmp_gasnet_buf[_XMP_world_rank];
    gasnet_get_bulk(src_addr, target, ((char *)src->addr[target])+src_point, (size_t)transfer_size);
    _XMP_unpack_coarray(((char *)dst), dst_dims, src_addr, dst_info, 0);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

void _xmp_gasnet_pack(gasnet_token_t t, const char* info, const size_t am_request_size, 
		      const int src_addr_hi, const int src_addr_lo, const int src_dims, 
		      const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _XMP_pack_coarray(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyMedium2(t, _XMP_GASNET_UNPACK_GET_REPLY, archive, tansfer_size,
      			dst_addr_hi, dst_addr_lo);
}

void _xmp_gasnet_pack_get(gasnet_token_t t, const char* info, const size_t am_request_size,
			  const int src_addr_hi, const int src_addr_lo, const int src_dims, const int dst_dims,
			  const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  size_t src_size = sizeof(_XMP_array_section_t) * src_dims;
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *src_info = malloc(src_size);
  memcpy(src_info, info, src_size);
  char archive[tansfer_size + dst_size];
  memcpy(archive, info + src_size, dst_size);
  _XMP_pack_coarray(archive+dst_size, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  free(src_info);
  gasnet_AMReplyMedium3(t, _XMP_GASNET_UNPACK_GET_REPLY_NONC, archive, tansfer_size + dst_size,
                        dst_addr_hi, dst_addr_lo, dst_dims);
}

void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t t, char *archive, size_t transfer_size,
				       const int dst_addr_hi, const int dst_addr_lo, const int dst_dims)
{
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst_info = malloc(dst_size);
  memcpy(dst_info, archive, dst_size);

  _XMP_unpack_coarray((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), dst_dims, archive+dst_size, dst_info, 0);
  done_get_flag = _XMP_N_INT_TRUE;
}


void _xmp_gasnet_unpack_get_reply(gasnet_token_t t, char *archive, size_t transfer_size, 
				  const int dst_addr_hi, const int dst_addr_lo)
{
  memcpy((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), archive, transfer_size);
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t t)
{
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_pack_using_buf(gasnet_token_t t, const char* info, const size_t am_request_size,
				const int src_addr_hi, const int src_addr_lo, const int src_dims,
				const int target)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _XMP_pack_coarray(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyShort0(t, _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF);
}

static void _gasnet_nonc_to_c_get(const int target, const size_t dst_point, const int src_dims, 
				  const _XMP_array_section_t *src_info, 
				  const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
  char archive[am_request_size];  // Note: Info. of transfer_size may have better in "archive".
  memcpy(archive, src_info, am_request_size);

  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // fix me
    gasnet_AMRequestMedium6(target, _XMP_GASNET_PACK, archive, am_request_size,
			    HIWORD(src->addr[target]), LOWORD(src->addr[target]), src_dims,
    			    (size_t)transfer_size, HIWORD((char *)dst+dst_point), LOWORD((char *)dst+dst_point));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_AMRequestMedium4(target, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target]), LOWORD(src->addr[target]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target, _xmp_gasnet_buf[target], transfer_size);
    memcpy(((char *)dst)+dst_point, _xmp_gasnet_buf[_XMP_world_rank], transfer_size);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

static void _gasnet_nonc_to_nonc_get(const int target, const int dst_dims, const int src_dims, 
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, 
				     const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // fix me
    size_t am_request_src_size = sizeof(_XMP_array_section_t) * src_dims;
    size_t am_request_dst_size = sizeof(_XMP_array_section_t) * dst_dims;
    char *archive = malloc(am_request_src_size + am_request_dst_size);
    memcpy(archive, src_info, am_request_src_size);
    memcpy(archive + am_request_src_size, dst_info, am_request_dst_size);
    gasnet_AMRequestMedium7(target, _XMP_GASNET_PACK_GET_HANDLER, archive, 
			    am_request_src_size+am_request_dst_size,
                            HIWORD(src->addr[target]), LOWORD(src->addr[target]), src_dims, dst_dims,
                            (size_t)transfer_size, HIWORD((char *)dst), LOWORD((char *)dst));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    free(archive);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
    char *archive = malloc(am_request_size);
    memcpy(archive, src_info, am_request_size);
    gasnet_AMRequestMedium4(target, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target]), LOWORD(src->addr[target]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target, _xmp_gasnet_buf[target], 
		    transfer_size);
    _XMP_unpack_coarray((char *)dst, dst_dims, _xmp_gasnet_buf[_XMP_world_rank], dst_info, 0);
    free(archive);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

static void _gasnet_scalar_mget(const int target, const size_t src_point, const int dst_dims, const _XMP_array_section_t *dst_info,
				const void *dst, const _XMP_coarray_t *src, const size_t elmt_size, const size_t elmts)
{
  char* src_addr = (char *)_xmp_gasnet_buf[_XMP_world_rank];
  gasnet_get_bulk(src_addr, target, ((char *)src->addr[target])+src_point, elmt_size);
  _XMP_unpack_coarray(((char *)dst), dst_dims, src_addr, dst_info, 1);
}

/* e.g. dst[:] = src[:]:[2]; The src must be a coarray. */
void _XMP_gasnet_get(const int src_continuous, const int dst_continuous, const int target, const int src_dims, 
		     const int dst_dims, const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		     const _XMP_coarray_t *src, const void *dst, const size_t src_elmts, const size_t dst_elmts)
{
  if(src_elmts == dst_elmts){
    size_t transfer_size = src->elmt_size*src_elmts;
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      size_t dst_point = _XMP_get_offset(dst_info, dst_dims);
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_c_get(target, dst_point, src_point, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
      size_t dst_point = _XMP_get_offset(dst_info, dst_dims);
      _gasnet_nonc_to_c_get(target, dst_point, src_dims, src_info, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_nonc_get(target, src_point, dst_dims, dst_info, dst, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
      _gasnet_nonc_to_nonc_get(target, dst_dims, src_dims, dst_info, src_info, dst, src, transfer_size);
    }
  }
  else{
    if(src_elmts == 1){
      size_t src_point = _XMP_get_offset(src_info, src_dims);
      _gasnet_scalar_mget(target, src_point, dst_dims, dst_info, dst, src, src->elmt_size,
			  dst_elmts);
    }
    else{
      _XMP_fatal("Unkown shape of coarray");
    }
  }
}
