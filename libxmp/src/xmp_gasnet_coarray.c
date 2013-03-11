#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_atomic.h"
static int *_xmp_gasnet_stride_queue;
static int _xmp_gasnet_stride_wait_size = 0;
static int _xmp_gasnet_stride_queue_size = _XMP_GASNET_STRIDE_INIT_SIZE;
static unsigned long long _xmp_coarray_shift = 0;
static char **_xmp_gasnet_buf;

gasnet_handlerentry_t htable[] = {
  { _XMP_GASNET_LOCK_REQUEST,             _xmp_gasnet_lock_request },
  { _XMP_GASNET_SETLOCKSTATE,             _xmp_gasnet_setlockstate },
  { _XMP_GASNET_UNLOCK_REQUEST,           _xmp_gasnet_unlock_request },
  { _XMP_GASNET_LOCKHANDOFF,              _xmp_gasnet_lockhandoff },
  { _XMP_GASNET_POST_REQUEST,             _xmp_gasnet_post_request },
  { _XMP_GASNET_UNPACK_HANDLER,           _xmp_gasnet_unpack },
  { _XMP_GASNET_UNPACK_HANDLER_USING_BUF, _xmp_gasnet_unpack_using_buf },
  { _XMP_GASNET_UNPACK_REPLY,             _xmp_gasnet_unpack_reply },
  { _XMP_GASNET_PACK_HANDLER,             _xmp_gasnet_pack }
};

void _XMP_gasnet_set_coarray(_XMP_coarray_t *coarray, void **addr, unsigned long long num_of_elmts, size_t elmt_size){
  int numprocs;
  char **each_addr;  // head address of a local array on each node

  numprocs = gasnet_nodes();
  each_addr = (char **)_XMP_alloc(sizeof(char *) * numprocs);

  gasnet_node_t i;
  for(i=0;i<numprocs;i++)
    each_addr[i] = (char *)(_xmp_gasnet_buf[i]) + _xmp_coarray_shift;

    if(elmt_size % _XMP_GASNET_ALIGNMENT == 0)
      _xmp_coarray_shift += elmt_size * num_of_elmts;
    else{
      int tmp = ((elmt_size / _XMP_GASNET_ALIGNMENT) + 1) * _XMP_GASNET_ALIGNMENT;
      _xmp_coarray_shift += tmp * num_of_elmts;
    }
    
  if(_xmp_coarray_shift > _xmp_heap_size){
    if(gasnet_mynode() == 0){
      fprintf(stderr, "Cannot allocate coarray. Now HEAP SIZE of coarray is %d MB\n", (int)(_xmp_heap_size/1024/1024));
      fprintf(stderr, "But %d MB is needed\n", (int)(_xmp_coarray_shift/1024/1024));
    }
    _XMP_fatal("Please set XMP_COARRAY_HEAP_SIZE=<number>\n");
  }

  coarray->addr = each_addr;
  coarray->elmt_size = elmt_size;

  *addr = each_addr[gasnet_mynode()];
}

void _XMP_gasnet_initialize(int argc, char **argv){
  int numprocs;
  unsigned long long xmp_heap_size;

  gasnet_init(&argc, &argv);

  if(_xmp_heap_size % GASNET_PAGESIZE != 0)
    xmp_heap_size = (_xmp_heap_size/GASNET_PAGESIZE -1) * GASNET_PAGESIZE;
  else
    xmp_heap_size = _xmp_heap_size;

  gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), xmp_heap_size, 0); 
  numprocs = gasnet_nodes();

  _xmp_gasnet_buf = (char **)malloc(sizeof(char*) * numprocs);

  gasnet_node_t i;
  gasnet_seginfo_t *s = (gasnet_seginfo_t *)malloc(gasnet_nodes()*sizeof(gasnet_seginfo_t)); 
  gasnet_getSegmentInfo(s, gasnet_nodes());
  for(i=0;i<numprocs;i++)
    _xmp_gasnet_buf[i] =  (char*)s[i].addr;

  _xmp_coarray_shift = _xmp_stride_size;
  _xmp_gasnet_stride_queue = malloc(sizeof(int) * _XMP_GASNET_STRIDE_INIT_SIZE);
}

void _XMP_gasnet_finalize(int val){
  _XMP_gasnet_sync_all();
  gasnet_exit(val);
}

void _XMP_gasnet_sync_memory(){
  int i;

  for(i=0;i<_xmp_gasnet_stride_wait_size;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[i] == 1);

  _xmp_gasnet_stride_wait_size = 0;

  gasnet_wait_syncnbi_puts();
}

void _XMP_gasnet_sync_all(){
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

static long long get_offset(_XMP_array_section_t *array, int dims){
  int i;
  long long offset = 0;
  for(i=0;i<dims;i++)
    offset += (array+i)->start * (array+i)->distance;

  return offset;
}

static void XMP_gasnet_c_put(const int target_image, const long long dest_point, const long long src_point,
			      const _XMP_coarray_t *dest, const void *src, const long long transfer_size){

  gasnet_put_nbi_bulk(target_image, dest->addr[target_image]+dest_point, ((char *)src)+src_point, transfer_size);

}

// How depth is memory continuity ?
static int get_depth(int dims, const _XMP_array_section_t* array_info){
  if(dims == 1)
    return 0;

  int i;
  int continuous_dim = dims - 2;

  for(i=dims-2;i>=0;i--){
    if(array_info[i+1].start == 0 && array_info[i+1].length == array_info[i+1].size &&
       array_info[i+1].stride == 1 && array_info[i].stride == 1){
      continuous_dim = i;
      continue;
    }
    else{
      continuous_dim++;
      break;
    }
  }

  return continuous_dim;
}

// Perhap, this function had better exist in xmp_lib.c
static void XMP_pack(char* archive_ptr, const char* src_ptr, const int dst_dims,
		     const int src_dims, const _XMP_array_section_t* src_info, 
		     const _XMP_array_section_t* dst_info, long long archive_offset){

  int i;
  size_t element_size = src_info[src_dims-1].distance;
  int index[src_dims+1], d = 1;                // d is a position of nested loop
  for(i=0;i<src_dims+1;i++)   index[i] = 0;    // Initialize index
  long long cnt[src_dims], src_offset;
  cnt[0] = 0;

  // How depth is memory continuity ?
  int continuous_dim = get_depth(src_dims, src_info);

  if(src_info[src_dims-1].stride != 1 || continuous_dim+1 == src_dims){
    while(index[0]==0){
      if(index[d]>=src_info[d-1].length){    // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < src_dims){                 // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*src_info[d-1].stride+src_info[d-1].start) * src_info[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == src_dims){                // the innermost loop
        src_offset = cnt[d-1] + (index[d]*src_info[d-1].stride+src_info[d-1].start) * src_info[d-1].distance;
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
        archive_offset += element_size;
        index[d]++;
      }
    }
  }
  else{
    while(index[0]==0){
      if(index[d]>=src_info[d-1].length){    // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < continuous_dim+1){         // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*src_info[d-1].stride+src_info[d-1].start) * src_info[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == continuous_dim+1){        // the innermost loop
        src_offset = cnt[d-1] + (index[d]*src_info[d-1].stride+src_info[d-1].start) * src_info[d-1].distance;
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset + (src_info[d].start * element_size),
               src_info[d].length * src_info[d].distance);
        archive_offset += src_info[d].length * src_info[d].distance;
        index[d]++;
      }
    }
  }
}

static void XMP_gasnet_from_nonc_to_c_put(int target_image, long long dst_point, int dst_dims, int src_dims, 
					  _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
					  _XMP_coarray_t *dst, void *src, long long transfer_size){
  char* archive = malloc(transfer_size);
  XMP_pack(archive, src, dst_dims, src_dims, src_info, dst_info, 0);
  XMP_gasnet_c_put(target_image, dst_point, (long long)0, dst, archive, transfer_size);
  free(archive);
}

void _xmp_gasnet_unpack_reply(gasnet_token_t t, const int ith){
  _xmp_gasnet_stride_queue[ith] = 1;
}

static void extend_stride_queue(){
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

static void unpack(char *dst_addr, const int dst_dims, const char* src_addr, 
		   const int continuous_dim, _XMP_array_section_t* dst, long long src_offset){
  unsigned long long dst_offset = 0;
  
  size_t element_size = dst[dst_dims-1].distance;
  int index[dst_dims+1], d = 1, i;                // d is a position of nested loop
  for(i=0;i<dst_dims+1;i++)   index[i] = 0;       // Initialize index
  unsigned long long cnt[dst_dims];
  cnt[0] = 0;

  if(dst[dst_dims-1].stride != 1 || continuous_dim+1 == dst_dims){
    while(index[0]==0){
      if(index[d]>=dst[d-1].length){          // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < dst_dims){                  // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == dst_dims){                 // the innermost loop
        dst_offset = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        memcpy(dst_addr + dst_offset, src_addr + src_offset, element_size);
        src_offset += element_size;
        index[d]++;
      }
    }
  }
  else{
    while(index[0]==0){
      if(index[d]>=dst[d-1].length){           // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < continuous_dim+1){           // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == continuous_dim+1){          // the innermost loop
        dst_offset = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        memcpy(dst_addr + dst_offset + (dst[d].start * element_size), src_addr + src_offset,
               dst[d].length * dst[d].distance);
        src_offset += dst[d].length * dst[d].distance;
        index[d]++;
      }
    }
  }
}

void _xmp_gasnet_unpack_using_buf(gasnet_token_t t, const int addr_hi, const int addr_lo, 
				  const int dst_dims, const int ith){

  _XMP_array_section_t *dst = malloc(sizeof(_XMP_array_section_t) * dst_dims);
  char* src_addr = _xmp_gasnet_buf[gasnet_mynode()];
  memcpy(dst, src_addr, sizeof(_XMP_array_section_t)*dst_dims);

  int continuous_dim = get_depth(dst_dims, dst);
  unpack((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr, continuous_dim, dst,
	 sizeof(_XMP_array_section_t) * dst_dims);

  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

void _xmp_gasnet_unpack(gasnet_token_t t, const char* src_addr, const size_t nbytes, 
			const int addr_hi, const int addr_lo, const int dst_dims, const int ith){

  _XMP_array_section_t *dst = malloc(sizeof(_XMP_array_section_t) * dst_dims); 
  memcpy(dst, src_addr, sizeof(_XMP_array_section_t)*dst_dims);
  
  int continuous_dim = get_depth(dst_dims, dst);
  unpack((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr, continuous_dim, dst, 
	 sizeof(_XMP_array_section_t) * dst_dims);

  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

static void coarray_stride_size_error(){
  fprintf(stderr, "Corray stride transfer size is too big\n");
  fprintf(stderr, "Reconfigure environmental variant BUFFER_FOR_STRIDE_SIZE > %lld\n", _xmp_stride_size);
  _XMP_fatal("");
}

static void XMP_gasnet_from_c_to_nonc_put(int target_image, long long src_point, int dst_dims, 
					  _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
                                          _XMP_coarray_t *dst, void *src, long long transfer_size){

  long long tmp_transfer_size = transfer_size;
  transfer_size += sizeof(_XMP_array_section_t) * dst_dims;
  char *archive = malloc(transfer_size);
  memcpy(archive, dst_info, sizeof(_XMP_array_section_t) * dst_dims);
  memcpy(archive+sizeof(_XMP_array_section_t)*dst_dims, (char *)src+src_point, tmp_transfer_size);

  extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK_HANDLER, archive, (size_t)transfer_size,
			    HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims, 
			    _xmp_gasnet_stride_wait_size);
  }
  else if(transfer_size < _xmp_stride_size){
    gasnet_put(target_image, _xmp_gasnet_buf[target_image], archive, (size_t)transfer_size);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_HANDLER_USING_BUF, HIWORD(dst->addr[target_image]), 
			   LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    coarray_stride_size_error();
  }
  _xmp_gasnet_stride_wait_size++;
  free(archive);
}

static void XMP_gasnet_from_nonc_to_nonc_put(int target_image, long long src_point, int dst_dims, int src_dims,
					     _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
					     _XMP_coarray_t *dst, void *src, long long transfer_size){
  transfer_size += sizeof(_XMP_array_section_t) * dst_dims;
  char *archive = malloc(transfer_size);
  memcpy(archive, dst_info, sizeof(_XMP_array_section_t) * dst_dims);
  XMP_pack(archive, src, dst_dims, src_dims, src_info, dst_info, sizeof(_XMP_array_section_t) * dst_dims);

  extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK_HANDLER, archive, (size_t)transfer_size,
                            HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims,
                            _xmp_gasnet_stride_wait_size);
  }
  else if(transfer_size < _xmp_stride_size){
    gasnet_put(target_image, _xmp_gasnet_buf[target_image], archive, (size_t)transfer_size);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_HANDLER_USING_BUF, HIWORD(dst->addr[target_image]),
                           LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    coarray_stride_size_error();
  }

  free(archive);

  GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] == 1);
}

void _XMP_gasnet_put(int dst_continuous, int src_continuous, int target_image, int dst_dims, 
		     int src_dims, _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
		     _XMP_coarray_t *dst, void *src, long long length){

  long long transfer_size = dst->elmt_size*length;
  long long dst_point = get_offset(dst_info, dst_dims);
  long long src_point  = get_offset(src_info, src_dims);

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    XMP_gasnet_c_put(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    XMP_gasnet_from_nonc_to_c_put(target_image, dst_point, dst_dims, src_dims, 
				  dst_info, src_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    XMP_gasnet_from_c_to_nonc_put(target_image, src_point, dst_dims, dst_info, 
				  src_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
    XMP_gasnet_from_nonc_to_nonc_put(target_image, src_point, dst_dims, src_dims, dst_info,
				     src_info, dst, src, transfer_size);
  }
}

static void XMP_gasnet_c_get(const int target_image, const long long dst_point, const long long src_point,
			     const void *dst, const _XMP_coarray_t *src, const long long transfer_size){

  gasnet_get_bulk(((char *)dst)+dst_point, target_image, ((char *)src->addr[target_image])+src_point,
		  transfer_size);

}

static void XMP_gasnet_from_c_to_nonc_get(int target_image, long long src_point, int dst_dims, _XMP_array_section_t *dst_info, 
					  void *dst, _XMP_coarray_t *src, long long transfer_size){
  if(transfer_size < _xmp_stride_size){
    char* src_addr = (char *)_xmp_gasnet_buf[gasnet_mynode()];
    gasnet_get_bulk(src_addr, target_image, ((char *)src->addr[target_image])+src_point, transfer_size);
    int continuous_dim = get_depth(dst_dims, dst_info);
    unpack(((char *)dst), dst_dims, src_addr, continuous_dim, dst_info, 0);
  }
  else{
    coarray_stride_size_error();
  }
}


void _xmp_gasnet_pack(gasnet_token_t t, const char* archive, const size_t transfer_size, 
		      const int addr_hi, const int addr_lo, const int src_dims){
  //  _XMP_array_section_t *src_info = malloc(sizeof(_XMP_array_section_t) * src_dims);
  //  memcpy(src_info, archive, sizeof(_XMP_array_section_t) * src_dims);
  

  //  free(src_info);
}
#ifdef _AA
static void XMP_gasnet_from_nonc_to_c_get(int target_image, int src_dims, _XMP_array_section_t *dst_info,
                                          void *dst, _XMP_coarray_t *src, long long transfer_size){

  char *archive = malloc(sizeof(_XMP_array_section_t) * src_dims);
  memcpy(archive, src_info, sizeof(_XMP_array_section_t) * src_dims);

  int done_flag = _XMP_N_INT_FALSE;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium3(target_image, _XMP_GASNET_PACK_HANDLER, archive, (sizeof(_XMP_array_section_t) * src_dims),
			    HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims);
  }
  else if(transfer_size < _xmp_stride_size){
  }
  else{
    coarray_stride_size_error();
  }
  free(archive);
}
#endif
void _XMP_gasnet_get(int src_continuous, int dst_continuous, int target_image, int src_dims, 
		     int dst_dims, _XMP_array_section_t *src_info, _XMP_array_section_t *dst_info, 
		     _XMP_coarray_t *src, void *dst, long long length){

  long long transfer_size = src->elmt_size*length;
  long long dst_point = get_offset(dst_info, dst_dims);
  long long src_point = get_offset(src_info, src_dims);

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    XMP_gasnet_c_get(target_image, dst_point, src_point, dst, src, transfer_size);
  }
#ifdef _AA
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    XMP_gasnet_from_nonc_to_c_get(target_image, dst_point, dst_dims, src_dims,
  				  dst_info, src_info, dst, src, transfer_size);
  }
#endif
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    XMP_gasnet_from_c_to_nonc_get(target_image, src_point, dst_dims, dst_info, 
				  dst, src, transfer_size);
  }
}

