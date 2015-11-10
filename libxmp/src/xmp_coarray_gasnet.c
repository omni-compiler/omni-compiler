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
static struct _shift_queue_t _shift_queue; /** Queue which saves shift information */
#define _XMP_STRIDE_REG  0
#define _XMP_STRIDE_DONE 1
static unsigned int *_sync_images_table;
static gasnet_hsl_t _hsl;

/**
   Set initial value to the shift queue
 */
void _XMP_gasnet_build_shift_queue()
{
  _shift_queue.max_size = _XMP_GASNET_COARRAY_SHIFT_QUEUE_INITIAL_SIZE;
  _shift_queue.num      = 0;
  _shift_queue.shifts   = malloc(sizeof(size_t*) * _shift_queue.max_size);
}

/**
   Create new shift queue
 */
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

/**
   Push shift information to the shift queue
 */
static void _push_shift_queue(size_t s)
{
  if(_shift_queue.num >= _shift_queue.max_size)
    _rebuild_shift_queue();

  _shift_queue.shifts[_shift_queue.num++] = s;
}

/**
   Pop shift information from the shift queue
 */
static size_t _pop_shift_queue()
{
  if(_shift_queue.num == 0)  return 0;

  _shift_queue.num--;
  return _shift_queue.shifts[_shift_queue.num];
}

/**
   Deallocate memory region when calling _XMP_coarray_lastly_deallocate()
*/
void _XMP_gasnet_coarray_lastly_deallocate(){
  _xmp_gasnet_coarray_shift -= _pop_shift_queue();
}

/**********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                 */
/* ARGUMENT    : [OUT] *coarray_desc : Descriptor of new coarray      */
/*               [OUT] **addr        : Double pointer of new coarray  */
/*               [IN] coarray_size   : Coarray size                   */
/**********************************************************************/
void _XMP_gasnet_malloc_do(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size)
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
      fprintf(stderr, "        e.g.) export XMP_ONESIDED_HEAP_SIZE=%zuM (or more).\n",
	      (_xmp_gasnet_coarray_shift/1024/1024)+1);
    }
    _XMP_fatal_nomsg();
  }

  coarray_desc->addr = each_addr;
  coarray_desc->real_addr = each_addr[_XMP_world_rank];
  *addr = each_addr[_XMP_world_rank];
}

/**
   Execute sync_memory
 */
void _XMP_gasnet_sync_memory()
{
  for(int i=0;i<_xmp_gasnet_stride_wait_size;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[i] == _XMP_STRIDE_DONE);

  _xmp_gasnet_stride_wait_size = 0;

  gasnet_wait_syncnbi_puts();
}

/**
   Execute sync_all
 */
void _XMP_gasnet_sync_all()
{
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

/*************************************************************************************/
/* DESCRIPTION : Execute put operation (from continuous region to continuous region) */
/* ARGUMENT    : [IN] target_rank    : Target rank                                   */
/*               [IN] dst_offset     : Offset size of destination array              */
/*               [IN] src_offset     : Offset size of source array                   */
/*               [OUT] *dst_desc     : Descriptor of destination coarray             */
/*               [IN] *src           : Pointer of source array                       */
/*               [IN] transfer_size  : Transfer size                                 */
/* EXAMPLE    :                                                                      */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src                        */
/*************************************************************************************/
static void _gasnet_c_to_c_put(const int target_rank, const size_t dst_offset, 
			       const size_t src_offset, const _XMP_coarray_t *dst_desc, 
			       const void *src, const size_t transfer_size)
{
  gasnet_put_nbi_bulk(target_rank, dst_desc->addr[target_rank]+dst_offset, (char *)src+src_offset, 
		      transfer_size);
}

/*****************************************************************************************/
/* DESCRIPTION : Execute put operation (from NON-continuous region to continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                        */
/*               [IN] dst_offset    : Offset size of destination array                   */
/*               [IN] src_dims      : Number of dimensions of source array               */
/*               [IN] *src_info     : Information of source array                        */
/*               [IN] *dst_desc     : Descriptor of destination coarray                  */
/*               [IN] *src          : Pointer of source array                            */
/*               [IN] transfer_size : Transfer size                                      */
/* EXAMPLE    :                                                                          */
/*     a[0:100]:[1] = b[0:100:2]; // a[] is a dst, b[] is a src                          */
/*****************************************************************************************/
static void _gasnet_nonc_to_c_put(const int target_rank, const size_t dst_offset, const int src_dims, 
				  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
				  const void *src, const size_t transfer_size)
{
  char archive[transfer_size];
  _XMP_pack_coarray(archive, src, src_dims, src_info);
  _gasnet_c_to_c_put(target_rank, dst_offset, 0, dst_desc, archive, transfer_size);
}

/**
   Registor finish information of unpack operation 
*/
void _xmp_gasnet_unpack_reply(gasnet_token_t t, const int ith)
{
  _xmp_gasnet_stride_queue[ith] = _XMP_STRIDE_DONE;
}

/**
   Create new stride queue
 */
static void _extend_stride_queue()
{
  if(_xmp_gasnet_stride_wait_size >= _xmp_gasnet_stride_queue_size){
    _xmp_gasnet_stride_queue_size *= _XMP_GASNET_STRIDE_INCREMENT_RATIO;
    int *tmp;
    int next_size = _xmp_gasnet_stride_queue_size * sizeof(int);
    if((tmp = realloc(_xmp_gasnet_stride_queue, next_size)) == NULL)
      _XMP_fatal("cannot allocate memory");
    else
      _xmp_gasnet_stride_queue = tmp;
  }
}

/**
   Unpack received data which is stored in buffer
 */
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

/**
   Unpack received data
 */
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

/**
   Output error message
 */
static void _stride_size_error(size_t request_size){
  if(_XMP_world_rank == 0){
    fprintf(stderr, "[ERROR] Memory size for coarray stride transfer is too small.\n");
    fprintf(stderr, "        Please set the environmental variable \"XMP_COARRAY_STRIDE_SIZE\".\n");
    fprintf(stderr, "        e.g.) export XMP_COARRAY_STRIDE_SIZE=%zuM (or more).\n", (request_size/1024/1024)+1);
  }
  _XMP_fatal_nomsg();
}

/*****************************************************************************************/
/* DESCRIPTION : Execute put operation (from continuous region to NON-continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                        */
/*               [IN] src_offset    : Offset size of source array                        */
/*               [IN] dst_dims      : Number of dimensions of destination array          */
/*               [IN] *dst_info     : Information of destination array                   */
/*               [IN] *dst_desc     : Descriptor of destination coarray                  */
/*               [IN] *src          : Pointer of source array                            */
/*               [IN] transfer_size : Transfer size                                      */
/* EXAMPLE    :                                                                          */
/*     a[0:100:2]:[1] = b[0:100]; // a[] is a dst, b[] is a src                          */
/*****************************************************************************************/
static void _gasnet_c_to_nonc_put(const int target_rank, const size_t src_offset, const int dst_dims, 
				  const _XMP_array_section_t *dst_info, 
				  const _XMP_coarray_t *dst_desc, const void *src, size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, (char *)src+src_offset, transfer_size - dst_info_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = _XMP_STRIDE_REG;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target_rank, _XMP_GASNET_UNPACK, archive, transfer_size,
			    HIWORD(dst_desc->addr[target_rank]), LOWORD(dst_desc->addr[target_rank]), dst_dims, 
			    _xmp_gasnet_stride_wait_size, _XMP_UNPACK);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put(target_rank, _xmp_gasnet_buf[target_rank], archive, transfer_size);
    gasnet_AMRequestShort5(target_rank, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst_desc->addr[target_rank]), 
			   LOWORD(dst_desc->addr[target_rank]), dst_dims, _xmp_gasnet_stride_wait_size, _XMP_UNPACK);
  }
  else{
    _stride_size_error(transfer_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

/*********************************************************************************************/
/* DESCRIPTION : Execute put operation (from NON-continuous region to NON-continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                            */
/*               [IN] dst_dims      : Number of dimensions of destination array              */
/*               [IN] src_dims      : Number of dimensions of source array                   */
/*               [IN] *dst_info     : Information of destination array                       */
/*               [IN] *src_info     : Information of source array                            */
/*               [OUT] *dst_desc    : Descriptor of destination coarray                      */
/*               [IN] *src          : Pointer of source array                                */
/*               [IN] transfer_size : Transfer size                                          */
/* EXAMPLE    :                                                                              */
/*     a[0:100:2]:[1] = b[0:100:2]; // a[] is a dst, b[] is a src                            */
/*********************************************************************************************/
static void _gasnet_nonc_to_nonc_put(const int target_rank, const int dst_dims, const int src_dims,
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				     const _XMP_coarray_t *dst_desc, const void *src, size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  _XMP_pack_coarray(archive + dst_info_size, src, src_dims, src_info);
  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = _XMP_STRIDE_REG;

  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target_rank, _XMP_GASNET_UNPACK, archive, transfer_size,
    			    HIWORD(dst_desc->addr[target_rank]), LOWORD(dst_desc->addr[target_rank]), dst_dims,
    			    _xmp_gasnet_stride_wait_size, _XMP_UNPACK);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put_bulk(target_rank, _xmp_gasnet_buf[target_rank], archive, transfer_size);
    gasnet_AMRequestShort5(target_rank, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst_desc->addr[target_rank]),
                           LOWORD(dst_desc->addr[target_rank]), dst_dims, _xmp_gasnet_stride_wait_size, _XMP_UNPACK);
  }
  else{
    _stride_size_error(transfer_size);
  }

  GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] == _XMP_STRIDE_DONE);
}

/******************************************************************************/
/* DESCRIPTION : Execute multiple put operation for scalar                    */
/* ARGUMENT    : [IN] target_rank : Target rank                               */
/*               [IN] dst_dims    : Number of dimensions of destination array */
/*               [IN] *dst_info   : Information of destination array          */
/*               [OUT] *dst_desc  : Descriptor of destination coarray         */
/*               [IN] *src        : Pointer of source coarray                 */
/*               [IN] elmt_size   : Element size                              */
/* EXAMPLE    :                                                               */
/*     a[0:100]:[1] = b[0]; // a[] is a dst, b[] is a src                     */
/******************************************************************************/
static void _gasnet_scalar_mput(const int target_rank, const int dst_dims,
				const _XMP_array_section_t *dst_info, 
				const _XMP_coarray_t *dst_desc, const void *src, 
				const size_t elmt_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  size_t transfer_size = elmt_size + dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, src, elmt_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = _XMP_STRIDE_REG;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target_rank, _XMP_GASNET_UNPACK, archive, transfer_size,
                            HIWORD(dst_desc->addr[target_rank]), LOWORD(dst_desc->addr[target_rank]), 
			    dst_dims, _xmp_gasnet_stride_wait_size, _XMP_SCALAR_MCOPY);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put(target_rank, _xmp_gasnet_buf[target_rank], archive, transfer_size);
    gasnet_AMRequestShort5(target_rank, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst_desc->addr[target_rank]),
                           LOWORD(dst_desc->addr[target_rank]), dst_dims, _xmp_gasnet_stride_wait_size, _XMP_SCALAR_MCOPY);
  }
  else{
    _stride_size_error(transfer_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

/***************************************************************************************/
/* DESCRIPTION : Execute put operation                                                 */
/* ARGUMENT    : [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE) */
/*               [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)      */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] *dst_info      : Information of destination array                */ 
/*               [IN] *src_info      : Information of source array                     */
/*               [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/* EXAMPLE    :                                                                        */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src                          */
/***************************************************************************************/
void _XMP_gasnet_put(const int dst_continuous, const int src_continuous, const int target_rank, 
		     const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		     const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		     const void *src, const size_t dst_elmts, const size_t src_elmts)
{
  if(dst_elmts == src_elmts){
    size_t transfer_size = dst_desc->elmt_size*dst_elmts;
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_c_put(target_rank, dst_offset, src_offset, dst_desc, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
      size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
      _gasnet_nonc_to_c_put(target_rank, dst_offset, src_dims, src_info, dst_desc, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_nonc_put(target_rank, src_offset, dst_dims, dst_info, dst_desc, src, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
      _gasnet_nonc_to_nonc_put(target_rank, dst_dims, src_dims, dst_info, src_info, 
			       dst_desc, src, transfer_size);
    }
  }
  else{
    if(src_elmts == 1){
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_scalar_mput(target_rank, dst_dims, dst_info, dst_desc, (char *)src+src_offset, dst_desc->elmt_size);
    }
    else{
      _XMP_fatal("Unkown shape of coarray");
    }
  }
}

/*************************************************************************************/
/* DESCRIPTION : Execute get operation (from continuous region to continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                    */
/*               [IN] dst_offset    : Offset size of destination array               */
/*               [IN] src_offset    : Offset size of source array                    */
/*               [IN] *src_desc     : Descriptor of source coarray                   */
/*               [OUT] *dst         : Pointer of destination array                   */
/*               [IN] transfer_size : Transfer size                                  */
/* EXAMPLE    :                                                                      */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src                        */
/*************************************************************************************/
static void _gasnet_c_to_c_get(const int target_rank, const size_t dst_offset, const size_t src_offset, 
			       const void *dst, const _XMP_coarray_t *src_desc, const size_t transfer_size)
{
  gasnet_get_bulk(((char *)dst)+dst_offset, target_rank, ((char *)src_desc->addr[target_rank])+src_offset,
		  transfer_size);

}

/*****************************************************************************************/
/* DESCRIPTION : Execute get operation (from NON-continuous region to continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                        */
/*               [IN] src_offset    : Offset size of source array                        */
/*               [IN] dst_dims      : Number of dimensions of destination array          */
/*               [IN] *dst_info     : Information of destination array                   */
/*               [IN] *dst          : Pointer of destination array                       */
/*               [IN] *src_desc     : Descriptor of source coarray                       */
/*               [IN] transfer_size : Transfer size                                      */
/* EXAMPLE    :                                                                          */
/*     a[0:100] = b[0:100:2]:[1]; // a[] is a dst, b[] is a src                          */
/*****************************************************************************************/
static void _gasnet_c_to_nonc_get(const int target_rank, const size_t src_offset, const int dst_dims,
				  const _XMP_array_section_t *dst_info, const void *dst,
				  const _XMP_coarray_t *src_desc, const size_t transfer_size)
{
  if(transfer_size < _xmp_gasnet_stride_size){
    char* src_addr = (char *)_xmp_gasnet_buf[_XMP_world_rank];
    gasnet_get_bulk(src_addr, target_rank, ((char *)src_desc->addr[target_rank])+src_offset, (size_t)transfer_size);
    _XMP_unpack_coarray(((char *)dst), dst_dims, src_addr, dst_info, _XMP_UNPACK);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

/*****************************************************************************/
/* DESCRIPTION : Execute pack operation                                      */
/* ARGUMENT    : [IN] t               : Token for Active Messages            */
/*               [IN] *array_info     : Information of array                 */
/*               [IN] am_request_size : Request size for Active Messages     */
/*               [IN] src_addr_hi     : Address of source (High 32 bits)     */
/*               [IN] src_addr_lo     : Address of source (Low  32 bits)     */
/*               [IN] src_dims        : Number of dimensions of source array */
/*               [IN] transfer_size   : Transfer size                        */
/*               [IN] dst_addr_hi     : Address of source (High 32 bits)     */
/*               [IN] dst_addr_lo     : Address of source (Low  32 bits)     */
/* Note       : This function is called by Active Messages, and defined in   */
/*              table of xmp_onesided_gasnet.c                               */
/*****************************************************************************/
void _xmp_gasnet_pack(gasnet_token_t t, const char* array_info, const size_t am_request_size, 
		      const int src_addr_hi, const int src_addr_lo, const int src_dims, 
		      const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)array_info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _XMP_pack_coarray(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyMedium2(t, _XMP_GASNET_UNPACK_GET_REPLY, archive, tansfer_size,
      			dst_addr_hi, dst_addr_lo);
}

/**********************************************************************************/
/* DESCRIPTION : Execute pack and get operations                                  */
/* ARGUMENT    : [IN] t               : Token for Active Messages                 */
/*               [IN] *array_info     : Information of array                      */
/*               [IN] am_request_size : Request size for Active Messages          */
/*               [IN] src_addr_hi     : Address of source (High 32 bits)          */
/*               [IN] src_addr_lo     : Address of source (Low  32 bits)          */
/*               [IN] src_dims        : Number of dimensions of source array      */
/*               [IN] dst_dims        : Number of dimensions of destination array */
/*               [IN] transfer_size   : Transfer size                             */
/*               [IN] dst_addr_hi     : Address of source (High 32 bits)          */
/*               [IN] dst_addr_lo     : Address of source (Low  32 bits)          */
/* Note       : This function is called by Active Messages, and defined in        */
/*              table of xmp_onesided_gasnet.c                                    */
/**********************************************************************************/
void _xmp_gasnet_pack_get(gasnet_token_t t, const char* array_info, const size_t am_request_size,
			  const int src_addr_hi, const int src_addr_lo, const int src_dims, const int dst_dims,
			  const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  size_t src_size = sizeof(_XMP_array_section_t) * src_dims;
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *src_info = malloc(src_size);
  memcpy(src_info, array_info, src_size);
  char archive[tansfer_size + dst_size];
  memcpy(archive, array_info + src_size, dst_size);
  _XMP_pack_coarray(archive+dst_size, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  free(src_info);
  gasnet_AMReplyMedium3(t, _XMP_GASNET_UNPACK_GET_REPLY_NONC, archive, tansfer_size + dst_size,
                        dst_addr_hi, dst_addr_lo, dst_dims);
}

/********************************************************************************/
/* DESCRIPTION : Execute unpack operations for Non-continuous GET               */
/* ARGUMENT    : [IN] t             : Token for Active Messages                 */
/*               [IN] *archives     : Recieved message                          */
/*               [IN] transfer_size : Transfer size                             */
/*               [IN] dst_addr_hi   : Address of source (High 32 bits)          */
/*               [IN] dst_addr_lo   : Address of source (Low  32 bits)          */
/*               [IN] dst_dims      : Number of dimensions of destination array */
/* Note       : This function is called by Active Messages, and defined in      */
/*              table of xmp_onesided_gasnet.c                                  */
/********************************************************************************/
void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t t, char *archive, size_t transfer_size,
				       const int dst_addr_hi, const int dst_addr_lo,
				       const int dst_dims)
{
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst_info = malloc(dst_size);
  memcpy(dst_info, archive, dst_size);

  _XMP_unpack_coarray((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), dst_dims, archive+dst_size, dst_info, _XMP_UNPACK);
  done_get_flag = _XMP_N_INT_TRUE;
}


/********************************************************************************/
/* DESCRIPTION : Execute unpack operations                                      */
/* ARGUMENT    : [IN] t             : Token for Active Messages                 */
/*               [IN] *archives     : Recieved message                          */
/*               [IN] transfer_size : Transfer size                             */
/*               [IN] dst_addr_hi   : Address of source (High 32 bits)          */
/*               [IN] dst_addr_lo   : Address of source (Low  32 bits)          */
/* Note       : This function is called by Active Messages, and defined in      */
/*              table of xmp_onesided_gasnet.c                                  */
/********************************************************************************/
void _xmp_gasnet_unpack_get_reply(gasnet_token_t t, char *archive, size_t transfer_size, 
				  const int dst_addr_hi, const int dst_addr_lo)
{
  memcpy((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), archive, transfer_size);
  done_get_flag = _XMP_N_INT_TRUE;
}

/**
   Set done flag for get operation
 */
void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t t)
{
  done_get_flag = _XMP_N_INT_TRUE;
}

/**********************************************************************************/
/* DESCRIPTION : Execute pack operations which uses buffer                        */
/* ARGUMENT    : [IN] t               : Token for Active Messages                 */
/*               [IN] *array_info     : Information of array                      */
/*               [IN] am_request_size : Request size for Active Messages          */
/*               [IN] src_addr_hi     : Address of source (High 32 bits)          */
/*               [IN] src_addr_lo     : Address of source (Low  32 bits)          */
/*               [IN] target_rank     : Target rank                               */
/* Note       : This function is called by Active Messages, and defined in        */
/*              table of xmp_onesided_gasnet.c                                    */
/**********************************************************************************/
void _xmp_gasnet_pack_using_buf(gasnet_token_t t, const char* array_info, const size_t am_request_size,
				const int src_addr_hi, const int src_addr_lo, const int src_dims,
				const int target_rank)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)array_info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _XMP_pack_coarray(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyShort0(t, _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF);
}

/*****************************************************************************************/
/* DESCRIPTION : Execute get operation (from NON-continuous region to continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                        */
/*               [IN] dst_offset    : Offset size of destination array                   */
/*               [IN] src_dims      : Number of dimensions of source array               */
/*               [IN] *src_info     : Information of source array                        */
/*               [IN] *dst          : Pointer of destination coarray                     */
/*               [IN] *src_desc     : Descriptor of source array                         */
/*               [IN] transfer_size : Transfer size                                      */
/* EXAMPLE    :                                                                          */
/*     a[0:100] = b[0:100:2]:[1]; // a[] is a dst, b[] is a src                          */
/*****************************************************************************************/
static void _gasnet_nonc_to_c_get(const int target_rank, const size_t dst_offset, const int src_dims, 
				  const _XMP_array_section_t *src_info, 
				  const void *dst, const _XMP_coarray_t *src_desc, const size_t transfer_size)
{
  size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
  char archive[am_request_size];  // Note: Info. of transfer_size may have better in "archive".
  memcpy(archive, src_info, am_request_size);

  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // fix me
    gasnet_AMRequestMedium6(target_rank, _XMP_GASNET_PACK, archive, am_request_size,
			    HIWORD(src_desc->addr[target_rank]), LOWORD(src_desc->addr[target_rank]), src_dims,
    			    (size_t)transfer_size, HIWORD((char *)dst+dst_offset), LOWORD((char *)dst+dst_offset));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_AMRequestMedium4(target_rank, _XMP_GASNET_PACK_USING_BUF, archive, am_request_size,
                            HIWORD(src_desc->addr[target_rank]), LOWORD(src_desc->addr[target_rank]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target_rank, _xmp_gasnet_buf[target_rank], transfer_size);
    memcpy(((char *)dst)+dst_offset, _xmp_gasnet_buf[_XMP_world_rank], transfer_size);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

/*********************************************************************************************/
/* DESCRIPTION : Execute get operation (from NON-continuous region to NON-continuous region) */
/* ARGUMENT    : [IN] target_rank   : Target rank                                            */
/*               [IN] dst_dims      : Number of dimensions of destination array              */
/*               [IN] src_dims      : Number of dimensions of source array                   */
/*               [IN] *dst_info     : Information of destination array                       */
/*               [IN] *src_info     : Information of source array                            */
/*               [IN] *dst          : Pointer of destination coarray                         */
/*               [IN] *src_desc     : Descriptor of source array                             */
/*               [IN] transfer_size : Transfer size                                          */
/* EXAMPLE    :                                                                              */
/*     a[0:100:2] = b[0:100:2]:[1]; // a[] is a dst, b[] is a src                            */
/*********************************************************************************************/
static void _gasnet_nonc_to_nonc_get(const int target_rank, const int dst_dims, const int src_dims, 
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, 
				     const void *dst, const _XMP_coarray_t *src_desc, const size_t transfer_size)
{
  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // fix me
    size_t am_request_src_size = sizeof(_XMP_array_section_t) * src_dims;
    size_t am_request_dst_size = sizeof(_XMP_array_section_t) * dst_dims;
    char *archive = malloc(am_request_src_size + am_request_dst_size);
    memcpy(archive, src_info, am_request_src_size);
    memcpy(archive + am_request_src_size, dst_info, am_request_dst_size);
    gasnet_AMRequestMedium7(target_rank, _XMP_GASNET_PACK_GET_HANDLER, archive, 
			    am_request_src_size+am_request_dst_size,
                            HIWORD(src_desc->addr[target_rank]), LOWORD(src_desc->addr[target_rank]), src_dims, dst_dims,
                            (size_t)transfer_size, HIWORD((char *)dst), LOWORD((char *)dst));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    free(archive);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
    char *archive = malloc(am_request_size);
    memcpy(archive, src_info, am_request_size);
    gasnet_AMRequestMedium4(target_rank, _XMP_GASNET_PACK_USING_BUF, archive, am_request_size,
                            HIWORD(src_desc->addr[target_rank]), LOWORD(src_desc->addr[target_rank]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target_rank, _xmp_gasnet_buf[target_rank], 
		    transfer_size);
    _XMP_unpack_coarray((char *)dst, dst_dims, _xmp_gasnet_buf[_XMP_world_rank], dst_info, _XMP_UNPACK);
    free(archive);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

/******************************************************************************/
/* DESCRIPTION : Execute multiple put operation for scalar                    */
/* ARGUMENT    : [IN] target_rank : Target rank                               */
/*               [IN] src_offset  : Offset size of source array               */
/*               [IN] dst_dims    : Number of dimensions of destination array */
/*               [IN] *dst_info   : Information of destination array          */
/*               [IN] *dst        : Pointer of destination array              */
/*               [IN] *src_desc   : Descriptor of source coarray              */
/*               [IN] elmt_size   : Element size                              */
/* EXAMPLE    :                                                               */
/*     a[0:100] = b[0]:[1]; // a[] is a dst, b[] is a src                     */
/******************************************************************************/
static void _gasnet_scalar_mget(const int target_rank, const size_t src_offset, const int dst_dims,
				const _XMP_array_section_t *dst_info, const void *dst,
				const _XMP_coarray_t *src_desc, const size_t elmt_size)
{
  char* src_addr = (char *)_xmp_gasnet_buf[_XMP_world_rank];
  gasnet_get_bulk(src_addr, target_rank, ((char *)src_desc->addr[target_rank])+src_offset, elmt_size);
  _XMP_unpack_coarray(((char *)dst), dst_dims, src_addr, dst_info, _XMP_SCALAR_MCOPY);
}

/***************************************************************************************/
/* DESCRIPTION : Execute get operation                                                 */
/* ARGUMENT    : [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)      */
/*               [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE) */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] *src_info      : Information of source array                     */
/*               [IN] *dst_info      : Information of destination array                */
/*               [OUT] *src_desc     : Descriptor of source coarray                    */
/*               [IN] *dst           : Pointer of destination array                    */
/*               [IN] src_elmts      : Number of elements of source array              */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/* EXAMPLE    :                                                                        */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src                          */
/***************************************************************************************/
void _XMP_gasnet_get(const int src_continuous, const int dst_continuous, const int target_rank, const int src_dims, 
		     const int dst_dims, const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		     const _XMP_coarray_t *src_desc, const void *dst, const size_t src_elmts, const size_t dst_elmts)
{
  if(src_elmts == dst_elmts){
    size_t transfer_size = src_desc->elmt_size*src_elmts;
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_c_get(target_rank, dst_offset, src_offset, dst, src_desc, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
      size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
      _gasnet_nonc_to_c_get(target_rank, dst_offset, src_dims, src_info, dst, src_desc, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_c_to_nonc_get(target_rank, src_offset, dst_dims, dst_info, dst, src_desc, transfer_size);
    }
    else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
      _gasnet_nonc_to_nonc_get(target_rank, dst_dims, src_dims, dst_info, src_info, dst, src_desc, transfer_size);
    }
  }
  else{
    if(src_elmts == 1){
      size_t src_offset = _XMP_get_offset(src_info, src_dims);
      _gasnet_scalar_mget(target_rank, src_offset, dst_dims, dst_info, dst, src_desc, src_desc->elmt_size);
    }
    else{
      _XMP_fatal("Unkown shape of coarray");
    }
  }
}

/**********************************************************************/
/* DESCRIPTION : Execute multiple put operation without preprocessing */
/* ARGUMENT    : [IN] target_rank : Target rank                       */
/*               [OUT] *dst_desc  : Descriptor of destination coarray */
/*               [IN] *src        : Pointer of source coarray         */
/*               [IN] dst_offset  : Offset size of destination array  */
/*               [IN] dst_elmts   : Number of elements of destination */
/*               [IN] elmt_size   : Element size                      */
/* NOTE       : Both dst and src are continuous coarrays              */
/*              target_rank != __XMP_world_rank.                      */
/* EXAMPLE    :                                                       */
/*     a[0:100]:[1] = b[0]; // a[] is a dst, b[] is a src             */
/**********************************************************************/
static void _gasnet_scalar_shortcut_mput(const int target_rank, _XMP_coarray_t *dst_desc, const void *src, 
					 const size_t dst_offset, const size_t dst_elmts,
					 const size_t elmt_size)
{
  _XMP_array_section_t dst_info[1];
  dst_info[0].start    = dst_offset/elmt_size;
  dst_info[0].length   = dst_elmts;
  dst_info[0].stride   = 1;
  dst_info[0].distance = elmt_size;

  size_t dst_info_size = sizeof(_XMP_array_section_t);
  size_t trans_size = elmt_size + dst_info_size;
  char archive[trans_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, src, elmt_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = _XMP_STRIDE_REG;

  if(trans_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium5(target_rank, _XMP_GASNET_UNPACK, archive, trans_size,
                            HIWORD(dst_desc->addr[target_rank]), LOWORD(dst_desc->addr[target_rank]), 1,
                            _xmp_gasnet_stride_wait_size, _XMP_SCALAR_MCOPY);
  }
  else if(trans_size < _xmp_gasnet_stride_size){
    gasnet_put(target_rank, _xmp_gasnet_buf[target_rank], archive, trans_size);
    gasnet_AMRequestShort5(target_rank, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst_desc->addr[target_rank]),
                           LOWORD(dst_desc->addr[target_rank]), 1, _xmp_gasnet_stride_wait_size,
			   _XMP_SCALAR_MCOPY);
  }
  else{
    _stride_size_error(trans_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

/****************************************************************************/
/* DESCRIPTION : Execute put operation without preprocessing                */
/* ARGUMENT    : [IN] target_rank : Target rank                             */
/*               [OUT] *dst_desc  : Descriptor of destination coarray       */
/*               [IN] *src        : Pointer of source array                 */
/*               [IN] dst_offset  : Offset size of destination array        */
/*               [IN] dst_elmts   : Number of elements of destination array */
/*               [IN] src_elmts   : Number of elements of source array      */
/*               [IN] elmt_size   : Element size                            */
/* NOTE       : Both dst and src are continuous coarrays                    */
/*              target_rank != __XMP_world_rank.                            */
/* EXAMPLE    :                                                             */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src               */
/****************************************************************************/
void _XMP_gasnet_shortcut_put(const int target_rank, _XMP_coarray_t *dst_desc, void *src, 
			      const size_t dst_offset, const size_t dst_elmts, 
			      const size_t src_elmts, const size_t elmt_size)
{
  if(dst_elmts == src_elmts){
    gasnet_put_nbi_bulk(target_rank, dst_desc->addr[target_rank]+dst_offset, src, src_elmts*elmt_size);
  }
  else if(src_elmts == 1){
    _gasnet_scalar_shortcut_mput(target_rank, dst_desc, src, dst_offset, dst_elmts, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/****************************************************************************/
/* DESCRIPTION : Execute multiple get operation without preprocessing       */
/* ARGUMENT    : [IN] target_rank : Target rank                             */
/*               [OUT] *dst_desc  : Descriptor of destination coarray       */
/*               [IN] *src        : Pointer of source array                 */
/*               [IN] dst_offset  : Offset size of destination array        */
/*               [IN] dst_elmts   : Number of elements of destination array */
/*               [IN] elmt_size   : Element size                            */
/* NOTE       : Both dst and src are continuous coarrays                    */
/*              target_rank != __XMP_world_rank.                            */
/* EXAMPLE    :                                                             */
/*     a[0:100] = b[0]:[1]; // a[] is a dst, b[] is a src                   */
/****************************************************************************/
static void _gasnet_scalar_shortcut_mget(const int target_rank, _XMP_coarray_t *dst_desc, void *src,
                                         const size_t dst_offset, const size_t dst_elmts,
                                         const size_t elmt_size)
{
  char *dst_addr = dst_desc->addr[_XMP_world_rank]+dst_offset;
  gasnet_get_bulk(dst_addr, target_rank, src, elmt_size);
  for(int i=1;i<dst_elmts;i++)
    memcpy(dst_addr+i*elmt_size, dst_addr, elmt_size);
}

/****************************************************************************/
/* DESCRIPTION : Execute get operation without preprocessing                */
/* ARGUMENT    : [IN] target_rank : Target rank                             */
/*               [OUT] *dst_desc  : Descriptor of destination coarray       */
/*               [IN] *src        : Pointer of source array                 */
/*               [IN] dst_offset  : Offset size of destination array        */
/*               [IN] dst_elmts   : Number of elements of destination array */
/*               [IN] src_elmts   : Number of elements of source array      */
/*               [IN] elmt_size   : Element size                            */
/* NOTE       : Both dst and src are continuous coarrays                    */
/*              target_rank != __XMP_world_rank.                            */
/* EXAMPLE    :                                                             */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src               */
/****************************************************************************/
void _XMP_gasnet_shortcut_get(const int target_rank, _XMP_coarray_t *dst_desc, void *src,
			      const size_t dst_offset, const size_t dst_elmts, const size_t src_elmts,
			      const size_t elmt_size)
{
  if(dst_elmts == src_elmts){
    gasnet_get_bulk(dst_desc->addr[_XMP_world_rank]+dst_offset, target_rank, src, src_elmts*elmt_size);
  }
  else if(src_elmts == 1){
    _gasnet_scalar_shortcut_mget(target_rank, dst_desc, src, dst_offset, dst_elmts, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/**
 * Build table and Initialize for sync images
 */
void _XMP_gasnet_build_sync_images_table()
{
  _sync_images_table = malloc(sizeof(unsigned int) * _XMP_world_size);

  gasnet_hsl_lock(&_hsl);
  for(int i=0;i<_XMP_world_size;i++)
    _sync_images_table[i] = 0;
  gasnet_hsl_unlock(&_hsl);
}

static void _add_notify(const int rank)
{
  gasnet_hsl_lock(&_hsl);
  _sync_images_table[rank]++;
  gasnet_hsl_unlock(&_hsl);
}

void _xmp_gasnet_add_notify(gasnet_token_t token, const int rank)
{
  _add_notify(rank);
}

/**
   Notify to nodes
   *
   * @param[in]  num        number of nodes
   * @param[in]  *rank_set  rank set
   */
static void _notify_sync_images(const int num, int *rank_set)
{
  for(int i=0;i<num;i++){
    if(rank_set[i] == _XMP_world_rank){
      _add_notify(_XMP_world_rank);
    }
    else{
      gasnet_AMRequestShort1(rank_set[i], _XMP_GASNET_ADD_NOTIFY, _XMP_world_rank);
    }
  }
}

/**
   Check to recieve all request from all node
   *
   * @param[in] num                    number of nodes
   * @param[in] *rank_set              rank set
   */
static _Bool _check_sync_images_table(const int num, int *rank_set)
{
  int checked = 0;

  for(int i=0;i<num;i++)
    if(_sync_images_table[rank_set[i]] > 0) checked++;

  if(checked == num) return true;
  else               return false;
}

/**
   Wait until recieving all request from all node
   *
   * @param[in] num                    number of nodes
   * @param[in] *rank_set              rank set
   */
static void _wait_sync_images(const int num, int *rank_set)
{
  while(1){
    if(_check_sync_images_table(num, rank_set)) break;
    gasnet_AMPoll();
  }
}

/**
   Execute sync images
   *
   * @param[in]  num         number of nodes
   * @param[in]  *image_set  image set
   * @param[out] status      status
*/
void _XMP_gasnet_sync_images(const int num, int image_set[num], int *status)
{
  _XMP_gasnet_sync_memory();
  
  if(num == 0){
    return;
  }
  else if(num < 0){
    fprintf(stderr, "Invalid value is used in xmp_sync_memory. The first argument is %d\n", num);
    _XMP_fatal_nomsg();
  }
  
  int rank_set[num];

  for(int i=0;i<num;i++)
    rank_set[i] = image_set[i] - 1;

  _notify_sync_images(num, rank_set);
  _wait_sync_images(num, rank_set);

  // Update table for post-processing
  gasnet_hsl_lock(&_hsl);
  for(int i=0;i<num;i++)
    _sync_images_table[rank_set[i]]--;
  gasnet_hsl_unlock(&_hsl);
}
