//#define DEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"

struct _shift_queue_t{
  unsigned int max_size;   /**< Max size of queue */
  unsigned int      num;   /**< How many shifts are in this queue */
  size_t        *shifts;   /**< shifts array */
  size_t    total_shift;   /**< all amount of shifts */
};
static struct _shift_queue_t _shift_queue; /** Queue which saves shift information */
static struct _shift_queue_t _shift_queue_acc;
static bool _is_coarray_win_flushed = true;
static bool _is_coarray_win_acc_flushed = true;
static bool _is_distarray_win_flushed = true;
static bool _is_distarray_win_acc_flushed = true;

static void _mpi_continuous_put(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				const size_t dst_offset, const size_t src_offset, const size_t transfer_size, const int is_dst_on_acc);
static void _mpi_continuous_get(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				const size_t dst_offset, const size_t src_offset, const size_t transfer_size, const int is_dst_on_acc);


static void set_flushed_flag(bool is_normal, bool is_acc, bool flag)
{
  if(is_normal){
    if(is_acc){
      _is_coarray_win_acc_flushed = flag;
    }else{
      _is_coarray_win_flushed = flag;
    }
  }else{
    if(is_acc){
      _is_distarray_win_acc_flushed = flag;
    }else{
      _is_distarray_win_flushed = flag;
    }
  }
}

static MPI_Win get_window(const _XMP_coarray_t *desc, bool is_acc)
{
  MPI_Win win = desc->win;
#ifdef _XMP_XACC
  if(is_acc){
    win = desc->win_acc;
  }
#endif
  if(win == MPI_WIN_NULL){ //when the coarray is normal
    win = _xmp_mpi_onesided_win;
#ifdef _XMP_XACC
    if(is_acc){
      win = _xmp_mpi_onesided_win_acc;
    }
#endif
    set_flushed_flag(true, is_acc, false);
  }else{
    set_flushed_flag(false, is_acc, false);
  }
  return win;
}

MPI_Win _XMP_mpi_coarray_get_window(const _XMP_coarray_t *desc, bool is_acc)
{
  return get_window(desc, is_acc);
}

static char *get_remote_addr(const _XMP_coarray_t *desc, const int target_rank, const bool is_acc)
{
#ifdef _XMP_XACC
  if(is_acc){
    return desc->addr_dev[target_rank];
  }
#endif
  return desc->addr[target_rank];
}

static char *get_local_addr(const _XMP_coarray_t *desc, const bool is_acc)
{
#ifdef _XMP_XACC
  if(is_acc){
    return desc->real_addr_dev;
  }
#endif
  return desc->real_addr;
}

/**
   Set initial value to the shift queue
 */
void _XMP_mpi_build_shift_queue(bool is_acc)
{
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  
  shift_queue->max_size = _XMP_MPI_ONESIDED_COARRAY_SHIFT_QUEUE_INITIAL_SIZE;
  shift_queue->num      = 0;
  shift_queue->shifts   = malloc(sizeof(size_t*) * shift_queue->max_size);
  shift_queue->total_shift = 0;
}

/**
   Destroy shift queue
 */
void _XMP_mpi_destroy_shift_queue(bool is_acc)
{
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;

  free(shift_queue->shifts);
  shift_queue->shifts = NULL;
}

/**
   Create new shift queue
 */
static void _rebuild_shift_queue(struct _shift_queue_t *shift_queue)
{
  shift_queue->max_size *= _XMP_MPI_ONESIDED_COARRAY_SHIFT_QUEUE_INCREMENT_RAITO;
  size_t *tmp;
  size_t next_size = shift_queue->max_size * sizeof(size_t*);
  if((tmp = realloc(shift_queue->shifts, next_size)) == NULL)
    _XMP_fatal("cannot allocate memory");
  else
    shift_queue->shifts = tmp;
}

/**
   Push shift information to the shift queue
 */
static void _push_shift_queue(struct _shift_queue_t *shift_queue, size_t s)
{
  if(shift_queue->num >= shift_queue->max_size)
    _rebuild_shift_queue(shift_queue);

  shift_queue->shifts[shift_queue->num++] = s;
  shift_queue->total_shift += s;
}

/**
   Pop shift information from the shift queue
 */
static size_t _pop_shift_queue(struct _shift_queue_t *shift_queue)
{
  if(shift_queue->num == 0)  return 0;

  shift_queue->num--;
  size_t shift = shift_queue->shifts[shift_queue->num];
  shift_queue->total_shift -= shift;
  return shift;
}

/**
   Deallocate memory region when calling _XMP_coarray_lastly_deallocate()
*/
void _XMP_mpi_onesided_coarray_lastly_deallocate(bool is_acc){
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  _pop_shift_queue(shift_queue);
}


/**********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                 */
/* ARGUMENT    : [OUT] *coarray_desc : Descriptor of new coarray      */
/*               [OUT] **addr        : Double pointer of new coarray  */
/*               [IN] coarray_size   : Coarray size                   */
/**********************************************************************/
void _XMP_mpi_coarray_malloc_do(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size, bool is_acc)
{
  char **each_addr;  // gap_size on each node
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  size_t shift;

  each_addr = (char**)_XMP_alloc(sizeof(char *) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++){
    each_addr[i] = (char *)(shift_queue->total_shift);
  }
  char *real_addr = _xmp_mpi_onesided_buf;
#ifdef _XMP_XACC
  if(is_acc){
    real_addr = _xmp_mpi_onesided_buf_acc;
  }
#endif
  real_addr += shift_queue->total_shift;

  XACC_DEBUG("malloc_do: addr=%p, shift=%zd, is_acc=%d", real_addr, shift_queue->total_shift, is_acc);
  
  if(coarray_size % _XMP_MPI_ALIGNMENT == 0)
    shift = coarray_size;
  else{
    shift = ((coarray_size / _XMP_MPI_ALIGNMENT) + 1) * _XMP_MPI_ALIGNMENT;
  }
  
  _push_shift_queue(shift_queue, shift);

  size_t total_shift = shift_queue->total_shift;

  if(total_shift > _xmp_mpi_onesided_heap_size){
    fprintf(stderr, "_xmp_mpi_onesided_heap_size=%zd\n", _xmp_mpi_onesided_heap_size);
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Cannot allocate coarray. Heap memory size of corray is too small.\n");
      fprintf(stderr, "        Please set the environmental variable \"XMP_ONESIDED_HEAP_SIZE\".\n");
      fprintf(stderr, "        e.g.) export XMP_ONESIDED_HEAP_SIZE=%zuM (or more).\n",
	      (total_shift/1024/1024)+1);
    }
    _XMP_fatal_nomsg();
  }

  if(is_acc){
#ifdef _XMP_XACC
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = real_addr;
    coarray_desc->win_acc = MPI_WIN_NULL;
#endif
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = real_addr;
    coarray_desc->win = MPI_WIN_NULL;
  }
  *addr = real_addr;
}

/************************************************************************/
/* DESCRIPTION : Call put operation without preprocessing               */
/* ARGUMENT    : [IN] target_rank  : Target rank                        */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] elmt_size    : Element size                       */
/*               [IN] is_acc       : Whether src and dst are acc or not */
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_shortcut_put(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			   const size_t dst_offset, const size_t src_offset,
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_acc)
{
  if(dst_elmts == src_elmts){
    size_t transfer_size = elmt_size * dst_elmts;
    char *src = get_local_addr(src_desc, is_acc);
    _mpi_continuous_put(target_rank, dst_desc, src, dst_offset, src_offset, transfer_size, is_acc);
  }else if(src_elmts == 1){
    _XMP_fatal("unimplemented");
    //_gasnet_scalar_shortcut_mput(target_rank, dst_desc, src, dst_offset, dst_elmts, elmt_size);
  }else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/************************************************************************/
/* DESCRIPTION : Execute get operation without preprocessing            */
/* ARGUMENT    : [IN] target_rank  : Target rank                        */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] elmt_size    : Element size                       */
/*               [IN] is_acc       : Whether src and dst are acc or not */
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_shortcut_get(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			   const size_t dst_offset, const size_t src_offset,
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_acc)
{
  if(dst_elmts == src_elmts){
    size_t transfer_size = elmt_size * dst_elmts;
    char *src = get_local_addr(dst_desc, is_acc);
    _mpi_continuous_get(target_rank, dst_desc, src,
			dst_offset, src_offset, transfer_size, is_acc);
  }else if(src_elmts == 1){
    _XMP_fatal("unimplemented");
  }else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/**
   Execute sync_memory
 */
void _XMP_mpi_sync_memory()
{
  if(! _is_coarray_win_flushed){
    XACC_DEBUG("sync_memory(normal, host)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win);
    MPI_Win_flush_all(_xmp_mpi_onesided_win);

    _is_coarray_win_flushed = true;
  }

  if(! _is_distarray_win_flushed){
    XACC_DEBUG("sync_memory(distarray, host)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win);
    MPI_Win_flush_all(_xmp_mpi_distarray_win);

    _is_distarray_win_flushed = true;
  }

#ifdef _XMP_XACC
  if(! _is_coarray_win_acc_flushed){
    XACC_DEBUG("sync_memory(normal, acc)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win_acc);
    MPI_Win_flush_all(_xmp_mpi_onesided_win_acc);

    _is_coarray_win_acc_flushed = true;
  }

  if(! _is_distarray_win_acc_flushed){
    XACC_DEBUG("sync_memory(distarray, acc)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win_acc);
    MPI_Win_flush_all(_xmp_mpi_distarray_win_acc);

    _is_distarray_win_acc_flushed = true;
  }
#endif
}

/**
   Execute sync_all
 */
void _XMP_mpi_sync_all()
{
  _XMP_mpi_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}

static void _mpi_continuous_put(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				const size_t dst_offset, const size_t src_offset, const size_t transfer_size, const int is_dst_on_acc)
{
  if(transfer_size == 0) return;
  char *laddr = (char*)src + src_offset;
  char *raddr = get_remote_addr(dst_desc, target_rank, is_dst_on_acc) + dst_offset;
  MPI_Win win = get_window(dst_desc, is_dst_on_acc);

  XACC_DEBUG("continuous_put(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_dst_on_acc);

#if 1
  MPI_Put((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	  (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	  win);
#else
  size_t size_multiple128k = (transfer_size / (128*1024)) * (128*1024);
  size_t size_rest = transfer_size - size_multiple128k;
  if(transfer_size >= (128*1024) && size_rest > 0 && size_rest <= (8*1024)){
    XACC_DEBUG("put(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d) divied! (%d,%d)", laddr, transfer_size, target_rank, raddr, is_dst_on_acc, size128k, size_rest);
    MPI_Put((void*)laddr, size_multiple128k, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, size_multiple128k, MPI_BYTE,
	    is_dst_on_acc? _xmp_mpi_onesided_win_acc : _xmp_mpi_onesided_win);
    MPI_Put((void*)(laddr+size_multiple128k), size_rest, MPI_BYTE, target_rank,
	    (MPI_Aint)(raddr+size_multiple128k), size_rest, MPI_BYTE,
	    is_dst_on_acc? _xmp_mpi_onesided_win_acc : _xmp_mpi_onesided_win);
  }else{
    MPI_Put((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    win);
  }
#endif

}

static void _mpi_continuous_get(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				const size_t dst_offset, const size_t src_offset, const size_t transfer_size, const int is_dst_on_acc)
{
  if(transfer_size == 0) return;
  char *laddr = (char*)src + src_offset;
  char *raddr = get_remote_addr(dst_desc, target_rank, is_dst_on_acc) + dst_offset;
  MPI_Win win = get_window(dst_desc, is_dst_on_acc);

  XACC_DEBUG("continuous_get(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_dst_on_acc);
  MPI_Request req;
  MPI_Rget((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	  (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	  win, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
}

static bool _check_block_stride(const int dims, const _XMP_array_section_t *info, long long *cnt, long long *bl, long long *str)
{
  long long count = 1;
  long long blocklength = 1;
  long long stride = 1;

  for(int i = 0; i < dims; i++){
    long long length = info[i].length;
    long long elmts = info[i].elmts;

    if(info[i].stride != 1){
      if(count == 1 && blocklength == 1){
	count = length;
	blocklength = 1;
	stride = info[i].stride;
      }else{
	return false;
      }
    }else if(blocklength == 1 || length == elmts){
      blocklength *= length;
      stride *= elmts;
    }else if(count == 1){ //continuous -> blockstride
      count = blocklength;
      blocklength = length;
      stride = elmts;
    }else{
      return false;
    }
  }

  //output
  *cnt = count;
  *bl = blocklength;
  *str = stride;
  return true;
}

static void _mpi_non_continuous_put(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				    const size_t dst_offset, const size_t src_offset,
				    const int dst_dims, const int src_dims, 
				    const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				    const size_t dst_elmts, const int is_dst_on_acc)
{
  //check stride
  long long dst_cnt, dst_bl, dst_str;
  long long src_cnt, src_bl, src_str;

  bool is_dst_blockstride = _check_block_stride(dst_dims, dst_info, &dst_cnt, &dst_bl, &dst_str);
  bool is_src_blockstride = _check_block_stride(src_dims, src_info, &src_cnt, &src_bl, &src_str);

  if(is_dst_blockstride && is_src_blockstride && dst_cnt == src_cnt && dst_bl == src_bl && dst_str == src_str){
    char *laddr = (char*)src + src_offset;
    char *raddr = get_remote_addr(dst_desc, target_rank, is_dst_on_acc) + dst_offset;
    MPI_Win win = get_window(dst_desc, is_dst_on_acc);

    XACC_DEBUG("blockstride_put(src_p=%p, (c,bl,s)=(%lld,%lld,%lld), target=%d, dst_p=%p, is_acc=%d)", laddr, src_cnt,src_bl,src_str, target_rank, raddr, is_dst_on_acc);

    MPI_Datatype blockstride_type;
    size_t elmt_size = dst_desc->elmt_size;
    MPI_Type_vector(dst_cnt, dst_bl * elmt_size, dst_str  * elmt_size, MPI_BYTE, &blockstride_type);
    MPI_Type_commit(&blockstride_type);

    int result=
    MPI_Put((void*)laddr, 1, blockstride_type, target_rank,
	    (MPI_Aint)raddr, 1, blockstride_type,
	    win);

    if(result != MPI_SUCCESS){
      _XMP_fatal("put error");
    }
    MPI_Type_free(&blockstride_type);

  }else{
    _XMP_fatal("not implemented non-continuous data");
  }
}

static void _mpi_non_continuous_get(const int target_rank, const _XMP_coarray_t *dst_desc, const void *src,
				    const size_t dst_offset, const size_t src_offset,
				    const int dst_dims, const int src_dims,
				    const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				    const size_t dst_elmts, const int is_dst_on_acc)
{
  //check stride
  long long dst_cnt, dst_bl, dst_str;
  long long src_cnt, src_bl, src_str;

  bool is_dst_blockstride = _check_block_stride(dst_dims, dst_info, &dst_cnt, &dst_bl, &dst_str);
  bool is_src_blockstride = _check_block_stride(src_dims, src_info, &src_cnt, &src_bl, &src_str);

  if(is_dst_blockstride && is_src_blockstride && dst_cnt == src_cnt && dst_bl == src_bl && dst_str == src_str){
    char *laddr = (char*)src + src_offset;
    char *raddr = get_remote_addr(dst_desc, target_rank, is_dst_on_acc) + dst_offset;
    MPI_Win win = get_window(dst_desc, is_dst_on_acc);

    XACC_DEBUG("blockstride_get(src_p=%p, (c,bl,s)=(%lld,%lld,%lld), target=%d, dst_p=%p, is_acc=%d)", laddr, src_cnt,src_bl,src_str, target_rank, raddr, is_dst_on_acc);

    MPI_Datatype blockstride_type;
    size_t elmt_size = dst_desc->elmt_size;
    MPI_Type_vector(dst_cnt, dst_bl * elmt_size, dst_str  * elmt_size, MPI_BYTE, &blockstride_type);
    MPI_Type_commit(&blockstride_type);

    MPI_Request req;
    int result=
      MPI_Rget((void*)laddr, 1, blockstride_type, target_rank,
	      (MPI_Aint)raddr, 1, blockstride_type,
	       win, &req);
    
    if(result != MPI_SUCCESS){
      _XMP_fatal("put error");
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Type_free(&blockstride_type);


  }else{
    _XMP_fatal("not implemented non-continuous data");
  }
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
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/***************************************************************************************/
void _XMP_mpi_put(const int dst_continuous, const int src_continuous, const int target_rank, 
		  const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		  const _XMP_coarray_t *src_desc, void *src, const int dst_elmts, const int src_elmts,
		  const int is_dst_on_acc, const int is_src_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);

  size_t transfer_size = dst_desc->elmt_size * dst_elmts;
  //_check_transfer_size(transfer_size);

  if(dst_elmts == src_elmts){
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      _mpi_continuous_put(target_rank, dst_desc, src, dst_offset, src_offset, transfer_size, is_dst_on_acc);
    }
    else{
      _mpi_non_continuous_put(target_rank, dst_desc, src,
			      dst_offset, src_offset, dst_dims, src_dims, dst_info, src_info, dst_elmts, is_dst_on_acc);
    }
  }
  else{
    if(src_elmts == 1){
      _XMP_fatal("_XMP_mpi_put: scalar_mput is unimplemented");
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
}

/***************************************************************************************/
/* DESCRIPTION : Execute get operation                                                 */
/* ARGUMENT    : [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE) */
/*               [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)      */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] *dst_info      : Information of destination array                */
/*               [IN] *src_info      : Information of source array                     */
/*               [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/***************************************************************************************/
void _XMP_mpi_get(const int dst_continuous, const int src_continuous, const int target_rank,
		  const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info,
		  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc,
		  const _XMP_coarray_t *src_desc, void *src, const int dst_elmts, const int src_elmts,
		  const int is_dst_on_acc, const int is_src_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);

  size_t transfer_size = dst_desc->elmt_size * dst_elmts;
  //_check_transfer_size(transfer_size);

  if(dst_elmts == src_elmts){
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      _mpi_continuous_get(target_rank, dst_desc, src, dst_offset, src_offset, transfer_size, is_dst_on_acc);
    }
    else{
      _mpi_non_continuous_get(target_rank, dst_desc, src,
			      dst_offset, src_offset, dst_dims, src_dims, dst_info, src_info, dst_elmts, is_dst_on_acc);
    }
  }
  else{
    if(src_elmts == 1){
      _XMP_fatal("_XMP_mpi_get: scalar_mput is unimplemented");
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
}

void _XMP_mpi_coarray_attach(_XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size, const bool is_acc)
{
  MPI_Win win = _xmp_mpi_distarray_win;
#ifdef _XMP_XACC
  if(is_acc){
    win = _xmp_mpi_distarray_win_acc;
  }
#endif
  MPI_Win_attach(win, addr, coarray_size);
  
  _XMP_nodes_t *nodes = _XMP_get_execution_nodes();
  int comm_size = nodes->comm_size;
  MPI_Comm comm = *(MPI_Comm *)nodes->comm;

  char **each_addr;  // head address of a local array on each node
  each_addr = (char**)_XMP_alloc(sizeof(char *) * comm_size);

  MPI_Allgather(&addr, sizeof(char *), MPI_BYTE,
		each_addr, sizeof(char *), MPI_BYTE,
		comm); // exchange displacement

  if(is_acc){
#ifdef _XMP_XACC
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = addr;
    coarray_desc->win_acc = win;
#endif
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = addr;
    coarray_desc->win = win;
  }
}
