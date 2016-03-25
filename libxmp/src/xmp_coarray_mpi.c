//#define DEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"
#include <signal.h>
#include <string.h>

#define _SYNCIMAGE_SENDRECV

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
static unsigned int *_sync_images_table;
static unsigned int *_sync_images_table_disp;

static void _mpi_continuous(const int op,
			    const int target_rank, 
			    const _XMP_coarray_t *remote_desc, const void *local,
			    const size_t remote_offset, const size_t local_offset,
			    const size_t transfer_size, const int is_remote_on_acc);

static void _mpi_non_continuous(const int op,
				const int target_rank,
				const _XMP_coarray_t *remote_desc, const void *local_ptr,
				const size_t remote_offset, const size_t local_offset,
				const int remote_dims, const int local_dims,
				const _XMP_array_section_t *remote_info, const _XMP_array_section_t *local_info,
				const size_t remote_elmts, const int is_remote_on_acc);

static void _mpi_scalar_mput(const int target_rank, 
			     const _XMP_coarray_t *dst_desc, const void *src,
			     const size_t dst_offset, const size_t src_offset,
			     const int dst_dims, 
			     const _XMP_array_section_t *dst_info,
			     const bool is_dst_on_acc);
static void _mpi_scalar_mget(const int target_rank, 
			     void *dst, const _XMP_coarray_t *src_desc,
			     const size_t dst_offset, const size_t src_offset,
			     const int dst_dims, 
			     const _XMP_array_section_t *dst_info,
			     const bool is_src_on_acc);

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

  _XMP_free(shift_queue->shifts);
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
void _XMP_mpi_coarray_lastly_deallocate(bool is_acc){
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
      fprintf(stderr, "[ERROR] Cannot allocate coarray. Heap memory size of coarray is too small.\n");
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
/*               [IN] is_dst_on_acc: Whether dst is on acc or not       */
/*               [IN] is_src_on_acc: Whether src is on acc or not       */
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_shortcut_put(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			   const size_t dst_offset, const size_t src_offset,
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_dst_on_acc, const bool is_src_on_acc)
{
  size_t transfer_size = elmt_size * dst_elmts;
  char *src = get_local_addr(src_desc, is_src_on_acc);
  if(dst_elmts == src_elmts){
    _mpi_continuous(_XMP_N_COARRAY_PUT,
		    target_rank,
		    dst_desc, src,
		    dst_offset, src_offset,
		    transfer_size, is_dst_on_acc);
  }else if(src_elmts == 1){
    _XMP_array_section_t dst_info;
    dst_info.start = dst_offset / elmt_size;
    dst_info.length = dst_elmts;
    dst_info.stride = 1;
    dst_info.elmts = dst_info.start + dst_info.length;
    dst_info.distance = elmt_size;

    _mpi_scalar_mput(target_rank,
		     dst_desc, src,
		     dst_offset, src_offset,
		     1 /*dst_dims*/,
		     &dst_info,
		     is_dst_on_acc);
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
/*               [IN] is_dst_on_acc: Whether dst is on acc or not       */
/*               [IN] is_src_on_acc: Whether src is on acc or not       */
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_shortcut_get(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			   const size_t dst_offset, const size_t src_offset,
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_dst_on_acc, const bool is_src_on_acc)
{
  size_t transfer_size = elmt_size * dst_elmts;
  char *dst = get_local_addr(dst_desc, is_dst_on_acc);
  if(dst_elmts == src_elmts){
    _mpi_continuous(_XMP_N_COARRAY_GET,
		    target_rank,
		    src_desc, dst,
		    src_offset, dst_offset,
		    transfer_size, is_src_on_acc);
  }else if(src_elmts == 1){
    _XMP_array_section_t dst_info;
    dst_info.start = dst_offset / elmt_size;
    dst_info.length = dst_elmts;
    dst_info.stride = 1;
    dst_info.elmts = dst_info.start + dst_info.length;
    dst_info.distance = elmt_size;
    _mpi_scalar_mget(target_rank,
		     dst, src_desc,
		     dst_offset, src_offset,
		     1 /*dst_dims*/,
		     &dst_info,
		     is_src_on_acc);
  }else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
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
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/*               [IN] is_dst_on_acc  : Is destination on accelerator ? (TRUE/FALSE)    */
/***************************************************************************************/
void _XMP_mpi_put(const int dst_continuous, const int src_continuous, const int target_rank, 
		  const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		  const void *src, const int dst_elmts, const int src_elmts,
		  const int is_dst_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t transfer_size = dst_desc->elmt_size * dst_elmts;

  if(dst_elmts == src_elmts){
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      _mpi_continuous(_XMP_N_COARRAY_PUT,
		      target_rank,
		      dst_desc, src,
		      dst_offset, src_offset,
		      transfer_size, is_dst_on_acc);
    }
    else{
      _mpi_non_continuous(_XMP_N_COARRAY_PUT, target_rank,
			  dst_desc, src,
			  dst_offset, src_offset,
			  dst_dims, src_dims,
			  dst_info, src_info,
			  dst_elmts, is_dst_on_acc);
    }
  }
  else{
    if(src_elmts == 1){
      _mpi_scalar_mput(target_rank,
		       dst_desc, src,
		       dst_offset, src_offset,
		       dst_dims,
		       dst_info,
		       is_dst_on_acc);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
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
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [OUT] *dst          : Pointer of destination array                    */
/*               [IN] src_elmts      : Number of elements of source array              */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] is_src_on_acc  : Is source on accelerator ? (TRUE/FALSE)         */
/***************************************************************************************/
void _XMP_mpi_get(const int src_continuous, const int dst_continuous, const int target_rank,
		  const int src_dims, const int dst_dims, const _XMP_array_section_t *src_info,
		  const _XMP_array_section_t *dst_info, const _XMP_coarray_t *src_desc,
		  void *dst, const int src_elmts, const int dst_elmts,
		  const int is_src_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t transfer_size = src_desc->elmt_size * src_elmts;

  XACC_DEBUG("_XMP_mpi_get, dst_elmts = %d, src_elmts = %d\n", src_elmts, dst_elmts);

  if(src_elmts == dst_elmts){
    if(src_continuous == _XMP_N_INT_TRUE && dst_continuous == _XMP_N_INT_TRUE){
      _mpi_continuous(_XMP_N_COARRAY_GET, 
		      target_rank,
		      src_desc, dst,
		      src_offset, dst_offset,
		      transfer_size, is_src_on_acc);
    }else{
      _mpi_non_continuous(_XMP_N_COARRAY_GET, target_rank,
			  src_desc, dst,
			  src_offset, dst_offset,
			  src_dims, dst_dims,
			  src_info, dst_info,
			  src_elmts, is_src_on_acc);
    }
  }else if(src_elmts == 1){
    _mpi_scalar_mget(target_rank,
		     dst, src_desc,
		     dst_offset, src_offset,
		     dst_dims,
		     dst_info,
		     is_src_on_acc);
  }else{
    _XMP_fatal("Number of elements is invalid");
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

static void _mpi_continuous(const int op,
			    const int target_rank, 
			    const _XMP_coarray_t *remote_desc, const void *local,
			    const size_t remote_offset, const size_t local_offset,
			    const size_t transfer_size, const int is_remote_on_acc)
{
  if(transfer_size == 0) return;
  char *laddr = (char*)local + local_offset;
  char *raddr = get_remote_addr(remote_desc, target_rank, is_remote_on_acc) + remote_offset;
  MPI_Win win = get_window(remote_desc, is_remote_on_acc);

  if(op == _XMP_N_COARRAY_PUT){
    XACC_DEBUG("continuous_put(local=%p, size=%zd, target=%d, remote=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_remote_on_acc);
    MPI_Put((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    win);
  }else if(op == _XMP_N_COARRAY_GET){
    XACC_DEBUG("continuous_get(local=%p, size=%zd, target=%d, remote=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_remote_on_acc);
    MPI_Get((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    win);
  }else{
    _XMP_fatal("invalid coarray operation type");
  }
  MPI_Win_flush_local(target_rank, win);

  /*
  MPI_Request req[2];
  size_t size_multiple128k = (transfer_size / (128*1024)) * (128*1024);
  size_t size_rest = transfer_size - size_multiple128k;
  if(transfer_size >= (128*1024) && size_rest > 0 && size_rest <= (8*1024)){
    XACC_DEBUG("put(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d) divied! (%d,%d)", laddr, transfer_size, target_rank, raddr, is_dst_on_acc, size128k, size_rest);
    MPI_Rput((void*)laddr, size_multiple128k, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, size_multiple128k, MPI_BYTE,
	     win,req);
    MPI_Rput((void*)(laddr+size_multiple128k), size_rest, MPI_BYTE, target_rank,
	    (MPI_Aint)(raddr+size_multiple128k), size_rest, MPI_BYTE,
	    win,req+1);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
  }else{
    MPI_Rput((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	     win,req);
    MPI_Wait(req, MPI_STATUS_IGNORE);
  }
  */
}

static void _mpi_non_continuous(const int op, const int target_rank,
				const _XMP_coarray_t *remote_desc, const void *local_ptr,
				const size_t remote_offset, const size_t local_offset,
				const int remote_dims, const int local_dims,
				const _XMP_array_section_t *remote_info, const _XMP_array_section_t *local_info,
				const size_t remote_elmts, const int is_remote_on_acc)
{
  char *laddr = (char*)local_ptr + local_offset;
  char *raddr = get_remote_addr(remote_desc, target_rank, is_remote_on_acc) + remote_offset;
  MPI_Win win = get_window(remote_desc, is_remote_on_acc);

  MPI_Datatype local_types[_XMP_N_MAX_DIM], remote_types[_XMP_N_MAX_DIM];
  size_t element_size = remote_desc->elmt_size;

  for(int i = local_dims - 1; i >= 0; i--){
    const _XMP_array_section_t *section = local_info + i;
    int count = section->length;
    int blocklength = (i == local_dims - 1)? element_size : 1;
    int stride = section->distance * section->stride;
    MPI_Datatype oldtype = (i == local_dims - 1)? MPI_BYTE : local_types[i+1];
    XACC_DEBUG("local, dim=%d, start=%lld, length=%lld, stride=%lld, (c,b,s)=(%d,%d,%d)\n", i, section->start, section->length, section->stride, count, blocklength, stride);
    MPI_Type_create_hvector(count, blocklength, stride, oldtype, local_types + i);
    MPI_Type_commit(local_types + i);
  }
  for(int i = remote_dims - 1; i >= 0; i--){
    const _XMP_array_section_t *section = remote_info + i;
    int count = section->length;
    int blocklength = (i == remote_dims - 1)? element_size : 1;
    int stride = section->distance * section->stride;
    MPI_Datatype oldtype = (i == remote_dims - 1)? MPI_BYTE : remote_types[i+1];
    XACC_DEBUG("remote, dim=%d, start=%lld, length=%lld, stride=%lld, (c,b,s)=(%d,%d,%d)\n", i, section->start, section->length, section->stride, count, blocklength, stride);
    MPI_Type_create_hvector(count, blocklength, stride, oldtype, remote_types + i);
    MPI_Type_commit(remote_types + i);
  }


  //  XACC_DEBUG("nonc_put(src_p=%p, target=%d, dst_p=%p, is_acc=%d)", laddr, src_cnt,src_bl,src_str, target_rank, raddr, is_dst_on_acc);
  if(op == _XMP_N_COARRAY_PUT){
    MPI_Put((void*)laddr, 1, local_types[0], target_rank,
	    (MPI_Aint)raddr, 1, remote_types[0], 
	    win);
  }else if (op == _XMP_N_COARRAY_GET){
    MPI_Get((void*)laddr, 1, local_types[0], target_rank,
	    (MPI_Aint)raddr, 1, remote_types[0], 
	    win);
  }else{
    _XMP_fatal("invalid coarray operation type");
  }
  MPI_Win_flush_local(target_rank, win);

  //free datatype
  for(int i = 0; i < local_dims; i++){
    MPI_Type_free(local_types + i);
  }
  for(int i = 0; i < remote_dims; i++){
    MPI_Type_free(remote_types + i);
  }
}



static void _mpi_scalar_mput(const int target_rank, 
			     const _XMP_coarray_t *dst_desc, const void *src,
			     const size_t dst_offset, const size_t src_offset,
			     const int dst_dims, 
			     const _XMP_array_section_t *dst_info,
			     const bool is_dst_on_acc)
{
  int allelmt_dim = _XMP_get_dim_of_allelmts(dst_dims, dst_info);
  size_t element_size = dst_desc->elmt_size;
  size_t allelmt_size = (allelmt_dim == dst_dims)? element_size : dst_info[allelmt_dim].distance * dst_info[allelmt_dim].elmts;
  char *laddr = (allelmt_dim == dst_dims)? ((char*)src + src_offset) : _XMP_alloc(allelmt_size);
  char *raddr = get_remote_addr(dst_desc, target_rank, is_dst_on_acc) + dst_offset;
  MPI_Win win = get_window(dst_desc, is_dst_on_acc);

  XACC_DEBUG("scalar_mput(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d)", laddr, element_size, target_rank, raddr, is_dst_on_acc);

  XACC_DEBUG("allelmt_dim=%d, dst_dims=%d", allelmt_dim, dst_dims);
  if(allelmt_dim != dst_dims){
    //mcopy
    _XMP_array_section_t info;
    info.start = 0;
    info.length = allelmt_size/element_size;
    info.stride = 1;
    info.elmts = info.length;
    info.distance = element_size;
    _XMP_stride_memcpy_1dim(laddr, (char*)src+src_offset, &info, element_size, _XMP_SCALAR_MCOPY);
    XACC_DEBUG("mcopy(%lld, %lld, %lld), %lld",info.start, info.length, info.stride, info.elmts);
  }
  long long idxs[allelmt_dim+1];
  for(int i = 0; i < allelmt_dim+1; i++) idxs[i]=0;

  while(1){
    size_t offset = 0;
    for(int i = 0; i < allelmt_dim; i++){
      offset += dst_info[i].distance * idxs[i+1] * dst_info[i].stride;
    }

    MPI_Put((void*)laddr, allelmt_size, MPI_BYTE, target_rank,
	    (MPI_Aint)(raddr+offset), allelmt_size, MPI_BYTE,
	    win);

    ++idxs[allelmt_dim];
    for(int i = allelmt_dim-1; i >= 0; i--){
      long long length = dst_info[i].length;
      if(idxs[i+1] >= length){
	idxs[i+1] -= length;
	++idxs[i];
      }else{
	break;
      }
    }
    if(idxs[0] > 0){
      break;
    }
  }
  MPI_Win_flush_local(target_rank, win);
  if(allelmt_dim != dst_dims){
    _XMP_free(laddr);
  }
}

static void _unpack_scalar(char *dst, const int dst_dims, const char* src, 
			   const _XMP_array_section_t* dst_info)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  switch (dst_dims){
  case 1:
    _XMP_stride_memcpy_1dim(dst + dst_offset, src, dst_info, dst_info[0].distance, _XMP_SCALAR_MCOPY);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst + dst_offset, src, dst_info, dst_info[1].distance, _XMP_SCALAR_MCOPY);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst + dst_offset, src, dst_info, dst_info[2].distance, _XMP_SCALAR_MCOPY);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst + dst_offset, src, dst_info, dst_info[3].distance, _XMP_SCALAR_MCOPY);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst + dst_offset, src, dst_info, dst_info[4].distance, _XMP_SCALAR_MCOPY);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst + dst_offset, src, dst_info, dst_info[5].distance, _XMP_SCALAR_MCOPY);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst + dst_offset, src, dst_info, dst_info[6].distance, _XMP_SCALAR_MCOPY);
    break;
  default:
    _XMP_fatal("Dimension of coarray is too big.");
    break;
  }
}

static void _mpi_scalar_mget(const int target_rank, 
			     void *dst, const _XMP_coarray_t *src_desc,
			     const size_t dst_offset, const size_t src_offset,
			     const int dst_dims, 
			     const _XMP_array_section_t *dst_info,
			     const bool is_src_on_acc)
{
  char *laddr = (char*)dst + dst_offset;
  char *raddr = get_remote_addr(src_desc, target_rank, is_src_on_acc) + src_offset;
  MPI_Win win = get_window(src_desc, is_src_on_acc);
  size_t transfer_size = src_desc->elmt_size;

  XACC_DEBUG("scalar_mget(local_p=%p, size=%zd, target=%d, remote_p=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_src_on_acc);

  MPI_Get((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	  (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	  win);
  MPI_Win_flush_local(target_rank, win);

  _unpack_scalar((char*)dst, dst_dims, laddr, dst_info);
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

void _XMP_mpi_coarray_detach(_XMP_coarray_t *coarray_desc, const bool is_acc)
{
  MPI_Win win = _xmp_mpi_distarray_win;
  void *real_addr = coarray_desc->real_addr;
#ifdef _XMP_XACC
  if(is_acc){
    win = _xmp_mpi_distarray_win_acc;
    real_addr = coarray_desc->real_addr_dev;
  }
#endif

  MPI_Win_detach(win, real_addr);

  if(is_acc){
#ifdef _XMP_XACC
    _XMP_free(coarray_desc->addr_dev);
    coarray_desc->addr_dev = NULL;
    coarray_desc->real_addr_dev = NULL;
    coarray_desc->win_acc = MPI_WIN_NULL;
#endif
  }else{
    _XMP_free(coarray_desc->addr);
    coarray_desc->addr = NULL;
    coarray_desc->real_addr = NULL;
    coarray_desc->win = MPI_WIN_NULL;
  }
}


/**
 * Build table and Initialize for sync images
 */
void _XMP_mpi_build_sync_images_table()
{
  struct _shift_queue_t *shift_queue = &_shift_queue;
  _sync_images_table = (unsigned int*)(_xmp_mpi_onesided_buf + shift_queue->total_shift);
  _sync_images_table_disp = (unsigned int*)(shift_queue->total_shift);

  size_t table_size = sizeof(unsigned int) * _XMP_world_size;
  size_t shift;
  if(table_size % _XMP_MPI_ALIGNMENT == 0)
    shift = table_size;
  else{
    shift = ((table_size / _XMP_MPI_ALIGNMENT) + 1) * _XMP_MPI_ALIGNMENT;
  }
  _push_shift_queue(shift_queue, shift);

  for(int i=0;i<_XMP_world_size;i++)
    _sync_images_table[i] = 0;

  MPI_Barrier(MPI_COMM_WORLD);
}

#ifndef _SYNCIMAGE_SENDRECV
/**
   Add rank to remote table
   *
   * @param[in]  target_rank rank number
   * @param[in]  rank rank number
   * @param[in]  value value
   */
static void _add_remote_sync_images_table(const int target_rank, const int rank, const int value)
{
  const int val = value;
  MPI_Accumulate(&val, 1, MPI_INT, target_rank,
		 (MPI_Aint)&_sync_images_table_disp[rank], 1, MPI_INT, MPI_SUM, _xmp_mpi_onesided_win);
  XACC_DEBUG("accumulate(%d, %d) += %d", target_rank, rank, value);
}
#endif

/**
   Add rank to table
   *
   * @param[in]  rank rank number
   * @param[in]  value value
   */
static void _add_sync_images_table(const int rank, const int value)
{
#ifdef _SYNCIMAGE_SENDRECV
  _sync_images_table[rank] += value;
#else
  _add_remote_sync_images_table(_XMP_world_rank, rank, value);
#endif
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
      _add_sync_images_table(_XMP_world_rank, 1);
    }else{
#ifdef _SYNCIMAGE_SENDRECV
      MPI_Send(NULL, 0, MPI_BYTE, rank_set[i], _XMP_N_MPI_TAG_SYNCREQ, MPI_COMM_WORLD);
#else
      _add_remote_sync_images_table(rank_set[i], _XMP_world_rank, 1);
#endif
    }
  }

#ifndef _SYNCIMAGE_SENDRECV
  //MPI_Win_flush_all(_xmp_mpi_onesided_win);
#endif
}


/**
   Wait until recieving all request from all node
   *
   * @param[in]  num                       number of nodes
   * @param[in]  *rank_set                 rank set
*/
static void _wait_sync_images(const int num, int *rank_set)
{
  while(1){
    bool flag = true;

    for(int i=0;i<num;i++){
      if(rank_set[i] < 0) continue;
      if(_sync_images_table[rank_set[i]] > 0){
	_add_sync_images_table(rank_set[i], -1);
	rank_set[i] = -1;
      }else{
	flag = false;
      }
    }

    if(flag) break;

#ifdef _SYNCIMAGE_SENDRECV
    MPI_Status status;
    MPI_Recv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, _XMP_N_MPI_TAG_SYNCREQ, MPI_COMM_WORLD, &status);
    _add_sync_images_table(status.MPI_SOURCE, 1);
#else
    MPI_Win_flush_local_all(_xmp_mpi_onesided_win);
#endif
  }
}


/**
   Execute sync images
   *
   * @param[in]  num         number of nodes
   * @param[in]  *image_set  image set
   * @param[out] status      status
*/
void _XMP_mpi_sync_images(const int num, int* image_set, int* status)
{
  _XMP_mpi_sync_memory();

  if(num == 0){
    return;
  }
  else if(num < 0){
    fprintf(stderr, "Invalid value is used in xmp_sync_memory. The first argument is %d\n", num);
    _XMP_fatal_nomsg();
  }

  int rank_set[num];
  for(int i=0;i<num;i++){
    rank_set[i] = image_set[i] - 1;
  }

  _notify_sync_images(num, rank_set);
  _wait_sync_images(num, rank_set);
}


void _XMP_sync_images_EXEC(int* status)
{
  MPI_Win_flush_all(_xmp_mpi_onesided_win);
  _XMP_barrier_EXEC();
}


void _XMP_sync_images_COMM(MPI_Comm *comm, int* status)
{
  MPI_Win_flush_all(_xmp_mpi_onesided_win);
  MPI_Barrier(*comm);
}
