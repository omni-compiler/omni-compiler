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
#ifndef _SYNCIMAGE_SENDRECV
static unsigned int *_sync_images_table_disp;
#endif
static int _XMP_mpi_trans_rank(const _XMP_coarray_t *coarray, int const world_rank);

//external variables in xmp_onesided.c
extern int _XMP_flag_put_nb; // This variable is temporal
extern int _XMP_flag_get_nb; // This variable is temporal

extern int _XMP_flag_multi_win;

/*
static bool _is_put_blocking = true;
static bool _is_put_local_blocking = true;
static bool _is_get_blocking = true;
*/
#define _is_put_blocking (! _XMP_flag_put_nb)
#define _is_put_local_blocking (! _XMP_flag_put_nb)
#define _is_get_blocking (! _XMP_flag_get_nb)

static void _mpi_contiguous(const int op,
			    const int target_rank, 
			    const _XMP_coarray_t *remote_desc, const void *local,
			    const size_t remote_offset, const size_t local_offset,
			    const size_t transfer_size, const int is_remote_on_acc);

static void _mpi_non_contiguous(const int op,
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

static inline
MPI_Win get_window(const _XMP_coarray_t *desc, bool is_acc)
{
  MPI_Win win = desc->win;
#ifdef _XMP_XACC
  if(is_acc) win = desc->win_acc;
#endif

  if(! _XMP_flag_multi_win){
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
  }
  return win;
}

MPI_Win _XMP_mpi_coarray_get_window(const _XMP_coarray_t *desc, bool is_acc)
{
  return get_window(desc, is_acc);
}

static inline
char *get_remote_addr(const _XMP_coarray_t *desc, const int target_rank, const bool is_acc)
{
#ifdef _XMP_XACC
  if(is_acc){
    if(desc->addr_dev == NULL) return (char*)0;
    return desc->addr_dev[target_rank];
  }
#endif
  if(desc->addr == NULL) return (char*)0;
  return desc->addr[target_rank];
}

static inline
char *get_local_addr(const _XMP_coarray_t *desc, const bool is_acc)
{
#ifdef _XMP_XACC
  if(is_acc){
    return desc->real_addr_dev;
  }
#endif
  return desc->real_addr;
}

char *_XMP_mpi_coarray_get_remote_addr(const _XMP_coarray_t *desc, const int target_rank, const bool is_acc)
{
  return get_remote_addr(desc, target_rank, is_acc);
}

char *_XMP_mpi_coarray_get_local_addr(const _XMP_coarray_t *desc, const bool is_acc)
{
  return get_local_addr(desc, is_acc);
}

/**
   Set initial value to the shift queue
 */
void _XMP_mpi_build_shift_queue(bool is_acc)
{
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  
  shift_queue->max_size = _XMP_MPI_ONESIDED_COARRAY_SHIFT_QUEUE_INITIAL_SIZE;
  shift_queue->num      = 0;
  shift_queue->shifts   = malloc(sizeof(size_t) * shift_queue->max_size);
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
  size_t next_size = shift_queue->max_size * sizeof(size_t);
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


void _XMP_mpi_coarray_deallocate(_XMP_coarray_t *c, bool is_acc)
{
  if(_XMP_flag_multi_win){
    MPI_Win_unlock_all(c->win);
    _XMP_barrier(NULL);
    _XMP_mpi_onesided_dealloc_win(&(c->win), (void **)&(c->real_addr), is_acc);
  }
}

/**********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                 */
/* ARGUMENT    : [OUT] *coarray_desc : Descriptor of new coarray      */
/*               [OUT] **addr        : Double pointer of new coarray  */
/*               [IN] coarray_size   : Coarray size                   */
/**********************************************************************/
void _XMP_mpi_coarray_malloc(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size, bool is_acc)
{
  char **each_addr = NULL;  // gap_size on each node
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  size_t shift;
  char *real_addr = NULL;
  MPI_Win win = MPI_WIN_NULL;
  _XMP_nodes_t *nodes = _XMP_get_execution_nodes();
  MPI_Comm comm = *(MPI_Comm *)nodes->comm;

  if(coarray_size == 0){
    _XMP_fatal("_XMP_mpi_coarray_malloc: zero size is not allowed");
  }

  if(_XMP_flag_multi_win){
    _XMP_mpi_onesided_alloc_win(&win, (void**)&real_addr, coarray_size, comm, is_acc);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    XACC_DEBUG("addr=%p, size=%zd, is_acc=%d", real_addr, coarray_size, is_acc);
  }else{
  each_addr = (char**)_XMP_alloc(sizeof(char *) * _XMP_world_size);
  for(int i=0;i<_XMP_world_size;i++){
    each_addr[i] = (char *)(shift_queue->total_shift);
  }
  real_addr = _xmp_mpi_onesided_buf;
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
  }

  if(is_acc){
#ifdef _XMP_XACC
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = real_addr;
    coarray_desc->win_acc = win;
    coarray_desc->nodes = nodes;
#endif
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = real_addr;
    coarray_desc->win = win;
    coarray_desc->win_acc = MPI_WIN_NULL;
    coarray_desc->nodes = nodes;
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
/* NOTE       : Both dst and src are contiguous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_contiguous_put(const int org_target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			     const size_t dst_offset, const size_t src_offset,
			     const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_dst_on_acc, const bool is_src_on_acc)
{
  const int target_rank = _XMP_mpi_trans_rank(dst_desc, org_target_rank);

  size_t transfer_size = elmt_size * dst_elmts;
  char *src = get_local_addr(src_desc, is_src_on_acc);
  if(dst_elmts == src_elmts){
    _mpi_contiguous(_XMP_N_COARRAY_PUT,
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
/* NOTE       : Both dst and src are contiguous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_contiguous_get(const int org_target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			     const size_t dst_offset, const size_t src_offset,
			     const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_dst_on_acc, const bool is_src_on_acc)
{
  const int target_rank = _XMP_mpi_trans_rank(src_desc, org_target_rank);

  size_t transfer_size = elmt_size * dst_elmts;
  char *dst = get_local_addr(dst_desc, is_dst_on_acc);
  if(dst_elmts == src_elmts){
    _mpi_contiguous(_XMP_N_COARRAY_GET,
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
/* ARGUMENT    : [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE) */
/*               [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)      */
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
void _XMP_mpi_put(const int dst_contiguous, const int src_contiguous, const int org_target_rank, 
		  const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		  const void *src, const int dst_elmts, const int src_elmts,
		  const int is_dst_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t transfer_size = dst_desc->elmt_size * dst_elmts;

  const int target_rank = _XMP_mpi_trans_rank(dst_desc, org_target_rank);

  if(dst_elmts == src_elmts){
    if(dst_contiguous == _XMP_N_INT_TRUE && src_contiguous == _XMP_N_INT_TRUE){
      _mpi_contiguous(_XMP_N_COARRAY_PUT,
		      target_rank,
		      dst_desc, src,
		      dst_offset, src_offset,
		      transfer_size, is_dst_on_acc);
    }
    else{
      _mpi_non_contiguous(_XMP_N_COARRAY_PUT, target_rank,
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
/* ARGUMENT    : [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)      */
/*               [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE) */
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
void _XMP_mpi_get(const int src_contiguous, const int dst_contiguous, const int org_target_rank,
		  const int src_dims, const int dst_dims, const _XMP_array_section_t *src_info,
		  const _XMP_array_section_t *dst_info, const _XMP_coarray_t *src_desc,
		  void *dst, const int src_elmts, const int dst_elmts,
		  const int is_src_on_acc)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t transfer_size = src_desc->elmt_size * src_elmts;

  const int target_rank = _XMP_mpi_trans_rank(src_desc, org_target_rank);

  XACC_DEBUG("_XMP_mpi_get, dst_elmts = %d, src_elmts = %d\n", src_elmts, dst_elmts);

  if(src_elmts == dst_elmts){
    if(src_contiguous == _XMP_N_INT_TRUE && dst_contiguous == _XMP_N_INT_TRUE){
      _mpi_contiguous(_XMP_N_COARRAY_GET, 
		      target_rank,
		      src_desc, dst,
		      src_offset, dst_offset,
		      transfer_size, is_src_on_acc);
    }else{
      _mpi_non_contiguous(_XMP_N_COARRAY_GET, target_rank,
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

static void _win_sync()
{
  XACC_DEBUG("sync for host single coarray");
  XACC_DEBUG("sync for host single distarray");
  MPI_Win_sync(_xmp_mpi_onesided_win);
  MPI_Win_sync(_xmp_mpi_distarray_win);

#ifdef _XMP_XACC
  XACC_DEBUG("sync for acc single coarray");
  XACC_DEBUG("sync for acc single distarray");
  MPI_Win_sync(_xmp_mpi_onesided_win_acc);
  MPI_Win_sync(_xmp_mpi_distarray_win_acc);
#endif
}

/**
   Execute sync_memory
 */
void _XMP_mpi_sync_memory()
{
  if(_XMP_flag_multi_win){
    int num = 0;
    _XMP_coarray_t **coarrays = _XMP_coarray_get_list(&num);
    for(int i = 0; i < num; i++){
      MPI_Win win = coarrays[i]->win;
      if(win != MPI_WIN_NULL){
	XACC_DEBUG("flush_all for host a coarray (%ld)", (long)win);
	MPI_Win_flush_all(win);
	XACC_DEBUG("sync for host a coarray (%ld)", (long)win);
	MPI_Win_sync(win);
      }
#ifdef _XMP_XACC
      MPI_Win win_acc = coarrays[i]->win_acc;
      if(win_acc != MPI_WIN_NULL){
	XACC_DEBUG("flush_all for acc a coarray (%ld)", (long)win_acc);
	MPI_Win_flush_all(win_acc);
	XACC_DEBUG("sync for acc a coarray (%ld)", (long)win_acc);
	MPI_Win_sync(win_acc);
      }
#endif
    }
  }else{
    if(! _is_coarray_win_flushed){
      XACC_DEBUG("flush_all for host single coarray(%ld)", (long)_xmp_mpi_onesided_win);
      MPI_Win_flush_all(_xmp_mpi_onesided_win);

      _is_coarray_win_flushed = true;
    }

    if(! _is_distarray_win_flushed){
      XACC_DEBUG("flush_all for host single distarray(%ld)", (long)_xmp_mpi_distarray_win);
      MPI_Win_flush_all(_xmp_mpi_distarray_win);

      _is_distarray_win_flushed = true;
    }

#ifdef _XMP_XACC
    if(! _is_coarray_win_acc_flushed){
      XACC_DEBUG("flush_all for acc single coarray(%ld)", (long)_xmp_mpi_onesided_win_acc);
      MPI_Win_flush_all(_xmp_mpi_onesided_win_acc);

      _is_coarray_win_acc_flushed = true;
    }

    if(! _is_distarray_win_acc_flushed){
      XACC_DEBUG("flush_all for acc single distarray(%ld)", (long)_xmp_mpi_distarray_win_acc);
      MPI_Win_flush_all(_xmp_mpi_distarray_win_acc);

      _is_distarray_win_acc_flushed = true;
    }
#endif

    _win_sync();
  }
}

/**
   Execute sync_all
 */
void _XMP_mpi_sync_all()
{
  _XMP_mpi_sync_memory();
  _XMP_barrier(NULL);
  _XMP_mpi_sync_memory();
}

static inline
void _wait_puts(const int target_rank, const MPI_Win win)
{
    if(_is_put_blocking){
      XACC_DEBUG("flush(%d) for [host|acc]", target_rank);
      MPI_Win_flush(target_rank, win);
    }else if(_is_put_local_blocking){
      XACC_DEBUG("flush_local(%d) for [host|acc]", target_rank);
      MPI_Win_flush_local(target_rank, win);
    }
}
static inline
void _wait_gets(const int target_rank, const MPI_Win win)
{
  if(_is_get_blocking){
    XACC_DEBUG("flush_local(%d) for [host|acc]", target_rank);
    MPI_Win_flush_local(target_rank, win);
  }
}

static void _mpi_contiguous(const int op,
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
    XACC_DEBUG("contiguous_put(local=%p, size=%zd, target=%d, remote=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_remote_on_acc);
    MPI_Put((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    win);
    _wait_puts(target_rank, win);
  }else if(op == _XMP_N_COARRAY_GET){
    XACC_DEBUG("contiguous_get(local=%p, size=%zd, target=%d, remote=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_remote_on_acc);
    MPI_Get((void*)laddr, transfer_size, MPI_BYTE, target_rank,
	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    win);
    _wait_gets(target_rank, win);
  }else{
    _XMP_fatal("invalid coarray operation type");
  }

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

static void _mpi_make_types(MPI_Datatype types[], const int dims, const _XMP_array_section_t sections[], const size_t element_size)
{
  for(int i = dims - 1; i >= 0; i--){
    const _XMP_array_section_t *section = sections + i;
    const int count = section->length;
    const int blocklength = (i == dims - 1)? element_size : 1;
    const int stride = section->distance * section->stride;
    const MPI_Datatype oldtype = (i == dims - 1)? MPI_BYTE : types[i + 1];
    XACC_DEBUG("type, dim=%d, start=%lld, length=%lld, stride=%lld, (c,b,s)=(%d,%d,%d)\n", i, section->start, section->length, section->stride, count, blocklength, stride);
    MPI_Type_create_hvector(count, blocklength, stride, oldtype, types + i);
    MPI_Type_commit(types + i);
  }
}

static void _mpi_free_types(MPI_Datatype types[], const int dims)
{
  for(int i = 0; i < dims; i++){
    MPI_Type_free(types + i);
  }
}

static void _mpi_non_contiguous(const int op, const int target_rank,
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

  _mpi_make_types(local_types,  local_dims,  local_info,  element_size);
  _mpi_make_types(remote_types, remote_dims, remote_info, element_size);

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

  //free datatype
  _mpi_free_types(local_types,  local_dims );
  _mpi_free_types(remote_types, remote_dims);

  if(op == _XMP_N_COARRAY_PUT){
    _wait_puts(target_rank, win);
  }else if (op == _XMP_N_COARRAY_GET){
    _wait_gets(target_rank, win);
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
  _wait_puts(target_rank, win);
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

  //we have to wait completion of the get
  XACC_DEBUG("flush_local(%d) for [host|acc]", target_rank);
  MPI_Win_flush_local(target_rank, win);

  _unpack_scalar((char*)dst, dst_dims, laddr, dst_info);
}




void _XMP_mpi_coarray_attach(_XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size, const bool is_acc)
{
  MPI_Win win = MPI_WIN_NULL;
  char **each_addr = NULL;  // head address of a local array on each node

  _XMP_nodes_t *nodes = _XMP_get_execution_nodes();
  int comm_size = nodes->comm_size;
  MPI_Comm comm = *(MPI_Comm *)nodes->comm;

  XACC_DEBUG("attach addr=%p, size=%zd, is_acc=%d", addr, coarray_size, is_acc);

  if(_XMP_flag_multi_win){
    _XMP_mpi_onesided_create_win(&win, addr, coarray_size, comm);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
  }else{
    win = _xmp_mpi_distarray_win;
#ifdef _XMP_XACC
    if(is_acc){
      win = _xmp_mpi_distarray_win_acc;
    }
#endif
    MPI_Win_attach(win, addr, coarray_size);

    each_addr = (char**)_XMP_alloc(sizeof(char *) * comm_size);

    MPI_Allgather(&addr, sizeof(char *), MPI_BYTE,
		  each_addr, sizeof(char *), MPI_BYTE,
		  comm); // exchange displacement
  }

  if(is_acc){
#ifdef _XMP_XACC
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = addr;
    coarray_desc->win_acc = win;
    coarray_desc->nodes = nodes;
#endif
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = addr;
    coarray_desc->win = win;
    coarray_desc->win_acc = MPI_WIN_NULL;
    coarray_desc->nodes = nodes;
  }
}

void _XMP_mpi_coarray_detach(_XMP_coarray_t *coarray_desc, const bool is_acc)
{
  if(_XMP_flag_multi_win){
    MPI_Win win = is_acc? coarray_desc->win_acc : coarray_desc->win;
    MPI_Win_unlock_all(win);
    _XMP_barrier(NULL);
    _XMP_mpi_onesided_destroy_win(&win);
  }else{
    MPI_Win win = _xmp_mpi_distarray_win;
    void *real_addr = coarray_desc->real_addr;
#ifdef _XMP_XACC
    if(is_acc){
      win = _xmp_mpi_distarray_win_acc;
      real_addr = coarray_desc->real_addr_dev;
    }
#endif

    MPI_Win_detach(win, real_addr);
  }

  if(is_acc){
#ifdef _XMP_XACC
    _XMP_free(coarray_desc->addr_dev); //FIXME may be wrong
    coarray_desc->addr_dev = NULL;
    coarray_desc->real_addr_dev = NULL;
    coarray_desc->win_acc = MPI_WIN_NULL;
    coarray_desc->nodes = NULL;
#endif
  }else{
    _XMP_free(coarray_desc->addr);
    coarray_desc->addr = NULL;
    coarray_desc->real_addr = NULL;
    coarray_desc->win = MPI_WIN_NULL;
    coarray_desc->nodes = NULL;
  }
}


/**
 * Build table and Initialize for sync images
 */
void _XMP_mpi_build_sync_images_table()
{
  size_t table_size = sizeof(unsigned int) * _XMP_world_size;
#ifdef _SYNCIMAGE_SENDRECV
  _sync_images_table = _XMP_alloc(table_size);
#else
  struct _shift_queue_t *shift_queue = &_shift_queue;
  _sync_images_table = (unsigned int*)(_xmp_mpi_onesided_buf + shift_queue->total_shift);
  _sync_images_table_disp = (unsigned int*)(shift_queue->total_shift);

  size_t shift;
  if(table_size % _XMP_MPI_ALIGNMENT == 0)
    shift = table_size;
  else{
    shift = ((table_size / _XMP_MPI_ALIGNMENT) + 1) * _XMP_MPI_ALIGNMENT;
  }
  _push_shift_queue(shift_queue, shift);
#endif

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
static void _notify_sync_images(const int num, const int *rank_set)
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
static void _wait_sync_images(const int num, const int *rank_set)
{
  int unrecieved_rank_set[num];
  int num_unrecieved_ranks = num;

  for(int i = 0; i < num; i++){
    unrecieved_rank_set[i] = rank_set[i];
  }

  while(1){
    for(int i = 0; i < num_unrecieved_ranks; i++){
      int rank = unrecieved_rank_set[i];
      if(_sync_images_table[rank] > 0){
	_add_sync_images_table(rank, -1);

	// decrement num_unrecieved_ranks and overwrite the current element with the tail element
	unrecieved_rank_set[i] = unrecieved_rank_set[--num_unrecieved_ranks];
      }
    }

    if(num_unrecieved_ranks == 0) break;

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
void _XMP_mpi_sync_images(const int num, const int* image_set, int* status)
{
  _XMP_mpi_sync_memory();

  if(num == 0){
    return;
  }
  else if(num < 0){
    fprintf(stderr, "Invalid value is used in xmp_sync_memory. The first argument is %d\n", num);
    _XMP_fatal_nomsg();
  }

  _notify_sync_images(num, image_set);
  _wait_sync_images(num, image_set);

  _XMP_mpi_sync_memory();
}


void _XMP_sync_images_EXEC(int* status)
{
  _XMP_mpi_sync_memory();
  _XMP_barrier(NULL);
}

void _XMP_sync_images_COMM(MPI_Comm *comm, int* status)
{
  _XMP_mpi_sync_memory();
  MPI_Barrier(*comm);
}

static
int _transRank_withComm(MPI_Comm comm1, int rank1, MPI_Comm comm2)
{
  int rank2;
  MPI_Group group1, group2;
  int stat1, stat2, stat3;

  stat1 = MPI_Comm_group(comm1, &group1);
  stat2 = MPI_Comm_group(comm2, &group2);

  stat3 = MPI_Group_translate_ranks(group1, 1, &rank1, group2, &rank2);
  //(in:Group1, n, rank1[n], Group2, out:rank2[n])
  if (rank2 == MPI_UNDEFINED){
    rank2 = -1;
  }

  if (stat1 != 0 || stat2 != 0 || stat3 != 0)
    fprintf(stderr, "INTERNAL: _transRank_withComm failed with stat1=%d, stat2=%d, stat3=%d",
	    stat1, stat2, stat3);

  return rank2;
}

static int _XMP_mpi_trans_rank(const _XMP_coarray_t *coarray, int const world_rank)
{
  if(! _XMP_flag_multi_win){ //if single window mode, return world_rank
    return world_rank;
  }

  _XMP_nodes_t* nodes = coarray->nodes;

  if(nodes == NULL || nodes->attr == _XMP_ENTIRE_NODES){ //assume that comm is entire nodes if nodes is NULL
    return world_rank;
  }

  MPI_Comm comm = *(MPI_Comm*)nodes->comm;

  if(comm == MPI_COMM_WORLD){
    return world_rank;
  }

  int rank = _transRank_withComm(MPI_COMM_WORLD, world_rank, comm);

  return rank;
}

void _XMP_mpi_coarray_regmem(_XMP_coarray_t *coarray_desc, void *real_addr, const size_t coarray_size, bool is_acc)
{
  char **each_addr = NULL;
  MPI_Win win = MPI_WIN_NULL;
  _XMP_nodes_t *nodes = _XMP_get_execution_nodes();
  MPI_Comm comm = *(MPI_Comm *)nodes->comm;

  if(! _XMP_flag_multi_win){
    _XMP_fatal("single window mode does not support coarray regmem");
  }

  if(coarray_size == 0){
    _XMP_fatal("_XMP_mpi_coarray_regmem: zero size is not allowed");
  }

  _XMP_mpi_onesided_create_win(&win, real_addr, coarray_size, comm);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

  XACC_DEBUG("addr=%p, size=%zd, is_acc=%d", real_addr, coarray_size, is_acc);

  if(is_acc){
#ifdef _XMP_XACC
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = real_addr;
    coarray_desc->win_acc = win;
    coarray_desc->nodes = nodes;
#endif
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = real_addr;
    coarray_desc->win = win;
    coarray_desc->win_acc = MPI_WIN_NULL;
    coarray_desc->nodes = nodes;
  }
}

void _XMP_mpi_coarray_deregmem(_XMP_coarray_t *c)
{
  if(! _XMP_flag_multi_win){
    _XMP_fatal("single window mode does not support coarray deregmem");
  }

  MPI_Win_unlock_all(c->win);
  _XMP_barrier(NULL);
  _XMP_mpi_onesided_destroy_win(&(c->win));
}
