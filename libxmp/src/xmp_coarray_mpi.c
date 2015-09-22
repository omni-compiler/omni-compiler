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
static unsigned int _num_of_puts = 0;
static unsigned int _num_of_puts_acc = 0;

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
  char *real_addr;
  struct _shift_queue_t *shift_queue = is_acc? &_shift_queue_acc : &_shift_queue;
  size_t shift;

  each_addr = (char**)_XMP_alloc(sizeof(char *) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++){
    each_addr[i] = (char *)(shift_queue->total_shift);
  }
  real_addr = (is_acc? _xmp_mpi_onesided_buf_acc : _xmp_mpi_onesided_buf) + shift_queue->total_shift;

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
    coarray_desc->addr_dev = each_addr;
    coarray_desc->real_addr_dev = real_addr;
  }else{
    coarray_desc->addr = each_addr;
    coarray_desc->real_addr = real_addr;
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
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_mpi_shortcut_put(const int target_rank, const size_t dst_offset, const size_t src_offset,
			   const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size, const bool is_acc)
{
  if(dst_elmts == src_elmts){
    size_t transfer_size = elmt_size * dst_elmts;
    char *laddr = (is_acc? src_desc->real_addr_dev : src_desc->real_addr) + src_offset;
    char *raddr = (is_acc? dst_desc->addr_dev[target_rank] : dst_desc->addr[target_rank]) + dst_offset;

    XACC_DEBUG("put(src_p=%p, size=%zd, target=%d, dst_p=%p, is_acc=%d)", laddr, transfer_size, target_rank, raddr, is_acc);
    MPI_Put((void*)laddr, transfer_size, MPI_BYTE, target_rank,
    	    (MPI_Aint)raddr, transfer_size, MPI_BYTE,
	    is_acc? _xmp_mpi_onesided_win_acc : _xmp_mpi_onesided_win);

    if(is_acc){
      ++_num_of_puts_acc;
    }else{
      ++_num_of_puts;
    }
  }
  else if(src_elmts == 1){
    _XMP_fatal("unimplemented");
    //_gasnet_scalar_shortcut_mput(target_rank, dst_desc, src, dst_offset, dst_elmts, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/**
   Execute sync_memory
 */
void _XMP_mpi_sync_memory()
{
  if(_num_of_puts > 0){
    XACC_DEBUG("sync_memory(host)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win);
    MPI_Win_flush_all(_xmp_mpi_onesided_win);

    _num_of_puts = 0;
  }

  if(_num_of_puts_acc > 0){
    XACC_DEBUG("sync_memory(acc)");
    //MPI_Win_flush_local_all(_xmp_mpi_onesided_win_acc);
    MPI_Win_flush_all(_xmp_mpi_onesided_win_acc);
    _num_of_puts_acc = 0;
  }
}

/**
   Execute sync_all
 */
void _XMP_mpi_sync_all()
{
  _XMP_mpi_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}
