//#define DEBUG 1

#include "tca-api.h"
#include "xmp_internal.h"

#define NUM_COMM_CACHES (16)


const static int _WAIT_TAG_DATA = 2;
const static int _WAIT_TAG_EVENT = 3;
const static int _DMAC_CHANNEL = 0;
const static int _WAIT_SLOT_DATA = 4;

static tcaHandle *_incomplete_dma_handle = NULL;

//decl for caching
enum comm_pattern{
  contiguous,
  blockstride,
  others,
};

typedef struct _coarray_comm_t{
  enum comm_pattern pattern;
  int target_rank;
  const _XMP_coarray_t *dst_desc;
  const _XMP_coarray_t *src_desc;

  //for contiguous
  size_t dst_offset;
  size_t src_offset;
  size_t blocklen;
  size_t elmt_size;

  //for blockstride
  size_t dst_stride;
  size_t src_stride;
  size_t count;

  tcaDesc *tca_desc;
}_coarray_comm_t;

static _coarray_comm_t *_comm_cache[NUM_COMM_CACHES];
static int _comm_cache_tail = 0;

static _coarray_comm_t* find_comm(enum comm_pattern pattern,
				  int target_rank,
				  const _XMP_coarray_t *dst_desc,
				  const _XMP_coarray_t *src_desc,
				  size_t dst_offset,
				  size_t src_offset,
				  size_t blocklen,
				  size_t dst_stride,
				  size_t src_stride,
				  size_t count,
				  size_t elmt_size
				  );
static void exec_coarray_comm(_coarray_comm_t * comm);
/* static _coarray_comm_t* alloc_coarray_comm(enum comm_pattern pattern, */
/* 					   int target_rank, */
/* 					   const _XMP_coarray_t *dst_desc, */
/* 					   const _XMP_coarray_t *src_desc, */
/* 					   size_t dst_offset, */
/* 					   size_t src_offset, */
/* 					   size_t blocklen, */
/* 					   size_t dst_stride, */
/* 					   size_t src_stride, */
/* 					   size_t count, */
/* 					   size_t elmt_size */
/* 					   ); */
static _coarray_comm_t* create_coarray_comm(enum comm_pattern pattern,
					   int target_rank,
					   const _XMP_coarray_t *dst_desc,
					   const _XMP_coarray_t *src_desc,
					   size_t dst_offset,
					   size_t src_offset,
					   size_t blocklen,
					   size_t dst_stride,
					   size_t src_stride,
					   size_t count,
					   size_t elmt_size
					   );

static void cache_comm(_coarray_comm_t *comm);
/* static void free_coarray_comm(_coarray_comm_t *comm); */
static void destroy_coarray_comm(_coarray_comm_t *comm);
static void wait_incomplete_DMA();
static void set_incomplete_DMA(tcaHandle *handle);

static void _check_transfer(char* src_addr, char* dst_addr, size_t transfer_size)
{
  if(transfer_size & 0x3){
    fprintf(stderr, "transfer_size must be multiples of four (%zu)\n", transfer_size);
    exit(1);
  }
  if( ((uint64_t)src_addr & 0xF) != ( (uint64_t)dst_addr & 0xF) ){
    fprintf(stderr, "lower 4 bits of dst address must be equal to that of src address (src:%p, dst:%p)\n", src_addr, dst_addr);
  }
}

/**********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                 */
/* ARGUMENT    : [OUT] *coarray_desc : Descriptor of new coarray      */
/*               [OUT] **addr        : Double pointer of new coarray  */
/*               [IN] coarray_size   : Coarray size                   */
/**********************************************************************/
void _XMP_tca_malloc_do(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size)
{
  char *real_addr;
  tcaHandle handle;

  _XMP_tca_lock();
  TCA_CHECK(tcaMalloc((void**)&real_addr, coarray_size, tcaMemoryGPU));
  TCA_CHECK(tcaCreateHandle(&handle, real_addr, coarray_size, tcaMemoryGPU));
  _XMP_tca_unlock();
  
  tcaHandle* handles = (tcaHandle*)_XMP_alloc(sizeof(tcaHandle)*_XMP_world_size); // handle of a local array on each node

  MPI_Allgather(&handle, sizeof(tcaHandle), MPI_BYTE,
		handles, sizeof(tcaHandle), MPI_BYTE, MPI_COMM_WORLD);

  coarray_desc->addr_dev = (char**)handles;
  coarray_desc->real_addr_dev = real_addr;
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
void _XMP_tca_shortcut_put(const int target_rank, const size_t dst_offset, const size_t src_offset,
			   const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
			   const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  if(dst_elmts != src_elmts){
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }

  _coarray_comm_t *comm = find_comm(contiguous,
				    target_rank,
				    dst_desc, src_desc,
				    dst_offset, src_offset,
				    dst_elmts,
				    0,0,0,
				    elmt_size);

  if(comm != NULL){
    exec_coarray_comm(comm);
    return;
  }

  comm = create_coarray_comm(contiguous,
			    target_rank,
			    dst_desc, src_desc,
			    dst_offset, src_offset,
			    dst_elmts,
			    0,0,0,
			    elmt_size);
  exec_coarray_comm(comm);
  cache_comm(comm);
}

/**
   Execute sync_memory
 */
void _XMP_tca_sync_memory()
{
  wait_incomplete_DMA();
}

static _coarray_comm_t* find_comm(enum comm_pattern pattern,
				  int target_rank,
				  const _XMP_coarray_t *dst_desc,
				  const _XMP_coarray_t *src_desc,
				  size_t dst_offset,
				  size_t src_offset,
				  size_t blocklen,
				  size_t dst_stride,
				  size_t src_stride,
				  size_t count,
				  size_t elmt_size
				  )
{
  for(int i = 0; i < NUM_COMM_CACHES; i++){
    _coarray_comm_t* comm = _comm_cache[i];
    if(comm == NULL) continue;

    if(comm->pattern != pattern) continue;

    if(pattern == contiguous &&
       comm->target_rank == target_rank &&
       comm->dst_desc == dst_desc &&
       comm->src_desc == src_desc &&
       comm->dst_offset == dst_offset &&
       comm->src_offset == src_offset &&
       comm->blocklen == blocklen &&
       comm->elmt_size == elmt_size){
      XACC_DEBUG("hit cache (size=%zd)", blocklen * elmt_size);
      return comm;
    }

    if(pattern == blockstride){
      _XMP_fatal("unimplemented");
    }
  }

  XACC_DEBUG("miss cache (size=%zd)", blocklen * elmt_size);
  return NULL;
}

static void wait_incomplete_DMA()
{
  XACC_DEBUG("wait incomplete DMA");
  //wait last dma src handle
  if(_incomplete_dma_handle == NULL){
    XACC_DEBUG("no wait incomplete DMA exist");
    return;
  }
  _XMP_tca_lock();
  XACC_DEBUG("get lock");
  TCA_CHECK(tcaWaitDMARecvDesc(_incomplete_dma_handle, _WAIT_SLOT_DATA, _WAIT_TAG_DATA));
  TCA_CHECK(tcaWaitDMAC(_DMAC_CHANNEL));
  _XMP_tca_unlock();
  XACC_DEBUG("free lock");

  _incomplete_dma_handle = NULL; //reset
  XACC_DEBUG("complete DMA");
}

static void set_incomplete_DMA(tcaHandle *handle)
{
  if(_incomplete_dma_handle != NULL){
    _XMP_fatal("exist incomplete dma");
  }
  
  //set last dma src handle
  _incomplete_dma_handle = handle;
}

static void exec_coarray_comm(_coarray_comm_t * comm)
{
  if(comm == NULL){
    _XMP_fatal("try to execute null comm");
    return;
  }

  wait_incomplete_DMA();

  _XMP_tca_lock();
  TCA_CHECK(tcaDescSet(comm->tca_desc, _DMAC_CHANNEL));
  TCA_CHECK(tcaStartDMADesc(_DMAC_CHANNEL));
  _XMP_tca_unlock();

  tcaHandle* src_handles = (tcaHandle*)(comm->src_desc->addr_dev);
  set_incomplete_DMA(&(src_handles[_XMP_world_rank]));
}

/* static _coarray_comm_t* alloc_coarray_comm(enum comm_pattern pattern, */
/* 					   int target_rank, */
/* 					   const _XMP_coarray_t *dst_desc, */
/* 					   const _XMP_coarray_t *src_desc, */
/* 					   size_t dst_offset, */
/* 					   size_t src_offset, */
/* 					   size_t blocklen, */
/* 					   size_t dst_stride, */
/* 					   size_t src_stride, */
/* 					   size_t count, */
/* 					   size_t elmt_size) */
/* { */
/*   _coarray_comm_t* comm = (_coarray_comm_t*)_XMP_alloc(sizeof(_coarray_comm_t)); */
/*   comm->pattern = pattern; */
/*   comm->target_rank = target_rank; */
/*   comm->dst_desc = dst_desc; */
/*   comm->src_desc = src_desc; */
/*   comm->dst_offset = dst_offset; */
/*   comm->src_offset = src_offset; */
/*   comm->blocklen = blocklen; */
/*   comm->dst_stride = dst_stride; */
/*   comm->src_stride = src_stride; */
/*   comm->count = count; */
/*   comm->elmt_size = elmt_size; */
  
/*   return comm; */
/* } */


static void cache_comm(_coarray_comm_t *comm)
{
  if(_comm_cache[_comm_cache_tail] != NULL){
    destroy_coarray_comm(_comm_cache[_comm_cache_tail]);
  }
  _comm_cache[_comm_cache_tail] = comm;
  if(++_comm_cache_tail == NUM_COMM_CACHES){
    _comm_cache_tail = 0;
  }
}

/* static void free_coarray_comm(_coarray_comm_t *comm) */
/* { */
/*   TCA_CHECK(tcaDescFree(comm->tca_desc)); */
/*   _XMP_free(comm); */
/* } */

static void destroy_coarray_comm(_coarray_comm_t *comm)
{
  _XMP_tca_lock();
  TCA_CHECK(tcaDescFree(comm->tca_desc));
  _XMP_tca_unlock();
  _XMP_free(comm);
}

static _coarray_comm_t* create_coarray_comm(enum comm_pattern pattern,
					   int target_rank,
					   const _XMP_coarray_t *dst_desc,
					   const _XMP_coarray_t *src_desc,
					   size_t dst_offset,
					   size_t src_offset,
					   size_t blocklen,
					   size_t dst_stride,
					   size_t src_stride,
					   size_t count,
					   size_t elmt_size)
{
  _coarray_comm_t* comm = (_coarray_comm_t*)_XMP_alloc(sizeof(_coarray_comm_t));

  //set
  comm->pattern = pattern;
  comm->target_rank = target_rank;
  comm->dst_desc = dst_desc;
  comm->src_desc = src_desc;
  comm->dst_offset = dst_offset;
  comm->src_offset = src_offset;
  comm->blocklen = blocklen;
  comm->dst_stride = dst_stride;
  comm->src_stride = src_stride;
  comm->count = count;
  comm->elmt_size = elmt_size;

  
  tcaHandle* src_handles = (tcaHandle*)(src_desc->addr_dev);
  tcaHandle* dst_handles = (tcaHandle*)(dst_desc->addr_dev);
  const int dma_flag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotifySelf;

  _XMP_tca_lock();

  // create desc
  tcaDesc* desc = tcaDescNew();

  switch(pattern){
  case contiguous:
    {
      size_t transfer_size = blocklen * elmt_size;
      char *laddr = src_desc->real_addr_dev + src_offset;
      char *raddr = dst_desc->real_addr_dev + dst_offset;
      _check_transfer(laddr, raddr, transfer_size);

      // config desc
      TCA_CHECK(tcaDescSetMemcpy(desc,
				 &dst_handles[target_rank] /*dst_handle*/, dst_offset /*dst_offset*/,
				 &src_handles[_XMP_world_rank] /*src_handle*/, src_offset /*src_offset*/,
				 transfer_size /*size*/,
				 dma_flag, _WAIT_SLOT_DATA, _WAIT_TAG_DATA));
    }break;
  default:
    _XMP_fatal("unimplemented pattern");
  }

  _XMP_tca_unlock();
  
  comm->tca_desc = desc; 

  return comm;
}
