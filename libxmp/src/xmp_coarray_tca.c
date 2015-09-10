#include "tca-api.h"
#include "xmp_internal.h"

const static int _WAIT_TAG = 1;
const static int _WAIT_SLOT = 0;
const static int _DMAC_CHANNEL = 0;


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
  TCA_CHECK(tcaMalloc((void**)&real_addr, coarray_size, tcaMemoryGPU));
  
  tcaHandle handle;
  TCA_CHECK(tcaCreateHandle(&handle, real_addr, coarray_size, tcaMemoryGPU));
  
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

  size_t transfer_size = dst_elmts * elmt_size;
  char *laddr = src_desc->real_addr_dev + src_offset;
  char *raddr = dst_desc->real_addr_dev + dst_offset;
  
  _check_transfer(laddr, raddr, transfer_size);
  
  //    _num_of_puts++;
  tcaHandle* src_handles = (tcaHandle*)(src_desc->addr_dev);
  tcaHandle* dst_handles = (tcaHandle*)(dst_desc->addr_dev);
  const int dma_flag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotifySelf;

  // create desc
  tcaDesc* desc = tcaDescNew();
  
  // config desc
  TCA_CHECK(tcaDescSetMemcpy(desc,
			     &dst_handles[target_rank] /*dst_handle*/, dst_offset /*dst_offset*/,
			     &src_handles[_XMP_world_rank] /*src_handle*/, src_offset /*src_offset*/,
			     transfer_size /*size*/,
			     dma_flag,_WAIT_SLOT, _WAIT_TAG));

  // set desc to dmac
  TCA_CHECK(tcaDescSet(desc, _DMAC_CHANNEL));

  // start dmac
  TCA_CHECK(tcaStartDMADesc(_DMAC_CHANNEL));

  // wait put locally
  TCA_CHECK(tcaWaitDMARecvDesc(&src_handles[_XMP_world_rank], _WAIT_SLOT, _WAIT_TAG));

  // free desc
  TCA_CHECK(tcaDescFree(desc));
}
