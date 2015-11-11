#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "mpi-ext.h"
#include "xmp_internal.h"
#define _XMP_FJRDMA_MAX_SIZE   16777212
#define _XMP_FJRDMA_MAX_MEMID       511
#define _XMP_FJRDMA_MAX_MPUT       1993
#define _XMP_FJRDMA_MAX_MGET        100 /** This value is trial */
#define _XMP_FJRDMA_MAX_COMM         60 /** This value is trial */
#define _XMP_FJRDMA_TAG               0
#define _XMP_SYNC_IMAGES_TAG          1
#define _XMP_FJRDMA_START_MEMID       3

static int _num_of_puts = 0, _num_of_gets = 0;
static struct FJMPI_Rdma_cq _cq;
static int _memid = _XMP_FJRDMA_START_MEMID; // _memid = 0 is used for put/get operations.
                                             // _memid = 1 is used for post/wait operations.
                                             // _memid = 2 is used for sync images
static uint64_t _local_rdma_addr, *_remote_rdma_addr;
static unsigned int *_sync_images_table;

/**
   Execute sync_memory for put operation
 */
static void _XMP_fjrdma_sync_memory_put()
{
  while(_num_of_puts != 0)
    if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC, &_cq) == FJMPI_RDMA_NOTICE)
      _num_of_puts--;
}

/**
   Execute sync_memory for get operation
*/
static void _XMP_fjrdma_sync_memory_get()
{
  while(_num_of_gets != 0)
    if(FJMPI_Rdma_poll_cq(_XMP_COARRAY_SEND_NIC, &_cq) == FJMPI_RDMA_NOTICE)
      _num_of_gets--;
}

/**
   Execute sync_memory
*/
void _XMP_fjrdma_sync_memory()
{
  _XMP_fjrdma_sync_memory_put();
  // _XMP_fjrdma_sync_memory_get don't need to be executed
}

/**
   Execute sync_all
*/
void _XMP_fjrdma_sync_all()
{
  _XMP_fjrdma_sync_memory();
  MPI_Barrier(MPI_COMM_WORLD);
}


/**
   transfer_size must be 4-Byte align
*/
static void _check_transfer_size(const size_t transfer_size)
{
  if((transfer_size&0x3) != 0){  // transfer_size % 4 != 0
    fprintf(stderr, "transfer_size must be multiples of four (%zu)\n", transfer_size);
    exit(1);
  }
}

/**
   The _XMP_FJMPI_Rdma_put() is a wrapper function of the FJMPI_Rdma_put().
   FJMPI_Rdma_put() cannot transfer more than _XMP_FJRDMA_MAX_SIZE(about 16MB) data.
   Thus, the _XMP_FJMPI_Rdma_put() executes mutiple put operations to trasfer more than 16MB data.
*/
static void _XMP_FJMPI_Rdma_put(const int target_rank, uint64_t raddr, uint64_t laddr, 
				const size_t transfer_size)
{
  if(transfer_size <= _XMP_FJRDMA_MAX_SIZE){
    FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC);
    _num_of_puts++;
  }
  else{
    int times = transfer_size / _XMP_FJRDMA_MAX_SIZE;
    int rest  = transfer_size - _XMP_FJRDMA_MAX_SIZE * times;

    for(int i=0;i<times;i++){
      FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, _XMP_FJRDMA_MAX_SIZE, _XMP_COARRAY_FLAG_NIC);
      raddr += _XMP_FJRDMA_MAX_SIZE;
      laddr += _XMP_FJRDMA_MAX_SIZE;
      _num_of_puts++;
    }
    
    if(rest != 0){
      FJMPI_Rdma_put(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, rest, _XMP_COARRAY_FLAG_NIC);
      _num_of_puts++;
    }
  }
}

/**
   The _XMP_FJMPI_Rdma_get() is a wrapper function of the FJMPI_Rdma_get().
   FJMPI_Rdma_get() cannot transfer more than _XMP_FJRDMA_MAX_SIZE(about 16MB) data.
   Thus, the _XMP_FJMPI_Rdma_get() executes mutiple put operations to trasfer more than 16MB data.
*/
static void _XMP_FJMPI_Rdma_get(const int target_rank, uint64_t raddr, uint64_t laddr,
				const size_t transfer_size)
{
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();

  if(transfer_size <= _XMP_FJRDMA_MAX_SIZE){
    FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, transfer_size, _XMP_COARRAY_FLAG_NIC);
    _num_of_gets++;
  }
  else{
    int times = transfer_size / _XMP_FJRDMA_MAX_SIZE;
    int rest  = transfer_size - _XMP_FJRDMA_MAX_SIZE * times;

    for(int i=0;i<times;i++){
      FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, _XMP_FJRDMA_MAX_SIZE, _XMP_COARRAY_FLAG_NIC);
      raddr += _XMP_FJRDMA_MAX_SIZE;
      laddr += _XMP_FJRDMA_MAX_SIZE;
      _num_of_gets++;
    }

    if(rest != 0){
      FJMPI_Rdma_get(target_rank, _XMP_FJRDMA_TAG, raddr, laddr, rest, _XMP_COARRAY_FLAG_NIC);
      _num_of_gets++;
    }
  }
}

#if defined(OMNI_TARGET_CPU_FX10) || defined(OMNI_TARGET_CPU_FX100)
/*********************************************************************************/
/* DESCRIPTION : Execute multiple put operation for FX10                         */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] *raddrs        : Remote addresses                          */
/*               [IN] *laddrs        : Local addresses                           */
/*               [IN] *lengths       : Lengths                                   */
/*               [IN] stride         : Stride. If stride is 0, the first raadrs, */
/*                                     laddrs, and lengths is used               */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : This function is used instead of FJMPI_Rdma_mput() which is used */
/*              on the K computer                                                */
/*********************************************************************************/
static void _FX10_Rdma_mput(const int target_rank, uint64_t *raddrs, uint64_t *laddrs,
                            const size_t *lengths, const int stride, const size_t transfer_elmts)
{
  if(stride == 0){
    for(int i=0;i<transfer_elmts;i++)
      _XMP_FJMPI_Rdma_put(target_rank, raddrs[i], laddrs[i], lengths[i]);
  }
  else{
    for(int i=0;i<transfer_elmts;i++){
      _XMP_FJMPI_Rdma_put(target_rank, raddrs[0], laddrs[0], lengths[0]);
      raddrs[0] += stride;
      laddrs[0] += stride;
    }
  }
}
#endif

/**
   Prevent MRQ overflow : Without this function, MRQ overflow often occurs.
                          Perhaps, RDMA put/get functions internally use MRQ.
                          This function should be used before and in put/get operations.
*/
static void _release_MRQ()
{
  if(_num_of_puts > _XMP_FJRDMA_MAX_COMM)
    _XMP_fjrdma_sync_memory();
}

/*********************************************************************************/
/* DESCRIPTION : Wrapper function for multiple put operations                    */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] *raddrs        : Remote addresses                          */
/*               [IN] *laddrs        : Local addresses                           */
/*               [IN] *lengths       : Lengths                                   */
/*               [IN] stride         : Stride. If stride is 0, the first raadrs, */
/*                                     laddrs, and lengths is used               */
/*               [IN] transfer_elmts : Number of transfer elements               */
/*********************************************************************************/
static void _RDMA_mput(const size_t target_rank, uint64_t* raddrs, uint64_t* laddrs,
		       size_t* lengths, const int stride, const size_t transfer_elmts)
{
#if defined(OMNI_TARGET_CPU_KCOMPUTER)
  FJMPI_Rdma_mput(target_rank, _XMP_FJRDMA_TAG, raddrs, laddrs, lengths, stride, transfer_elmts, _XMP_COARRAY_FLAG_NIC);
  _num_of_puts++;
#elif defined(OMNI_TARGET_CPU_FX10) || defined(OMNI_TARGET_CPU_FX100)
  _FX10_Rdma_mput(target_rank, raddrs, laddrs, lengths, stride, transfer_elmts);
#endif

  _release_MRQ();
}

/************************************************************************/
/* DESCRIPTION : Execute scalar multiple put operation                  */
/* ARGUMENT    : [IN] target_rank    : Target rank                      */
/*               [IN] *raddrs        : Remote addresses                 */
/*               [IN] *laddrs        : Local addresses                  */
/*               [IN] *lengths       : Lengths                          */
/*               [IN] transfer_elmts : Number of transfer elements      */
/*               [IN] elmt_size      : Element size                     */
/************************************************************************/
static void _fjrdma_scalar_mput_do(const size_t target_rank, uint64_t* raddrs, uint64_t* laddrs,
                                   size_t* lengths, const size_t transfer_elmts, const size_t elmt_size)
{
  if(transfer_elmts <= _XMP_FJRDMA_MAX_MPUT){
    _RDMA_mput(target_rank, raddrs, laddrs, lengths, 0, transfer_elmts);
  }
  else{
    int times      = transfer_elmts / _XMP_FJRDMA_MAX_MPUT + 1;
    int rest_elmts = transfer_elmts - _XMP_FJRDMA_MAX_MPUT * (times - 1);
    for(int i=0;i<times;i++){
      size_t tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MPUT : rest_elmts;
      _RDMA_mput(target_rank, &raddrs[i*_XMP_FJRDMA_MAX_MPUT], &laddrs[i*_XMP_FJRDMA_MAX_MPUT],
		 &lengths[i*_XMP_FJRDMA_MAX_MPUT], 0, tmp_elmts);
    }
  }
}

/**********************************************************************/
/* DESCRIPTION : Execute malloc operation for coarray                 */
/* ARGUMENT    : [OUT] *coarray_desc : Descriptor of new coarray      */
/*               [OUT] **addr        : Double pointer of new coarray  */
/*               [IN] coarray_size   : Coarray size                   */
/**********************************************************************/
void _XMP_fjrdma_malloc_do(_XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size)
{
  uint64_t *each_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);
  if(_memid == _XMP_FJRDMA_MAX_MEMID)
    _XMP_fatal("Too many coarrays. Number of coarrays is not more than 510.");

  *addr = _XMP_alloc(coarray_size);
  uint64_t laddr = FJMPI_Rdma_reg_mem(_memid, *addr, coarray_size);

  MPI_Barrier(MPI_COMM_WORLD);
  for(int ncount=0,i=1; i<_XMP_world_size+1; ncount++,i++){
    int partner_rank = (_XMP_world_rank + _XMP_world_size - i) % _XMP_world_size;
    if(partner_rank == _XMP_world_rank)
      each_addr[partner_rank] = laddr;
    else
      each_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _memid);

    if(ncount > _XMP_INIT_RDMA_INTERVAL){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }

  coarray_desc->real_addr = *addr;
  coarray_desc->addr = (void *)each_addr;
  _memid++;
}

/**
   Deallocate memory region when calling _XMP_coarray_lastly_deallocate()
*/
void _XMP_fjrdma_coarray_lastly_deallocate()
{
  if(_memid == _XMP_FJRDMA_START_MEMID) return;

  _memid--;
  FJMPI_Rdma_dereg_mem(_memid);
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
void _XMP_fjrdma_shortcut_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
			      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
			      const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  size_t transfer_size = dst_elmts * elmt_size;
  _check_transfer_size(transfer_size);

  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_offset;

  if(dst_elmts == src_elmts){
    _XMP_FJMPI_Rdma_put(target_rank, raddr, laddr, transfer_size);
  }
  else if(src_elmts == 1){
    uint64_t raddrs[dst_elmts], laddrs[dst_elmts];
    size_t lengths[dst_elmts];
    for(int i=0;i<dst_elmts;i++) raddrs[i]  = raddr + i * elmt_size;
    for(int i=0;i<dst_elmts;i++) laddrs[i]  = laddr;
    for(int i=0;i<dst_elmts;i++) lengths[i] = elmt_size;
    _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, dst_elmts, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }

  _release_MRQ();
}

/*************************************************************************/
/* DESCRIPTION : Execute put operation without preprocessing             */
/* ARGUMENT    : [IN] target_rank   : Target rank                        */
/*               [IN] dst_offset    : Offset size of destination coarray */
/*               [IN] src_offset    : Offset size of source coarray      */
/*               [OUT] *dst_desc    : Descriptor of destination coarray  */
/*               [IN] *src_desc     : Descriptor of source coarray       */
/*               [IN] *src          : Pointer of source array            */
/*               [IN] transfer_size : Transfer size                      */
/* NOTE       : Both dst and src are continuous arrays.                  */
/*              If src is NOT a coarray, src_desc is NULL.               */
/* EXAMPLE    :                                                          */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src            */
/*************************************************************************/
static void _fjrdma_continuous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				   const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				   char *src, const size_t transfer_size)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src + src_offset, transfer_size);
  else
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_offset;

  _XMP_FJMPI_Rdma_put(target_rank, raddr, laddr, transfer_size);

  if(src_desc == NULL)   
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute scalar multiple put operation                           */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [OUT] *dst_desc     : Descriptor of destination coarray         */
/*               [IN] *src_desc      : Descriptor of source coarray              */
/*               [IN] *src           : Pointer of source array                   */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If src is NOT a coarray, src_desc is NULL.                       */
/* EXAMPLE    :                                                                  */
/*     a[0:100:2]:[1] = b[0]; // a[] is a dst, b[] is a src                      */
/*********************************************************************************/
static void _fjrdma_scalar_mput(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset, 
				const _XMP_array_section_t *dst_info, const int dst_dims,
				const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				char *src, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;
  uint64_t raddrs[transfer_elmts], laddrs[transfer_elmts];
  size_t lengths[transfer_elmts];

  if(src_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src + src_offset, elmt_size);
  else
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_offset;

  // Set parameters for FJMPI_Rdma_mput
  _XMP_set_coarray_addresses(raddr, dst_info, dst_dims, transfer_elmts, raddrs);
  for(int i=0;i<transfer_elmts;i++) laddrs[i] = laddr;
  for(int i=0;i<transfer_elmts;i++) lengths[i] = elmt_size;

  _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, transfer_elmts, elmt_size);

  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*******************************************************************/
/* DESCRIPTION : Get array size                                   */
/* ARGUMENT    : [IN] *array_info : Information of array          */
/*               [IN] dims        : Number of dimensions of array */
/* RETURN     : Array size                                        */
/* EXAMPLE    :                                                   */
/*     int a[10][20]; -> 800                                      */
/******************************************************************/
static size_t _get_array_size(const _XMP_array_section_t *array_info, const int dims)
{
  return array_info[0].distance * array_info[0].elmts;
}

/*********************************************************************/
/* DESCRIPTION : Execute multiple put operation with the same stride */
/* ARGUMENT    : [IN] target_rank : Target rank                      */
/*               [IN] raddr       : Remote address                   */
/*               [IN] laddr       : Local address                    */
/*               [IN] *array_info : Information of array             */
/*               [IN] *array_dims : Number of dimensions             */
/*               [IN] elmt_size   : Element size                     */
/* NOTE       : The sixth argument of FJMPI_Rdma_mput() can NOT be 0 */
/* EXAMPLE    :                                                      */
/*     a[0:10:2]:[2] = b[2:10:2]; // a[] is a dst, b[] is a src      */
/*********************************************************************/
static void _fjrdma_NON_continuous_the_same_stride_mput(const int target_rank, uint64_t raddr, uint64_t laddr,
							const size_t transfer_elmts, const _XMP_array_section_t *array_info,
							const int array_dims, size_t elmt_size)
{
  size_t copy_chunk_dim = (size_t)_XMP_get_dim_of_allelmts(array_dims, array_info);
  size_t copy_chunk     = (size_t)_XMP_calc_copy_chunk(copy_chunk_dim, array_info);
  size_t copy_elmts     = transfer_elmts/(copy_chunk/elmt_size);
  size_t stride         = _XMP_calc_stride(array_info, array_dims, copy_chunk);

  if(copy_elmts <= _XMP_FJRDMA_MAX_MPUT){
    _RDMA_mput(target_rank, &raddr, &laddr, &copy_chunk, stride, copy_elmts);
  }
  else{
    int times      = copy_elmts / _XMP_FJRDMA_MAX_MPUT + 1;
    int rest_elmts = copy_elmts - _XMP_FJRDMA_MAX_MPUT * (times - 1);
    size_t tmp_elmts;

    for(int i=0;i<times;i++){
      uint64_t tmp_raddr = raddr + (i*_XMP_FJRDMA_MAX_MPUT*stride);
      uint64_t tmp_laddr = laddr + (i*_XMP_FJRDMA_MAX_MPUT*stride);
      tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MPUT : rest_elmts;
      _RDMA_mput(target_rank, &tmp_raddr, &tmp_laddr, &copy_chunk, stride, tmp_elmts);
    }
  }
}

/*********************************************************************************/
/* DESCRIPTION : Execute multiple put operation in general                       */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] raddr          : Remote address                            */
/*               [IN] laddr          : Local address                             */
/*               [IN] transfer_elmts : Number of transfer elements               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] *src_info      : Information of source array               */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [IN] elmt_size      : Element size                              */
/*********************************************************************************/
static void _fjrdma_NON_continuous_general_mput(const int target_rank, uint64_t raddr, uint64_t laddr,
						const size_t transfer_elmts,
						const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
						const int dst_dims, const int src_dims, size_t elmt_size)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  uint64_t raddrs[copy_elmts], laddrs[copy_elmts];
  size_t   lengths[copy_elmts];

  // Set parameters for FJMPI_Rdma_mput
  _XMP_set_coarray_addresses_with_chunk(raddrs, raddr, dst_info, dst_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(laddrs, laddr, src_info, src_dims, copy_chunk, copy_elmts);
  for(int i=0;i<copy_elmts;i++) lengths[i] = copy_chunk;

  _fjrdma_scalar_mput_do(target_rank, raddrs, laddrs, lengths, copy_elmts, elmt_size);
}

/*********************************************************************************/
/* DESCRIPTION : Execute put operation for NON-continuous region                 */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] *dst_info      : Information of destination array          */
/*               [IN] *src_info      : Information of source array               */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [OUT] *dst_desc     : Descriptor of destination array           */
/*               [IN] *src_desc      : Descriptor of source array                */
/*               [IN] *src           : Pointer of source array                   */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : src and/or dst arrays are NOT continuous.                        */
/*              If src is NOT a coarray, src_desc is NULL                        */
/*********************************************************************************/
static void _fjrdma_NON_continuous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       const int dst_dims, const int src_dims, 
				       const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				       void *src, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)dst_desc->addr[target_rank] + dst_offset;
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;

  if(src_desc == NULL){
    size_t array_size = _get_array_size(src_info, src_dims);
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, src, array_size) + src_offset;
  }
  else{
    laddr = (uint64_t)src_desc->addr[_XMP_world_rank] + src_offset;
  }

  if(_XMP_is_the_same_constant_stride(dst_info, src_info, dst_dims, src_dims)){
    _fjrdma_NON_continuous_the_same_stride_mput(target_rank, raddr, laddr, transfer_elmts,
						dst_info, dst_dims, elmt_size);
  }
  else{
    _fjrdma_NON_continuous_general_mput(target_rank, raddr, laddr, transfer_elmts, 
    					dst_info, src_info, dst_dims, src_dims, elmt_size);
  }
  
  if(src_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
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
void _XMP_fjrdma_put(const int dst_continuous, const int src_continuous, const int target_rank, 
		     const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		     const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc, 
		     const _XMP_coarray_t *src_desc, void *src, const int dst_elmts, const int src_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = dst_desc->elmt_size * dst_elmts;
  _check_transfer_size(transfer_size);

  if(dst_elmts == src_elmts){
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      _fjrdma_continuous_put(target_rank, dst_offset, src_offset, dst_desc, src_desc, src, transfer_size);
    }
    else{
      _fjrdma_NON_continuous_put(target_rank, dst_offset, src_offset, dst_info, src_info, dst_dims, src_dims, 
      				 dst_desc, src_desc, src, dst_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _fjrdma_scalar_mput(target_rank, dst_offset, src_offset, dst_info, dst_dims, dst_desc, src_desc, 
			  src, dst_elmts);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
  _release_MRQ();
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
/* NOTE       : Both dst and src are continuous coarrays.               */
/*              target_rank != __XMP_world_rank.                        */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_fjrdma_shortcut_get(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			      const uint64_t dst_offset, const uint64_t src_offset,
			      const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  size_t transfer_size = dst_elmts * elmt_size;
  _check_transfer_size(transfer_size);

  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_offset;
  
  // To complete put operations before the following get operation.
  _XMP_fjrdma_sync_memory();

  if(dst_elmts == src_elmts){
    _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, transfer_size);
    _XMP_fjrdma_sync_memory_get();
  }
  else if(src_elmts == 1){
    _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, elmt_size);
    _XMP_fjrdma_sync_memory_get();

    char *dst = dst_desc->real_addr + dst_offset;
    for(int i=1;i<dst_elmts;i++)
      memcpy(dst+i*elmt_size, dst, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
  _release_MRQ();
}

/************************************************************************/
/* DESCRIPTION : Execute get operation for continuous region            */
/* ARGUMENT    : [IN] target_rank   : Target rank                       */
/*               [IN] dst_offset    : Offset size of destination array  */
/*               [IN] src_offset    : Offset size of source array       */
/*               [OUT] *dst         : Pointer of destination array      */
/*               [IN] *dst_desc     : Descriptor of destination coarray */
/*               [IN] *src_desc     : Descriptor of source coarray      */
/*               [IN] transfer_size : Transfer size                     */
/* NOTE       : Both dst and src are continuous arrays.                 */
/*              If dst is NOT a coarray, dst_desc is NULL.              */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
static void _fjrdma_continuous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				   char *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				   const size_t transfer_size)
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;

  if(dst_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst + dst_offset, transfer_size);
  else
    laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_offset;
  
  _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, transfer_size);
  _XMP_fjrdma_sync_memory_get();

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute get operation for NON-continuous region                 */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] dst_info       : Information of destination array          */
/*               [IN] src_info       : Information of source array               */
/*               [OUT] *dst          : Pointer of destination array              */
/*               [IN] *dst_desc      : Descriptor of destination coarray         */
/*               [IN] *src_desc      : Descriptor of source coarray              */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [IN] src_dims       : Number of dimensions of source array      */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If dst is NOT a coarray, dst_desc is NULL.                       */
/* EXAMPLE    :                                                                  */
/*     a[0:100:2] = b[0:100:2]:[1]; // a[] is a dst, b[] is a src                */
/*********************************************************************************/
static void _fjrdma_NON_continuous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       void *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				       const int dst_dims, const int src_dims, const int transfer_elmts)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t elmt_size  = src_desc->elmt_size;
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;
  uint64_t raddrs[copy_elmts], laddrs[copy_elmts];

  if(dst_desc == NULL){
    size_t array_size = _get_array_size(dst_info, dst_dims);
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst, array_size) + dst_offset;
  }
  else{
    laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_offset;
  }

  // Set parameters for multipul FJMPI_Rdma_get()
  _XMP_set_coarray_addresses_with_chunk(raddrs, raddr, src_info, src_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(laddrs, laddr, dst_info, dst_dims, copy_chunk, copy_elmts);

  if(copy_elmts <= _XMP_FJRDMA_MAX_MGET){
    for(int i=0;i<copy_elmts;i++)
      _XMP_FJMPI_Rdma_get(target_rank, raddrs[i], laddrs[i], copy_chunk);
  }
  else{
    int times      = copy_elmts / _XMP_FJRDMA_MAX_MGET + 1;
    int rest_elmts = copy_elmts - _XMP_FJRDMA_MAX_MGET * (times - 1);

    for(int i=0;i<times;i++){
      size_t tmp_elmts = (i != times-1)? _XMP_FJRDMA_MAX_MGET : rest_elmts;
      for(int j=0;j<tmp_elmts;j++)
	_XMP_FJMPI_Rdma_get(target_rank, raddrs[j+i*_XMP_FJRDMA_MAX_MGET], laddrs[j+i*_XMP_FJRDMA_MAX_MGET], copy_chunk);
    }
  }
  _XMP_fjrdma_sync_memory_get();

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/*********************************************************************************/
/* DESCRIPTION : Execute scalar multiple get operation                           */
/* ARGUMENT    : [IN] target_rank    : Target rank                               */
/*               [IN] dst_offset     : Offset size of destination array          */
/*               [IN] src_offset     : Offset size of source array               */
/*               [IN] dst_info       : Information of destination array          */
/*               [IN] dst_dims       : Number of dimensions of destination array */
/*               [OUT] *dst_desc     : Descriptor of destination array           */
/*               [IN] *src_desc      : Descriptor of source array                */
/*               [IN] *dst           : Pointer of destination array              */
/*               [IN] transfer_elmts : Number of transfer elements               */
/* NOTE       : If dst is NOT a coarray, dst_desc != NULL                        */
/* EXAMPLE    :                                                                  */
/*     a[0:100]:[1] = b[0]; // a[] is a dst, b[] is a src                        */
/*********************************************************************************/
static void _fjrdma_scalar_mget(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
				const _XMP_array_section_t *dst_info, const int dst_dims,
				const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				char *dst, const size_t transfer_elmts)
{
  uint64_t raddr = (uint64_t)src_desc->addr[target_rank] + src_offset;
  uint64_t laddr;
  size_t elmt_size = dst_desc->elmt_size;

  if(dst_desc == NULL)
    laddr = FJMPI_Rdma_reg_mem(_XMP_TEMP_MEMID, dst + dst_offset, elmt_size);
  else
    laddr = (uint64_t)dst_desc->addr[_XMP_world_rank] + dst_offset;

  _XMP_FJMPI_Rdma_get(target_rank, raddr, laddr, elmt_size);
  _XMP_fjrdma_sync_memory_get();

  // Local copy (Note that number of copies is one time more in following _XMP_stride_memcpy_Xdim())
  char *src_addr = dst + dst_offset;
  char *dst_addr = src_addr;
  switch (dst_dims){
  case 1:
    _XMP_stride_memcpy_1dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  default:
    _XMP_fatal("Coarray Error ! Dimension is too big.\n");
    break;
  }

  if(dst_desc == NULL)
    FJMPI_Rdma_dereg_mem(_XMP_TEMP_MEMID);
}

/***************************************************************************************/
/* DESCRIPTION : Execute put operation                                                 */
/* ARGUMENT    : [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)      */
/*               [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE) */
/*               [IN] target_rank    : Target rank                                     */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] *src_info      : Information of source array                     */
/*               [IN] *dst_info      : Information of destination array                */
/*               [IN] *src_desc      : Descriptor of source array                      */
/*               [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *dst           : Pointer of destination array                    */
/*               [IN] src_elmts      : Number of elements of source array              */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/***************************************************************************************/
void _XMP_fjrdma_get(const int src_continuous, const int dst_continuous, const int target_rank, 
		     const int src_dims, const int dst_dims, 
		     const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		     const _XMP_coarray_t *src_desc, const _XMP_coarray_t *dst_desc, void *dst,
		     const int src_elmts, const int dst_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = src_desc->elmt_size * src_elmts;

  _check_transfer_size(transfer_size);

  if(src_elmts == dst_elmts){
    if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
      _fjrdma_continuous_get(target_rank, dst_offset, src_offset, dst, dst_desc, src_desc, transfer_size);
    }
    else{
      _fjrdma_NON_continuous_get(target_rank, dst_offset, src_offset, dst_info, src_info,
				 dst, dst_desc, src_desc, dst_dims, src_dims, src_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _fjrdma_scalar_mget(target_rank, dst_offset, src_offset, dst_info, dst_dims, dst_desc, src_desc, (char *)dst, dst_elmts);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
  _release_MRQ();
}

/**
 * Build table and Initialize for sync images
 */
void _XMP_fjrdma_build_sync_images_table()
{
  _sync_images_table = malloc(sizeof(unsigned int) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++)
    _sync_images_table[i] = _XMP_N_INT_FALSE;

  double *token     = _XMP_alloc(sizeof(double));
  _local_rdma_addr  = FJMPI_Rdma_reg_mem(_XMP_SYNC_IMAGES_ID, token, sizeof(double));
  _remote_rdma_addr = _XMP_alloc(sizeof(uint64_t) * _XMP_world_size);

  // Obtain remote RDMA addresses
  MPI_Barrier(MPI_COMM_WORLD);
  for(int ncount=0,i=1; i<_XMP_world_size+1; ncount++,i++){
    int partner_rank = (_XMP_world_rank + _XMP_world_size - i) % _XMP_world_size;
    if(partner_rank == _XMP_world_rank)
      _remote_rdma_addr[partner_rank] = _local_rdma_addr;
    else
      _remote_rdma_addr[partner_rank] = FJMPI_Rdma_get_remote_addr(partner_rank, _XMP_SYNC_IMAGES_ID);

    if(ncount > _XMP_INIT_RDMA_INTERVAL){
      MPI_Barrier(MPI_COMM_WORLD);
      ncount = 0;
    }
  }
}

/**
   Add rank to table
   *
   * @param[in]  rank rank number
   */
static void _add_sync_images_table(const int rank)
{
  _sync_images_table[rank] = _XMP_N_INT_TRUE;
}

/**
   Notify to nodes
   *
   * @param[in]  num        number of nodes
   * @param[in]  *rank_set  rank set
   */
static void _notify_sync_images(const int num, int *rank_set)
{
  int num_of_requests = 0;
  
  for(int i=0;i<num;i++)
    if(rank_set[i] == _XMP_world_rank){
      _add_sync_images_table(_XMP_world_rank);
    }
    else{
      FJMPI_Rdma_put(rank_set[i], _XMP_SYNC_IMAGES_TAG, _remote_rdma_addr[rank_set[i]], 
		     _local_rdma_addr, sizeof(double), _XMP_SYNC_IMAGES_FLAG_NIC);
      num_of_requests++;
    }

  struct FJMPI_Rdma_cq cq;
  for(int i=0;i<num_of_requests;i++)
    while(FJMPI_Rdma_poll_cq(_XMP_SYNC_IMAGES_SEND_NIC, &cq) != FJMPI_RDMA_NOTICE);  // Wait until finishing above put operations
}

/**
   Check to recieve all request from all node
   *
   * @param[in]  num        number of nodes
   * @param[in]  *rank_set  rank set
   * @praam[in]  old_wait_sync_images[num] old images set
*/
static _Bool _check_sync_images_table(const int num, int *rank_set)
{
  int checked = 0;

  for(int i=0;i<num;i++)
    if(_sync_images_table[rank_set[i]] == _XMP_N_INT_TRUE)
      checked++;

  if(checked == num) return true;
  else               return false;
}

/**
   Wait until recieving all request from all node
   *
   * @param[in]  num                       number of nodes
   * @param[in]  *rank_set                 rank set
   * @praam[in]  old_wait_sync_images[num] old images set
*/
static void _wait_sync_images(const int num, int *rank_set)
{
  struct FJMPI_Rdma_cq cq;

  while(1){
    if(_check_sync_images_table(num, rank_set)) break;

    if(FJMPI_Rdma_poll_cq(_XMP_SYNC_IMAGES_RECV_NIC, &cq) == FJMPI_RDMA_HALFWAY_NOTICE)
      _add_sync_images_table(cq.pid);
  }
}

/**
   Execute sync images
   *
   * @param[in]  num         number of nodes
   * @param[in]  *image_set  image set
   * @param[out] status      status
*/
void _XMP_fjrdma_sync_images(const int num, int* image_set, int* status)
{
  _XMP_fjrdma_sync_memory();

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
  for(int i=0;i<num;i++)
    _sync_images_table[rank_set[i]] = _XMP_N_INT_FALSE;
}
