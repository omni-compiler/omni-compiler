#include <stdio.h>
#include "xmp_internal.h"
#ifdef _XMP_TCA
int _dma_slot = 0, _flag = 0, _wait_slot = 0, _wait_tag = 0;
off_t dst_offset = 0, src_offset = 0;
#define TCA_DMAC 0
#define TCA_CHECK(tca_call) do { \
  int status = tca_call;         \
  if(status != TCA_SUCCESS) {    \
  if(status == TCA_ERROR_INVALID_VALUE) {                 \
  fprintf(stderr,"(TCA) error TCA API, INVALID_VALUE\n"); \
  exit(-1);                                               \
  }else if(status == TCA_ERROR_OUT_OF_MEMORY){            \
  fprintf(stderr,"(TCA) error TCA API, OUT_OF_MEMORY\n"); \
  exit(-1);                                               \
  }else if(status == TCA_ERROR_NOT_SUPPORTED){            \
  fprintf(stderr,"(TCA) error TCA API, NOT_SUPPORTED\n"); \
  exit(-1);                                               \
  }else{                                                  \
  fprintf(stderr,"(TCA) error TCA API, UNKWON\n");        \
  exit(-1); \
  }         \
  }         \
  }while (0)
#endif

void _XMP_reflect_init_acc(void *acc_addr, _XMP_array_t *array_desc)
{
#ifdef _XMP_TCA
  if(array_desc->has_handle)
    return;

  int dim = array_desc->dim;
  int *lo = malloc(sizeof(int) * dim);
  int *hi = malloc(sizeof(int) * dim);
  
  for(int i=0;i<dim;i++){
    lo[i] = array_desc->info[i].shadow_size_lo;
    hi[i] = array_desc->info[i].shadow_size_hi;
  }

  size_t size = (size_t)(array_desc->type_size * array_desc->total_elmts); // local array size

  TCA_CHECK(tcaCreateHandle(array_desc->tca_src_handle, acc_addr, size, tcaMemoryGPU));
  tcaCreateHandle(array_desc->tca_src_handle, acc_addr, size, tcaMemoryGPU);
  
  MPI_Allgather(array_desc->tca_src_handle, sizeof(tcaHandle), MPI_BYTE,
		array_desc->tca_dst_handle, sizeof(tcaHandle), MPI_BYTE, MPI_COMM_WORLD);

  array_desc->has_handle = _XMP_N_INT_TRUE;
  
  int num_neighbor_nodes = (int)pow(3.0, (double)dim) - 1; // 1dim. -> 2, 2dims. -> 8, 3dims. -> 26

  /*
  if(int i=0;i<num_neighbor_nodes;i++){
    TCA_CHECK(tcaSetDMADescInt_Memcpy(_dma_slot, &_dma_slot, array_desc->tca_dst_handle[target], dst_offset,
				      array_desc->tca_src_handle, src_offset, size[target],
				      _flag, _wait_slot, _wait_tag));
  }
  tcaSetDMAChainInt(TCA_DMAC, _dma_slot);
  */
  free(lo);
  free(hi);
#endif
}

void _XMP_reflect_do_acc(void *addr)
{
#ifdef _XMP_TCA
#endif
}

void _XMP_reflect_acc(void *addr)
{
#ifdef _XMP_TCA
#endif
}
