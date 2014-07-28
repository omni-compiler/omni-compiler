#include "xmp_internal.h"
#include "tca-api.h"
#define _XMP_TCA_DMAC 0
#define _XMP_TCA_CHAIN_FLAG (tcaDMANotify|tcaDMAContinue)
#define _XMP_TCA_LAST_FLAG  (tcaDMANotify)
#define _DBG

void _XMP_create_TCA_handle(void *acc_addr, _XMP_array_t *adesc)
{
  if(adesc->set_handle)
    return;

  size_t size = (size_t)(adesc->type_size * adesc->total_elmts); // local array size

  tcaHandle tmp_handle;
  TCA_CHECK(tcaCreateHandle(&tmp_handle, acc_addr, size, tcaMemoryGPU));
#ifdef _DBG
  printf("[%d] tcaCreateHandle %d\n", _XMP_world_rank, size);
#endif

  adesc->tca_handle = _XMP_alloc(sizeof(tcaHandle) * _XMP_world_size);
  MPI_Allgather(&tmp_handle, sizeof(tcaHandle), MPI_BYTE,
                adesc->tca_handle, sizeof(tcaHandle), MPI_BYTE, MPI_COMM_WORLD);

  adesc->set_handle = _XMP_N_INT_TRUE;
}

static void _XMP_reflect_acc_sched_dim(_XMP_array_t *adesc, int dim, int is_periodic)
{
  _XMP_array_info_t *ai = &(adesc->info[dim]);
  _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
  _XMP_array_info_t *ainfo = adesc->info;
  int target_tdim = ai->align_template_index;
  _XMP_nodes_info_t *ni = adesc->align_template->chunk[target_tdim].onto_nodes_info;
  int lwidth = ai->shadow_size_lo;
  int uwidth = ai->shadow_size_hi;

  int my_pos = ni->rank;
  int lb_pos = _XMP_get_owner_pos(adesc, dim, ai->ser_lower);
  int ub_pos = _XMP_get_owner_pos(adesc, dim, ai->ser_upper);
  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;
  int my_rank = adesc->align_template->onto_nodes->comm_rank;
  int lo_rank = my_rank + (lo_pos - my_pos) * ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * ni->multiplier;

  int count, blocklength;
  int type_size = adesc->type_size;
  int ndims = adesc->dim;
  long long stride;

  if (_XMPF_running & !_XMPC_running){ /* for XMP/F */
    count = 1;
    blocklength = type_size;
    stride = ainfo[0].alloc_size * type_size;

    for(int i=ndims-2; i>=dim; i--)
      count *= ainfo[i+1].alloc_size;

    for(int i=1; i<=dim; i++){
      blocklength *= ainfo[i-1].alloc_size;
      stride *= ainfo[i].alloc_size;
    }
  }
  else if(!_XMPF_running & _XMPC_running){ /* for XMP/C */
    count = 1;
    blocklength = type_size;
    stride = ainfo[ndims-1].alloc_size * type_size;

    for(int i=1; i <= dim; i++)
      count *= ainfo[i-1].alloc_size;

    for(int i=ndims-2; i >= dim; i--){
      blocklength *= ainfo[i+1].alloc_size;
      stride *= ainfo[i].alloc_size;
    }
  }
  else{
    _XMP_fatal("cannot determin the base language.");
  }

  if (!is_periodic && my_pos == lb_pos) // no periodic
    lo_rank = -1;

  if (!is_periodic && my_pos == ub_pos) // no periodic
    hi_rank = -1;

  // Calulate offset
  off_t hi_src_offset = 0, hi_dst_offset = 0;
  if(lwidth){
    for(int i=0; i<ndims; i++) {
      int lb_send, lb_recv, dim_acc;

      if (i == dim){
        lb_send = ainfo[i].local_upper - lwidth + 1;
        lb_recv = ainfo[i].shadow_size_lo - lwidth;
      }
      else {
        // Note: including shadow area
        lb_send = 0;
        lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;
      hi_src_offset += lb_send * dim_acc * type_size;
      hi_dst_offset += lb_recv * dim_acc * type_size;
    }
  }

  off_t lo_src_offset = 0, lo_dst_offset = 0;
  if(uwidth){
    for(int i=0; i<ndims; i++){
      int lb_send, lb_recv, dim_acc;
      if (i == dim) {
        lb_send = ainfo[i].local_lower;
        lb_recv = ainfo[i].local_upper + 1;
      }
      else {
        // Note: including shadow area
        lb_send = 0;
        lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;
      lo_src_offset += lb_send * dim_acc * type_size;
      lo_dst_offset += lb_recv * dim_acc * type_size;
    }
  }

  reflect->count = count;
  reflect->blocklength = blocklength;
  reflect->stride = stride;
  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;
  reflect->lo_src_offset = lo_src_offset;
  reflect->lo_dst_offset = lo_dst_offset;
  reflect->hi_src_offset = hi_src_offset;
  reflect->hi_dst_offset = hi_dst_offset;
}

static void create_TCA_desc_intraMEM(_XMP_array_t *adesc)
{
  static int dma_slot = 0;
  adesc->dma_slot = dma_slot;
  int array_dim = adesc->dim;
  
  for(int i=0;i<array_dim;i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    int is_periodic = _XMP_N_INT_FALSE; // FIX me
    _XMP_reflect_acc_sched_dim(adesc, i, is_periodic);
  }

  int num_of_neighbors = 0;
  for(int i=0;i<array_dim;i++){
    int lo_rank = adesc->info[i].reflect_acc_sched->lo_rank;
    int hi_rank = adesc->info[i].reflect_acc_sched->hi_rank;
    if(lo_rank != -1)
      num_of_neighbors++;
    if(hi_rank != -1)
      num_of_neighbors++;
  }

  int j = 0;
  tcaHandle* h = (tcaHandle*)adesc->tca_handle;
  for(int i=0;i<array_dim;i++){
    int lo_rank = adesc->info[i].reflect_acc_sched->lo_rank;
    int hi_rank = adesc->info[i].reflect_acc_sched->hi_rank;
    _XMP_array_info_t *ai = &(adesc->info[i]);
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int count = reflect->count;
    int flag;
    off_t lo_dst_offset = reflect->lo_dst_offset;
    off_t lo_src_offset = reflect->lo_src_offset;
    off_t hi_dst_offset = reflect->hi_dst_offset;
    off_t hi_src_offset = reflect->hi_src_offset;

    for(int c=0;c<count;c++){
      if(lo_rank != -1){
	if(j != num_of_neighbors-1)
	  flag = _XMP_TCA_CHAIN_FLAG;
	else
	  flag = _XMP_TCA_LAST_FLAG;

	tcaSetDMADescInt_Memcpy(dma_slot, &dma_slot, &h[lo_rank], lo_dst_offset, 
				&h[_XMP_world_rank], lo_src_offset, reflect->blocklength, 
				tcaDMANotify, adesc->wait_slot, adesc->wait_tag);

#ifdef _DBG
	printf("[%d] lo_rank = %d lo_src_offset=%d lo_dst_offset=%d reflect->blocklength = %d\n", 
	       _XMP_world_rank, lo_rank, lo_src_offset, lo_dst_offset, reflect->blocklength);
#endif
	lo_src_offset += reflect->stride;
	lo_dst_offset += reflect->stride;
	j++;
      }
      if(hi_rank != -1){
	if(j != num_of_neighbors-1)
	  flag = _XMP_TCA_CHAIN_FLAG;
	else
	  flag = _XMP_TCA_LAST_FLAG;
	
	tcaSetDMADescInt_Memcpy(dma_slot, &dma_slot, &h[hi_rank], hi_dst_offset,
				&h[_XMP_world_rank], hi_src_offset, reflect->blocklength,
				tcaDMANotify, adesc->wait_slot, adesc->wait_tag);
#ifdef _DBG
	printf("[%d] hi_rank = %d hi_src_offset=%d hi_dst_offset=%d reflect->blocklength = %d\n",
               _XMP_world_rank, hi_rank, hi_src_offset, hi_dst_offset, reflect->blocklength);
#endif
	hi_src_offset += reflect->stride;
	hi_dst_offset += reflect->stride;
	j++;
      }
    } // for c
  }

  tcaSetDMAChainInt(_XMP_TCA_DMAC, adesc->dma_slot);
#ifdef _DBG
  printf("[%d] DMA SLOT = %d\n", _XMP_world_rank, dma_slot);
#endif
}

void _XMP_create_TCA_desc(_XMP_array_t *adesc)
{
  //  if(! full_of_intraMEM)
  create_TCA_desc_intraMEM(adesc);
  //  else
  //    create_TCA_desc_hostMEM(array_desc);
}

void _XMP_init_tca()
{
  if(_XMP_world_size > 16)
    _XMP_fatal("TCA reflect has been not implemented in 16 more than nodes.");
  tcaInit();
  tcaDMADescInt_Init(); // Initialize Descriptor (Internal Memory) Mode
}

void _XMP_alloc_tca(_XMP_array_t *adesc)
{
  adesc->set_handle = _XMP_N_INT_FALSE;
  int array_dim = adesc->dim;
  for(int i=0;i<array_dim;i++)
    adesc->info[i].reflect_acc_sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));

  adesc->wait_slot = 0;  // No change ?
  adesc->wait_tag  = 0;  // No change ?
}

void _XMP_reflect_do_tca(_XMP_array_t *adesc){
  tcaStartDMADesc(_XMP_TCA_DMAC);
  
  int array_dim = adesc->dim;
  tcaHandle* h = (tcaHandle*)adesc->tca_handle;
  for(int i=0;i<array_dim;i++){
    int lo_rank = adesc->info[i].reflect_acc_sched->lo_rank;
    int hi_rank = adesc->info[i].reflect_acc_sched->hi_rank;

#ifdef _DBG
    printf("[%d] lo wait from %d\n", _XMP_world_rank, lo_rank);
    printf("[%d] hi wait from %d\n", _XMP_world_rank, hi_rank);
#endif

    if(lo_rank != -1)
      tcaWaitDMARecvDesc(&h[lo_rank], adesc->wait_slot, adesc->wait_tag);

    if(hi_rank != -1)
      tcaWaitDMARecvDesc(&h[hi_rank], adesc->wait_slot, adesc->wait_tag);
  }
#ifdef _DBG
  printf("[%d] wait finish\n", _XMP_world_rank);
#endif
}
