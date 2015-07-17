#include "xmp_internal.h"
#include "tca-api.h"
#include "xmp.h"
#define _XMP_TCA_DMAC 0

static void _XMP_create_TCA_handle(void *acc_addr, _XMP_array_t *adesc)
{
  if(adesc->set_handle)
    return;

  size_t size = (size_t)(adesc->type_size * adesc->total_elmts);

  _XACC_DEBUG("[%d] tcaCreateHandle size = %zd addr=%p\n", _XMP_world_rank, size, acc_addr);
  tcaHandle tmp_handle;
  TCA_CHECK(tcaCreateHandle(&tmp_handle, acc_addr, size, tcaMemoryGPU));

  adesc->tca_handle = _XMP_alloc(sizeof(tcaHandle) * _XMP_world_size);
  MPI_Allgather(&tmp_handle, sizeof(tcaHandle), MPI_BYTE,
                adesc->tca_handle, sizeof(tcaHandle), MPI_BYTE, MPI_COMM_WORLD);

  adesc->tca_reflect_desc = tcaDescNew();

  adesc->set_handle = _XMP_N_INT_TRUE;
}

static void _XMP_reflect_acc_sched_dim(_XMP_array_t *adesc, void *acc_addr, int dim, int is_periodic)
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

  if (_XMPF_running && !_XMPC_running){ /* for XMP/F */
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
  else if(!_XMPF_running && _XMPC_running){ /* for XMP/C */
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
      long long lb_send, lb_recv, dim_acc;

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
      long long lb_send, lb_recv, dim_acc;
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
  reflect->lo_send_array = (char*)acc_addr + lo_src_offset;
  reflect->lo_recv_array = (char*)acc_addr + hi_dst_offset;
  reflect->hi_send_array = (char*)acc_addr + hi_src_offset;
  reflect->hi_recv_array = (char*)acc_addr + lo_dst_offset;
}

static void _XMP_create_TCA_reflect_desc(void *acc_addr, _XMP_array_t *adesc)
{
  int array_dim = adesc->dim;
  
  for(int i=0;i<array_dim;i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;
    int is_periodic = _XMP_N_INT_FALSE; // FIX me
    _XMP_reflect_acc_sched_dim(adesc, acc_addr, i, is_periodic);
  }

  tcaHandle* h = (tcaHandle*)adesc->tca_handle;
  tcaDesc* tca_reflect_desc = (tcaDesc*)adesc->tca_reflect_desc;
  for(int i=0;i<array_dim;i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)  continue;

    int lo_rank = ai->reflect_acc_sched->lo_rank;
    int hi_rank = ai->reflect_acc_sched->hi_rank;
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int count = reflect->count;
    off_t lo_dst_offset = reflect->lo_dst_offset;
    off_t lo_src_offset = reflect->lo_src_offset;
    off_t hi_dst_offset = reflect->hi_dst_offset;
    off_t hi_src_offset = reflect->hi_src_offset;
    size_t width  = reflect->blocklength;
    int wait_slot = adesc->wait_slot;
    int wait_tag = adesc->wait_tag;
    static int dma_flag = tcaDMAUseInternal|tcaDMAUseNotifyInternal|tcaDMANotify;
    int dim_index;

    xmp_nodes_index(adesc->array_nodes, i+1, &dim_index);

    if(count == 1){
      int target_lo_rank, target_hi_rank;

      if (lo_rank != -1)
	target_lo_rank = lo_rank;
      else
	target_lo_rank = MPI_PROC_NULL;

      MPI_Recv_init(reflect->lo_recv_array, width, MPI_BYTE, target_lo_rank,
		    0, MPI_COMM_WORLD, &reflect->req[0]);
      MPI_Send_init(reflect->lo_send_array, width, MPI_BYTE, target_lo_rank,
		    1, MPI_COMM_WORLD, &reflect->req[1]);
      lo_src_offset += reflect->stride;
      lo_dst_offset += reflect->stride;

      if(hi_rank != -1)
	target_hi_rank = hi_rank;
      else
	target_hi_rank = MPI_PROC_NULL;

      MPI_Recv_init(reflect->hi_recv_array,  width, MPI_BYTE, target_hi_rank,
		    1, MPI_COMM_WORLD, &reflect->req[2]);
      MPI_Send_init(reflect->hi_send_array,  width, MPI_BYTE, target_hi_rank,
		    0, MPI_COMM_WORLD, &reflect->req[3]);
      hi_src_offset += reflect->stride;
      hi_dst_offset += reflect->stride;
    }
    else if(count > 1){
      size_t pitch  = reflect->stride;
      if (dim_index % 2 == 0) {
	if(lo_rank != -1){
	  TCA_CHECK(tcaDescSetMemcpy2D(tca_reflect_desc, &h[lo_rank], lo_dst_offset, pitch,
				       &h[_XMP_world_rank], lo_src_offset, pitch,
				       width, count, dma_flag, wait_slot, wait_tag));
	}
	if(hi_rank != -1){
	  TCA_CHECK(tcaDescSetMemcpy2D(tca_reflect_desc, &h[hi_rank], hi_dst_offset, pitch,
				       &h[_XMP_world_rank], hi_src_offset, pitch,
				       width, count, dma_flag, wait_slot, wait_tag));
	}
      } else {
	if(hi_rank != -1){
	  TCA_CHECK(tcaDescSetMemcpy2D(tca_reflect_desc, &h[hi_rank], hi_dst_offset, pitch,
				       &h[_XMP_world_rank], hi_src_offset, pitch,
				       width, count, dma_flag, wait_slot, wait_tag));
	}
	if(lo_rank != -1){
	  TCA_CHECK(tcaDescSetMemcpy2D(tca_reflect_desc, &h[lo_rank], lo_dst_offset, pitch,
				       &h[_XMP_world_rank], lo_src_offset, pitch,
				       width, count, dma_flag, wait_slot, wait_tag));
	}
      }
    }
  }

  TCA_CHECK(tcaDescSet(tca_reflect_desc, _XMP_TCA_DMAC));
}

void _XMP_reflect_init_hybrid(void *acc_addr, _XMP_array_t *adesc)
{
  _XMP_create_TCA_handle(acc_addr, adesc);
  _XMP_create_TCA_reflect_desc(acc_addr, adesc);
}

static void _XMP_refect_wait_tca(_XMP_array_t *adesc)
{
  int array_dim = adesc->dim;
  tcaHandle* h = (tcaHandle*)adesc->tca_handle;
  int wait_slot = adesc->wait_slot;
  int wait_tag = adesc->wait_tag;
  
  for(int i=0;i<array_dim;i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;
    int lo_rank = ai->reflect_acc_sched->lo_rank;
    int hi_rank = ai->reflect_acc_sched->hi_rank;

    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int count = reflect->count;

    if (count > 1){
      if(lo_rank != -1) {
	_XACC_DEBUG("[%d] lo wait from %d\n", _XMP_world_rank, lo_rank);
	TCA_CHECK(tcaWaitDMARecvDesc(&h[lo_rank], wait_slot, wait_tag));
      }

      if(hi_rank != -1) {
	_XACC_DEBUG("[%d] hi wait from %d\n", _XMP_world_rank, hi_rank);
	TCA_CHECK(tcaWaitDMARecvDesc(&h[hi_rank], wait_slot, wait_tag));
      }
    }
  }
}

static void _XMP_reflect_start_mpi(_XMP_array_t *adesc)
{
  for (int i = 0; i < adesc->dim; i++) {
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;

    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int count = reflect->count;

    if (count == 1)
      MPI_Startall(4, reflect->req);
  }
}

static void _XMP_reflect_wait_mpi(_XMP_array_t *adesc)
{
  for (int i = 0; i < adesc->dim; i++) {
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if (ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;

    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int count = reflect->count;

    if (count == 1)
      MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
  }
}

void _XMP_reflect_do_hybrid(_XMP_array_t *adesc)
{
  _XMP_reflect_start_mpi(adesc);
  TCA_CHECK(tcaStartDMADesc(_XMP_TCA_DMAC));
  _XMP_reflect_wait_mpi(adesc);
  _XMP_refect_wait_tca(adesc);
  MPI_Barrier(MPI_COMM_WORLD);
}
