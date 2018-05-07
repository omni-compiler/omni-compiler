#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "xmp.h"
#include "xmp_internal.h"
#include "xmp_math_function.h"

#define _XACC_NUM_COMM_CACHES 4
#define _XACC_MAX_NUM_SENDRECVS 4

typedef struct _XACC_sendrecv_comm_type{
  void *buf;
  int count;
  MPI_Datatype datatype;
  int rank;
  MPI_Comm comm;
}_XACC_sendrecv_comm_t;

typedef struct _XACC_gmv_comm_type{
  _XMP_gmv_desc_t *desc_left;
  _XMP_gmv_desc_t *desc_right;
  int num_recvs;
  int num_sends;
  _XACC_sendrecv_comm_t recvs[_XACC_MAX_NUM_SENDRECVS];
  _XACC_sendrecv_comm_t sends[_XACC_MAX_NUM_SENDRECVS];
}_XACC_gmv_comm_t;


static _XACC_gmv_comm_t *g_comm_cache[_XACC_NUM_COMM_CACHES];
static int g_comm_cache_tail = 0;
static _XACC_gmv_comm_t g_current_comm;


static int is_same_desc(_XMP_gmv_desc_t *l, _XMP_gmv_desc_t *r)
{
  if ( l->is_global != r->is_global ||
       l->ndims != r->ndims ||
       l->a_desc != r->a_desc ||
       l->local_data != r->local_data ||
       l->a_dev != r->a_dev){
    return _XMP_N_INT_FALSE;
  }

  int ndims = l->ndims;
  if(l->local_data != NULL){
    for(int i = 0; i < ndims; i++){
      if(l->a_lb[i] != r->a_lb[i] ||
	 l->a_ub[i] != r->a_ub[i]){
	return _XMP_N_INT_FALSE;
      }
    }
  }

  if(l->a_desc != NULL){
    for(int i = 0; i < ndims; i++){
      if(l->lb[i] != r->lb[i] ||
	 l->ub[i] != r->ub[i] ||
	 l->st[i] != r->st[i]){
	return _XMP_N_INT_FALSE;
      }
    }
  }
  return _XMP_N_INT_TRUE;
}

static _XMP_gmv_desc_t* alloc_gmv_desc(_XMP_gmv_desc_t *desc)
{
  _XMP_gmv_desc_t* new_desc = (_XMP_gmv_desc_t*)malloc(sizeof(_XMP_gmv_desc_t));
  new_desc->is_global = desc->is_global;
  new_desc->ndims = desc->ndims;
  new_desc->a_desc = desc->a_desc;
  new_desc->local_data = desc->local_data;
  new_desc->a_dev = desc->a_dev;

  int ndims = desc->ndims;
  if(desc->local_data != NULL){
    int *lb = new_desc->a_lb = (int*)malloc(sizeof(int)*ndims);
    int *ub = new_desc->a_ub = (int*)malloc(sizeof(int)*ndims);
    for(int i = 0; i < ndims; i++){
      lb[i] = desc->a_lb[i];
      ub[i] = desc->a_ub[i];
    }
  }
  if(desc->a_desc != NULL){
    int *lb = new_desc->lb = (int*)malloc(sizeof(int)*ndims);
    int *ub = new_desc->ub = (int*)malloc(sizeof(int)*ndims);
    int *st = new_desc->st = (int*)malloc(sizeof(int)*ndims);
    for(int i = 0; i < ndims; i++){
      lb[i] = desc->lb[i];
      ub[i] = desc->ub[i];
      st[i] = desc->st[i];
    }
  }
  return new_desc;
}

static void free_gmv_desc(_XMP_gmv_desc_t *desc)
{
  if(desc->local_data != NULL){
    free(desc->a_lb);
    free(desc->a_ub);
  }
  if(desc->a_desc != NULL){
    free(desc->lb);
    free(desc->ub);
    free(desc->st);
  }
  free(desc);
}

static void free_gmv_comm(_XACC_gmv_comm_t* comm)
{
  free_gmv_desc(comm->desc_left);
  free_gmv_desc(comm->desc_right);

  /*
  for(int i = comm->num_recvs - 1; i >= 0; i--){
    MPI_Request_free(comm->req_recvs + i);
  }
  for(int i = comm->num_sends - 1; i >= 0; i--){
    MPI_Request_free(comm->req_sends + i);
  }
  free(comm->req_recvs);
  free(comm->req_sends);
  */
  free(comm);
}


static void cache_comm(_XACC_gmv_comm_t *comm)
{
  //printf("add cache\n");
  if(g_comm_cache[g_comm_cache_tail] != NULL){
    free_gmv_comm(g_comm_cache[g_comm_cache_tail]);
  }
  g_comm_cache[g_comm_cache_tail] = comm;
  if(++g_comm_cache_tail == _XACC_NUM_COMM_CACHES){
    g_comm_cache_tail = 0;
  }
}

static _XACC_gmv_comm_t* find_gmv_comm(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp)
{
  for(int i = 0; i < _XACC_NUM_COMM_CACHES; i++){
    _XACC_gmv_comm_t* comm = g_comm_cache[i];
    if(comm == NULL) continue;

    if(is_same_desc(comm->desc_left, gmv_desc_leftp) &&
       is_same_desc(comm->desc_right, gmv_desc_rightp)){
      return comm;
    }
  }
  return NULL;
}

static void exec_gmv_comm(_XACC_gmv_comm_t * comm)
{
  if(comm == NULL) return;

  //printf("exec cached comm\n");
  MPI_Request req[_XACC_MAX_NUM_SENDRECVS];

  //printf("comm_num_sends=%d, recvs=%d\n", comm->num_sends, comm->num_recvs);

  for(int i = 0; i < comm->num_recvs; i++){
    MPI_Irecv(comm->recvs[i].buf, comm->recvs[i].count, comm->recvs[i].datatype, comm->recvs[i].rank, _XMP_N_MPI_TAG_GMOVE, comm->recvs[i].comm, req + i);
  }
  for(int i = 0; i < comm->num_sends; i++){
    MPI_Send(comm->sends[i].buf, comm->sends[i].count, comm->sends[i].datatype, comm->sends[i].rank, _XMP_N_MPI_TAG_GMOVE, comm->sends[i].comm);
  }
  MPI_Waitall(comm->num_recvs, req, MPI_STATUSES_IGNORE);
}

static void set_sendrecv_comm(_XACC_sendrecv_comm_t *sendrecv_comm, void *buf, int count, MPI_Datatype datatype, int rank, MPI_Comm comm)
{
  sendrecv_comm->buf = buf;
  sendrecv_comm->count = count;
  sendrecv_comm->datatype = datatype;
  sendrecv_comm->rank = rank;
  sendrecv_comm->comm = comm;
}


void
xmpc_gmv_g_alloc_acc(_XMP_gmv_desc_t **gmv_desc, _XMP_array_t *ap, void *dev_addr)
{
  _XMP_gmv_desc_t *gp;
  int n = ap->dim;

  gp = (_XMP_gmv_desc_t *)_XMP_alloc(sizeof(_XMP_gmv_desc_t));

  gp->kind = (int *)_XMP_alloc(sizeof(int) * n);
  gp->lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->ub = (int *)_XMP_alloc(sizeof(int) * n);
  gp->st = (int *)_XMP_alloc(sizeof(int) * n);
  
  if (!gp || !gp->kind || !gp->lb || !gp->st)
    _XMP_fatal("gmv_g_alloc: cannot alloc memory");

  gp->is_global = true;
  gp->ndims = n;
  gp->a_desc = ap;

  gp->local_data = NULL;
  gp->a_lb = NULL;
  gp->a_ub = NULL;

  gp->a_dev = dev_addr;

  *gmv_desc = gp;
}

void
xmpc_gmv_l_alloc_acc(_XMP_gmv_desc_t **gmv_desc, void *local_data, int n)
{
  _XMP_gmv_desc_t *gp;

  gp = (_XMP_gmv_desc_t *)_XMP_alloc(sizeof(_XMP_gmv_desc_t));

  gp->kind = (int *)_XMP_alloc(sizeof(int) * n);
  gp->lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->ub = (int *)_XMP_alloc(sizeof(int) * n);
  gp->st = (int *)_XMP_alloc(sizeof(int) * n);
  gp->a_lb = (int *)_XMP_alloc(sizeof(int) * n);
  gp->a_ub = (int *)_XMP_alloc(sizeof(int) * n);

  gp->is_global = false;
  gp->ndims = n;
  gp->a_desc = NULL;

  gp->local_data = NULL;

  gp->a_dev = local_data;

  *gmv_desc = gp;
}

static void _XMP_sendrecv_ARRAY_acc(int type, int type_size, MPI_Datatype *mpi_datatype,
				    _XMP_array_t *dst_array, void *dst_array_dev, int *dst_array_nodes_ref,
				    int *dst_lower, int *dst_upper, int *dst_stride, unsigned long long *dst_dim_acc,
				    _XMP_array_t *src_array, void *src_array_dev, int *src_array_nodes_ref,
				    int *src_lower, int *src_upper, int *src_stride, unsigned long long *src_dim_acc) {
  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  void *dst_addr = dst_array_dev; //dst_array->array_addr_p;
  void *src_addr = src_array_dev; //src_array->array_addr_p;
  int dst_dim = dst_array->dim;
  int src_dim = src_array->dim;

  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();
  _XMP_ASSERT(exec_nodes->is_member);

  int exec_rank = exec_nodes->comm_rank;
  MPI_Comm *exec_comm = exec_nodes->comm;

  // calc dst_ranks
  _XMP_nodes_ref_t *dst_ref = _XMP_create_nodes_ref_for_target_nodes(dst_array_nodes, dst_array_nodes_ref, exec_nodes);
  int dst_shrink_nodes_size = dst_ref->shrink_nodes_size;
  int dst_ranks[dst_shrink_nodes_size];
  if (dst_shrink_nodes_size == 1) {
    dst_ranks[0] = _XMP_calc_linear_rank(dst_ref->nodes, dst_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(dst_ref->nodes, dst_ranks, dst_ref->ref, dst_shrink_nodes_size);
  }

  unsigned long long total_elmts = 1;
  for (int i = 0; i < dst_dim; i++) {
    total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_lower[i], dst_upper[i], dst_stride[i]);
  }

  // calc src_ranks
  _XMP_nodes_ref_t *src_ref = _XMP_create_nodes_ref_for_target_nodes(src_array_nodes, src_array_nodes_ref, exec_nodes);
  int src_shrink_nodes_size = src_ref->shrink_nodes_size;
  int src_ranks[src_shrink_nodes_size];
  if (src_shrink_nodes_size == 1) {
    src_ranks[0] = _XMP_calc_linear_rank(src_ref->nodes, src_ref->ref);
  } else {
    _XMP_translate_nodes_rank_array_to_ranks(src_ref->nodes, src_ranks, src_ref->ref, src_shrink_nodes_size);
  }

  unsigned long long src_total_elmts = 1;
  for (int i = 0; i < src_dim; i++) {
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_lower[i], src_upper[i], src_stride[i]);
  }

  if (total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  }

  int even_send_flag = 0;
  int even_send_region_local_nums = -1;
  int even_send_region_local_idx = -1;
  if(src_dim == 1 && dst_array_nodes->comm_size % src_array_nodes->comm_size == 0){
    int src_size0 = src_upper[0] - src_lower[0] + 1;
    int even_send_region_idx = src_lower[0] / src_size0;
    even_send_region_local_nums = src_array->info->par_size / src_size0;
    even_send_region_local_idx = even_send_region_idx % even_send_region_local_nums;
    even_send_flag =1;
    //printf("p(%d) region_idx = %d, num_local_regions=%d, num_local_regions = %d, localidx = %d\n", exec_rank,even_send_region_idx, even_send_region_size, even_send_region_local_nums, even_send_region_local_idx);
  }

  // recv phase
  void *recv_buffer = NULL;
  void *recv_alloc = NULL;
  int wait_recv = _XMP_N_INT_FALSE;
  MPI_Request gmove_request;

  for (int i = 0; i < dst_shrink_nodes_size; i++) {
    if (dst_ranks[i] == exec_rank) {
      wait_recv = _XMP_N_INT_TRUE;

      int src_rank;
      if(even_send_flag){
	int src_i = even_send_region_local_idx + even_send_region_local_nums * i;
	src_rank = src_ranks[src_i];
	//	printf("p(%d): src_rank[%d]=%d\n", exec_rank, src_i, src_rank);
      }else{
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
        src_rank = src_ranks[i];
      } else {
        src_rank = src_ranks[i % src_shrink_nodes_size];
      }
      }

      for (int i = 0; i < dst_dim; i++) {
	_XMP_gtol_array_ref_triplet(dst_array, i, &(dst_lower[i]), &(dst_upper[i]), &(dst_stride[i]));
      }

      if(dst_dim == 1 && dst_stride[0] == 1){
	recv_buffer = (char*)dst_addr + type_size * dst_lower[0];
      }else{
	recv_alloc = _XMP_alloc(total_elmts * type_size);
	recv_buffer = recv_alloc;
      }
      MPI_Irecv(recv_buffer, total_elmts, *mpi_datatype, src_rank, _XMP_N_MPI_TAG_GMOVE, *exec_comm, &gmove_request);
      //      fprintf(stderr, "DEBUG: Proc(%d), Irecv(src=%d, total_elmnt=%llu)\n", exec_rank, src_rank, total_elmts);
#if 1
      //save comm start
      if(g_current_comm.num_recvs < _XACC_MAX_NUM_SENDRECVS){
	set_sendrecv_comm(&(g_current_comm.recvs[g_current_comm.num_recvs]),
			  recv_buffer, total_elmts * type_size, MPI_BYTE, src_rank, *exec_comm);
      }
      g_current_comm.num_recvs++;
      //save comm end
#endif
    }
  }

  // send phase
  for (int i = 0; i < src_shrink_nodes_size; i++) {
    if (src_ranks[i] == exec_rank) {
      void *send_buffer = NULL;
      void *send_alloc = NULL;
      for (int j = 0; j < src_dim; j++) {
        _XMP_gtol_array_ref_triplet(src_array, j, &(src_lower[j]), &(src_upper[j]), &(src_stride[j]));
      }
      if(src_dim == 1 && src_stride[0] == 1){
	send_buffer = (char*)src_addr + type_size * src_lower[0];
      }else{
	send_alloc = _XMP_alloc(total_elmts * type_size);
	send_buffer = send_alloc;
	(*_xmp_pack_array)(send_buffer, src_addr, type, type_size, src_dim, src_lower, src_upper, src_stride, src_dim_acc);
      }
      if ((dst_shrink_nodes_size == src_shrink_nodes_size) ||
          (dst_shrink_nodes_size <  src_shrink_nodes_size)) {
	if(even_send_flag){
	  int dst_i = i / even_send_region_local_nums;
	  if(i % even_send_region_local_nums == even_send_region_local_idx && dst_i < dst_shrink_nodes_size){
	    //	  fprintf(stderr, "DEBUG: Proc(%d), Send(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[dst_i], total_elmts);
	  MPI_Send(send_buffer, total_elmts, *mpi_datatype, dst_ranks[dst_i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
#if 1
	  //save comm start
	  if(g_current_comm.num_sends < _XACC_MAX_NUM_SENDRECVS){
	    set_sendrecv_comm(&(g_current_comm.sends[g_current_comm.num_sends]),
			      send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[dst_i], *exec_comm);
	  }
	  g_current_comm.num_sends++;
	  //save comm end
#endif
	  }
	}else{
        if (i < dst_shrink_nodes_size) {
          MPI_Send(send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[i], _XMP_N_MPI_TAG_GMOVE, *exec_comm);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Send(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[i], total_elmts);
#if 1
	  //save comm start
	  if(g_current_comm.num_sends < _XACC_MAX_NUM_SENDRECVS){
	    set_sendrecv_comm(&(g_current_comm.sends[g_current_comm.num_sends]),
			      send_buffer, total_elmts * type_size, MPI_BYTE, dst_ranks[i], *exec_comm);
	  }
	  g_current_comm.num_sends++;
	  //save comm end
#endif
        }
	}
      } else {
        int request_size = _XMP_M_COUNT_TRIPLETi(i, dst_shrink_nodes_size - 1, src_shrink_nodes_size);
        MPI_Request *requests = _XMP_alloc(sizeof(MPI_Request) * request_size);

        int request_count = 0;
        for (int j = i; j < dst_shrink_nodes_size; j += src_shrink_nodes_size) {
          MPI_Isend(send_buffer, total_elmts, *mpi_datatype, dst_ranks[j], _XMP_N_MPI_TAG_GMOVE, *exec_comm,
                    requests + request_count);
	  //	  fprintf(stderr, "DEBUG: Proc(%d), Isend(dst=%d, total_elmnt=%llu)\n", exec_rank, dst_ranks[j], total_elmts);
          request_count++;
        }

        MPI_Waitall(request_size, requests, MPI_STATUSES_IGNORE);
        _XMP_free(requests);
      }

      _XMP_free(send_alloc);
    }
  }

  // wait recv phase
  if (wait_recv) {
    MPI_Wait(&gmove_request, MPI_STATUS_IGNORE);
    if(! (dst_dim == 1 && dst_stride[0] == 1)){
      (*_xmp_unpack_array)(dst_addr, recv_buffer, type, type_size, dst_dim, dst_lower, dst_upper, dst_stride, dst_dim_acc);
    }
    _XMP_free(recv_alloc);
  }

  _XMP_finalize_nodes_ref(dst_ref);
  _XMP_finalize_nodes_ref(src_ref);
}


static void _XMP_gmove_garray_garray_block_cyclic_acc(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
						      int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
						      int *src_l, int *src_u, int *src_s, unsigned long long *src_d){

  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

  int type = dst_array->type;
  int type_size = dst_array->type_size;

  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;

  _XMP_nodes_t *dst_array_nodes = dst_array->array_nodes;
  int dst_array_nodes_dim = dst_array_nodes->dim;
  int dst_array_nodes_ref[dst_array_nodes_dim];
  for (int i = 0; i < dst_array_nodes_dim; i++) {
    dst_array_nodes_ref[i] = 0;
  }

  _XMP_nodes_t *src_array_nodes = src_array->array_nodes;
  int src_array_nodes_dim = src_array_nodes->dim;
  int src_array_nodes_ref[src_array_nodes_dim];

  int dst_lower[dst_dim], dst_upper[dst_dim], dst_stride[dst_dim];
  int src_lower[src_dim], src_upper[src_dim], src_stride[src_dim];

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  do {
    for (int i = 0; i < dst_dim; i++) {
      dst_lower[i] = dst_l[i]; dst_upper[i] = dst_u[i]; dst_stride[i] = dst_s[i];
    }

    for (int i = 0; i < src_dim; i++) {
      src_lower[i] = src_l[i]; src_upper[i] = src_u[i]; src_stride[i] = src_s[i];
    }

    if (_XMP_calc_global_index_BCAST(src_dim, src_lower, src_upper, src_stride,
				     dst_array, dst_array_nodes_ref, dst_lower, dst_upper, dst_stride)) {
      for (int i = 0; i < src_array_nodes_dim; i++) {
	src_array_nodes_ref[i] = 0;
      }

      int recv_lower[dst_dim], recv_upper[dst_dim], recv_stride[dst_dim];
      int send_lower[src_dim], send_upper[src_dim], send_stride[src_dim];

      do {
	for (int i = 0; i < dst_dim; i++) {
	  recv_lower[i] = dst_lower[i]; recv_upper[i] = dst_upper[i]; recv_stride[i] = dst_stride[i];
	}

	for (int i = 0; i < src_dim; i++) {
	  send_lower[i] = src_lower[i]; send_upper[i] = src_upper[i]; send_stride[i] = src_stride[i];
	}

	if (_XMP_calc_global_index_BCAST(dst_dim, recv_lower, recv_upper, recv_stride,
					 src_array, src_array_nodes_ref, send_lower, send_upper, send_stride)) {

	  _XMP_sendrecv_ARRAY_acc(type, type_size, &mpi_datatype,
				  dst_array, gmv_desc_leftp->a_dev/*dst_dev_addr*/, dst_array_nodes_ref,
				  recv_lower, recv_upper, recv_stride, dst_d,
				  src_array, gmv_desc_rightp->a_dev/*src_dev_addr*/, src_array_nodes_ref,
				  send_lower, send_upper, send_stride, src_d);
	}

      } while (_XMP_get_next_rank(src_array_nodes, src_array_nodes_ref));

    }

  } while (_XMP_get_next_rank(dst_array_nodes, dst_array_nodes_ref));

  MPI_Type_free(&mpi_datatype);
}

static _Bool is_whole(_XMP_gmv_desc_t *gmv_desc)
{
  _XMP_array_t *adesc = gmv_desc->a_desc;

  for (int i = 0; i < adesc->dim; i++){
    if (gmv_desc->lb[i] == 0 && gmv_desc->ub[i] == 0 && gmv_desc->st[i] == 0) continue;
    if (adesc->info[i].ser_lower != gmv_desc->lb[i] ||
	adesc->info[i].ser_upper != gmv_desc->ub[i] ||
	gmv_desc->st[i] != 1) return false;
  }

  return true;
}

static int _XMP_gmove_garray_garray_acc_opt(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
					int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
					int *src_l, int *src_u, int *src_s, unsigned long long *src_d)
{
  _XMP_ASSERT(gmv_desc_leftp->is_global);
  _XMP_ASSERT(gmv_desc_rightp->is_global);

  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

//  int type = dst_array->type;
//  int type_size = dst_array->type_size;

//  void *dst_addr = dst_array->array_addr_p;
//  void *src_addr = src_array->array_addr_p;

  int dst_dim = gmv_desc_leftp->ndims;
  int src_dim = gmv_desc_rightp->ndims;

  _XMP_template_t *dst_template = dst_array->align_template;
  _XMP_template_t *src_template = src_array->align_template;

  _XMP_nodes_t *dst_nodes = dst_template->onto_nodes;
  _XMP_nodes_t *src_nodes = src_template->onto_nodes;

  int dst_comm_size = dst_nodes->comm_size;
  int src_comm_size = src_nodes->comm_size;

  int exec_comm_size = _XMP_get_execution_nodes()->comm_size;

  if (exec_comm_size != dst_comm_size || exec_comm_size != src_comm_size) return 0;

  //
  // Next, do the general algorithm
  //
    
  // check if _XMP_sendrecv_ARRAY can be applied.

  for (int i = 0; i < dst_template->dim; i++){
    if (dst_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK &&
	dst_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC){
      return 0;
    }
  }

  for (int i = 0; i < src_template->dim; i++){
    if (src_template->chunk[i].dist_manner !=_XMP_N_DIST_BLOCK &&
	src_template->chunk[i].dist_manner !=_XMP_N_DIST_CYCLIC){
      return 0;
    }
  }

  if (dst_dim == dst_nodes->dim){
    if (src_dim == src_nodes->dim){
      for (int i = 0; i < dst_dim; i++){
	if (dst_array->info[i].align_subscript != 0 ||
	    (dst_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK &&
	     dst_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC)){
	  return 0;
	}
      }

      for (int i = 0; i < src_dim; i++){
	if (src_array->info[i].align_subscript != 0 ||
	    (src_array->info[i].align_manner !=_XMP_N_ALIGN_BLOCK &&
	     src_array->info[i].align_manner !=_XMP_N_ALIGN_CYCLIC)){
	  return 0;
	}
      }

    }
    else return 0;

    if (!is_whole(gmv_desc_leftp) || !is_whole(gmv_desc_rightp)) return 0;

  }
  else if (dst_dim < dst_nodes->dim){
    if (src_dim < src_nodes->dim){
      if (dst_nodes->dim != 2 || src_nodes->dim != 2 ||
	  dst_array->info[0].align_subscript != 0 ||
	  src_array->info[0].align_subscript != 0){
	return 0;
      }
    }
    else return 0;
  }
  else if (dst_dim > dst_nodes->dim){
    if (src_dim > src_nodes->dim){
      if (_XMPF_running == 1 && _XMPC_running == 0) return 0;
    }
    else return 0;
  }

  _XMP_gmove_garray_garray_block_cyclic_acc(gmv_desc_leftp, gmv_desc_rightp,
					    dst_l, dst_u, dst_s, dst_d,
					    src_l, src_u, src_s, src_d);

  return 1;

}

void _XMP_gmove_array_array_common_acc(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
				   int *dst_l, int *dst_u, int *dst_s, unsigned long long *dst_d,
				   int *src_l, int *src_u, int *src_s, unsigned long long *src_d,
				   int mode){
  // NOTE: asynchronous gmove always done by _XMP_gmove_1to1
  if (!xmp_is_async() && gmv_desc_leftp->is_global && gmv_desc_rightp->is_global && mode == _XMP_N_GMOVE_NORMAL){
    if (_XMP_gmove_garray_garray_acc_opt(gmv_desc_leftp, gmv_desc_rightp,
					 dst_l, dst_u, dst_s, dst_d,
					 src_l, src_u, src_s, src_d)) return;
    // fall through
  }

//  _XMP_gmove_1to1(gmv_desc_leftp, gmv_desc_rightp, mode);
  _XMP_fatal("_XMP_gmove_1to1 for acc is not implemented");

  return;

}


//
// ga(:) = gb(:)
//
void
_XMP_gmove_garray_garray_acc(_XMP_gmv_desc_t *gmv_desc_leftp,
			 _XMP_gmv_desc_t *gmv_desc_rightp,
			 int mode)
{
  _XMP_array_t *dst_array = gmv_desc_leftp->a_desc;
  _XMP_array_t *src_array = gmv_desc_rightp->a_desc;

  _XMP_ASSERT(src_array->type == type);
  _XMP_ASSERT(src_array->type_size == dst_array->type_size);

  _XACC_gmv_comm_t *gmv_comm = find_gmv_comm(gmv_desc_leftp, gmv_desc_rightp);
  if(gmv_comm != NULL){
    //do cached comm
    exec_gmv_comm(gmv_comm);
    return;
  }

  g_current_comm.num_recvs = 0;
  g_current_comm.num_sends = 0;

  //unsigned long long gmove_total_elmts = 0;

  // get dst info
  unsigned long long dst_total_elmts = 1;
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  int dst_scalar_flag = 1;
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = gmv_desc_leftp->lb[i];
    dst_u[i] = gmv_desc_leftp->ub[i];
    dst_s[i] = gmv_desc_leftp->st[i];
    dst_d[i] = dst_array->info[i].dim_acc;
    _XMP_normalize_array_section(gmv_desc_leftp, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    if (dst_s[i] != 0) dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
    dst_scalar_flag &= (dst_s[i] == 0);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  int src_dim = src_array->dim;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim];
  unsigned long long src_d[src_dim];
  int src_scalar_flag = 1;
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = gmv_desc_rightp->lb[i];
    src_u[i] = gmv_desc_rightp->ub[i];
    src_s[i] = gmv_desc_rightp->st[i];
    src_d[i] = src_array->info[i].dim_acc;
    _XMP_normalize_array_section(gmv_desc_rightp, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    if (src_s[i] != 0) src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
    src_scalar_flag &= (src_s[i] == 0);
  }

  if (dst_total_elmts != src_total_elmts && !src_scalar_flag){
    _XMP_fatal("wrong assign statement for gmove");
  }
  else {
    //gmove_total_elmts = dst_total_elmts;
  }

  if (mode == _XMP_N_GMOVE_NORMAL){

    if (dst_scalar_flag && src_scalar_flag){
//      void *dst_addr = (char *)dst_array->array_addr_p + _XMP_gtol_calc_offset(dst_array, dst_l);
//      void *src_addr = (char *)src_array->array_addr_p + _XMP_gtol_calc_offset(src_array, src_l);
//      _XMP_gmove_SENDRECV_GSCALAR(dst_addr, src_addr,
//				  dst_array, src_array,
//				  dst_l, src_l);
      _XMP_fatal("_XMP_gmove_SENDRECV_GSCALAR() for acc is not implemented");
      return;
    }
    else if (!dst_scalar_flag && src_scalar_flag){
//      char *tmp = _XMP_alloc(src_array->type_size);
//      _XMP_gmove_BCAST_GSCALAR(tmp, src_array, src_l);
//      _XMP_gmove_gsection_scalar(dst_array, dst_l, dst_u, dst_s, tmp);
//      _XMP_free(tmp);
      _XMP_fatal("_XMP_gmove_BCAST_GSCALAR() and _XMP_gmove_gsection_scalar() for acc is not implemented");
      return;
    }
  }

  _XMP_gmove_array_array_common_acc(gmv_desc_leftp, gmv_desc_rightp,
				    dst_l, dst_u, dst_s, dst_d,
				    src_l, src_u, src_s, src_d,
				    mode);

//cache comm
  if(g_current_comm.num_recvs <= _XACC_MAX_NUM_SENDRECVS && g_current_comm.num_sends <= _XACC_MAX_NUM_SENDRECVS){
//    printf("cache comm\n");
    gmv_comm = (_XACC_gmv_comm_t*)malloc(sizeof(_XACC_gmv_comm_t));
    gmv_comm->desc_left = alloc_gmv_desc(gmv_desc_leftp);
    gmv_comm->desc_right = alloc_gmv_desc(gmv_desc_rightp);
    gmv_comm->num_recvs = g_current_comm.num_recvs;
    gmv_comm->num_sends = g_current_comm.num_sends;

    for(int i = 0; i < g_current_comm.num_recvs; i++){
      gmv_comm->recvs[i] = g_current_comm.recvs[i];
    }
    for(int i = 0; i < g_current_comm.num_sends; i++){
      gmv_comm->sends[i] = g_current_comm.sends[i];
    }
    cache_comm(gmv_comm);
  }
}

void
xmpc_gmv_do_acc(_XMP_gmv_desc_t *gmv_desc_leftp, _XMP_gmv_desc_t *gmv_desc_rightp,
		int mode)
{
  if (gmv_desc_leftp->is_global && gmv_desc_rightp->is_global){
    _XMP_gmove_garray_garray_acc(gmv_desc_leftp, gmv_desc_rightp, mode);
  }
  else {
    _XMP_fatal("gmv_do_acc: currently both sides must be global.");
  }
}
