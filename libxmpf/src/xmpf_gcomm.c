#include "xmpf_internal.h"

//#define DBG 1

#ifdef DBG
double t0;
double t_mem = 0;
double t_copy = 0;
double t_comm = 0;
#endif

//
// reflect
//

int _XMPF_get_owner_pos_BLOCK(_XMP_array_t *a, int dim, int index){

  _XMP_ASSERT(a->info[dim].align_manner == _XMP_N_ALIGN_BLOCK);

  int align_offset = a->info[dim].align_subscript;

  int tdim = a->info[dim].align_template_index;
  int tlb = a->align_template->info[tdim].ser_lower;
  int chunk = a->align_template->chunk[tdim].par_chunk_width;

  int pos = (index + align_offset - tlb) / chunk;

  return pos;
}


void _XMPF_optimized_reflect_dim(_XMP_array_t *adesc, int target_dim, _Bool is_periodic){

  _XMP_RETURN_IF_SINGLE;

  if (!adesc->is_allocated) return;

  _XMP_array_info_t *ai = &(adesc->info[target_dim]);
  _XMP_array_info_t *ainfo = adesc->info;
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int target_tdim = ai->align_template_index;
  _XMP_nodes_info_t *ni = adesc->align_template->chunk[target_tdim].onto_nodes_info;

  int ndims = adesc->dim;

  // 0-origin
  int my_pos = ni->rank;
  int lb_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_lower);
  int ub_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_upper);

  if (!is_periodic && lb_pos == ub_pos) {
    return;
  }

  MPI_Comm *comm = adesc->align_template->onto_nodes->comm;
  int my_rank = adesc->align_template->onto_nodes->comm_rank;

  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;

  int lo_rank = my_rank + (lo_pos - my_pos) * ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * ni->multiplier;

  int type_size = adesc->type_size;
  void *array_addr = adesc->array_addr_p;

  void *lo_recv_buf, *lo_send_buf;
  void *hi_recv_buf, *hi_send_buf;

  //
  // setup MPI_data_type
  //

  int count = 1;
  int blocklength = type_size;
  int stride = ainfo[0].alloc_size * type_size;

  for (int i = ndims - 2; i >= target_dim; i--){
    count *= ainfo[i+1].alloc_size;
  }

  for (int i = 1; i <= target_dim; i++){
    blocklength *= ainfo[i-1].alloc_size;
    stride *= ainfo[i].alloc_size;
  }

  // for lower shadow

  if (ai->shadow_size_lo){
    MPI_Type_vector(count, blocklength * ai->shadow_size_lo, stride,
		    MPI_BYTE, &ai->mpi_datatype_shadow_lo);
#ifdef DBG
    xmpf_dbg_printf("(%d, lower) count: %d, blocklength: %d, stride: %d\n",
		    target_dim, count, blocklength * ai->shadow_size_lo, stride);
#endif

    MPI_Type_commit(&ai->mpi_datatype_shadow_lo);
  }

  // for upper shadow

  if (ai->shadow_size_hi){
    MPI_Type_vector(count, blocklength * ai->shadow_size_hi, stride,
		    MPI_BYTE, &ai->mpi_datatype_shadow_hi);
#ifdef DBG
    xmpf_dbg_printf("(%d, upper) count: %d, blocklength: %d, stride: %d\n",
		    target_dim, count, blocklength * ai->shadow_size_lo, stride);
#endif

    MPI_Type_commit(&ai->mpi_datatype_shadow_hi);
  }

  //
  // calculate base address
  //

  // for lower shadow

  if (ai->shadow_size_lo){
      
    lo_send_buf = array_addr;
    lo_recv_buf = array_addr;

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_upper - ainfo[i].shadow_size_lo + 1;
	lb_recv = 0;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_buf = (void *)((char *)lo_send_buf + lb_send * dim_acc * type_size);
      lo_recv_buf = (void *)((char *)lo_recv_buf + lb_recv * dim_acc * type_size);
      
    }

  }

  // for upper shadow

  if (ai->shadow_size_hi){

    hi_send_buf = array_addr;
    hi_recv_buf = array_addr;

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_buf = (void *)((char *)hi_send_buf + lb_send * dim_acc * type_size);
      hi_recv_buf = (void *)((char *)hi_recv_buf + lb_recv * dim_acc * type_size);
      
    }

  }

  //
  // initialize communication
  //

  // for lower shadow

  if (ai->shadow_size_lo){
    if (is_periodic || my_pos != lb_pos){
      MPI_Recv_init(lo_recv_buf, 1, ai->mpi_datatype_shadow_lo,
		    lo_rank, _XMP_N_MPI_TAG_REFLECT_LO, *comm,
		    &adesc->mpi_req_shadow[adesc->num_reqs++]);
    }
    if (is_periodic || my_pos != ub_pos){
      MPI_Send_init(lo_send_buf, 1, ai->mpi_datatype_shadow_lo,
		    hi_rank, _XMP_N_MPI_TAG_REFLECT_LO, *comm,
		    &adesc->mpi_req_shadow[adesc->num_reqs++]);
    }
  }

  // for upper shadow

  if (ai->shadow_size_hi){
    if (is_periodic || my_pos != ub_pos){
      MPI_Recv_init(hi_recv_buf, 1, ai->mpi_datatype_shadow_hi,
		    hi_rank, _XMP_N_MPI_TAG_REFLECT_HI, *comm,
		    &adesc->mpi_req_shadow[adesc->num_reqs++]);
    }
    if (is_periodic || my_pos != lb_pos){
      MPI_Send_init(hi_send_buf, 1, ai->mpi_datatype_shadow_hi,
		    lo_rank, _XMP_N_MPI_TAG_REFLECT_HI, *comm,
		    &adesc->mpi_req_shadow[adesc->num_reqs++]);
    }
  }

}


void _XMPF_reflect_start(_XMP_array_t **a_desc, int lwidth[], int uwidth[],
			 _Bool is_periodic[])
{
  _XMP_array_t *a = *a_desc;

  // NOTE: now, lwidth and uwidth are not used and is_periodic is assumed to be true.

  if (a->num_reqs == -1){

    a->num_reqs = 0;

    for (int i = 0; i < a->dim; i++){

      _XMP_array_info_t *ai = &(a->info[i]);

      if (ai->shadow_type == _XMP_N_SHADOW_NONE){
	continue;
      }
      else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
	_XMPF_optimized_reflect_dim(a, i, true);
      }
      else { /* _XMP_N_SHADOW_FULL */
	_XMP_fatal("xmpf_reflect: not surport full shadow");
      }

    }

  }

  MPI_Startall(a->num_reqs, a->mpi_req_shadow);

}


void _XMPF_reflect_wait(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;
  MPI_Status stat[4 * (a->num_reqs)];
  MPI_Waitall(a->num_reqs, a->mpi_req_shadow, stat);
}


void xmpf_reflect__(_XMP_array_t **a_desc)
{
  _XMPF_reflect_start(a_desc, NULL, NULL, NULL);
  _XMPF_reflect_wait(a_desc);
}


/* void xmpf_reflect_1__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      _Bool *periodic_0) */
/* { */
/*   int lwidth[1], uwidth[1]; */
/*   _Bool is_periodic[1]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   is_periodic[0] = *periodic_0; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_2__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, */
/* 		      _Bool *periodic_0, _Bool *periodic_1) */
/* { */
/*   int lwidth[2], uwidth[2]; */
/*   _Bool is_periodic[2]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_3__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2, */
/* 		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2) */
/* { */
/*   int lwidth[3], uwidth[3]; */
/*   _Bool is_periodic[3]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   is_periodic[2] = *periodic_2; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_4__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2, */
/* 		      int *lwidth_3, int *uwidth_3, */
/* 		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2, */
/* 		      _Bool *periodic_3) */
/* { */
/*   int lwidth[4], uwidth[4]; */
/*   _Bool is_periodic[4]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2; */
/*   lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   is_periodic[2] = *periodic_2; */
/*   is_periodic[3] = *periodic_3; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_5__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2, */
/* 		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4, */
/* 		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2, */
/* 		      _Bool *periodic_3, _Bool *periodic_4) */
/* { */
/*   int lwidth[5], uwidth[5]; */
/*   _Bool is_periodic[5]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2; */
/*   lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3; */
/*   lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   is_periodic[2] = *periodic_2; */
/*   is_periodic[3] = *periodic_3; */
/*   is_periodic[4] = *periodic_4; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_6__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2, */
/* 		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4, */
/* 		      int *lwidth_5, int *uwidth_5, */
/* 		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2, */
/* 		      _Bool *periodic_3, _Bool *periodic_4, _Bool *periodic_5) */
/* { */
/*   int lwidth[6], uwidth[6]; */
/*   _Bool is_periodic[6]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2; */
/*   lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3; */
/*   lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4; */
/*   lwidth[5] = *lwidth_5; uwidth[5] = *uwidth_5; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   is_periodic[2] = *periodic_2; */
/*   is_periodic[3] = *periodic_3; */
/*   is_periodic[4] = *periodic_4; */
/*   is_periodic[5] = *periodic_5; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


/* void xmpf_reflect_7__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0, */
/* 		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2, */
/* 		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4, */
/* 		      int *lwidth_5, int *uwidth_5, int *lwidth_6, int *uwidth_6, */
/* 		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2, */
/* 		      _Bool *periodic_3, _Bool *periodic_4, _Bool *periodic_5, */
/* 		      _Bool *periodic_6) */
/* { */
/*   int lwidth[7], uwidth[7]; */
/*   _Bool is_periodic[7]; */
/*   lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0; */
/*   lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1; */
/*   lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2; */
/*   lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3; */
/*   lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4; */
/*   lwidth[5] = *lwidth_5; uwidth[5] = *uwidth_5; */
/*   lwidth[6] = *lwidth_6; uwidth[6] = *uwidth_6; */
/*   is_periodic[0] = *periodic_0; */
/*   is_periodic[1] = *periodic_1; */
/*   is_periodic[2] = *periodic_2; */
/*   is_periodic[3] = *periodic_3; */
/*   is_periodic[4] = *periodic_4; */
/*   is_periodic[5] = *periodic_5; */
/*   is_periodic[6] = *periodic_6; */
/*   _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic); */
/* } */


//
// reduction
//
void xmpf_reduction__(void *data_addr, int *count, int *datatype, int *op,
		      _XMP_object_ref_t **r_desc)
{
  // Now, r_desc is ignored.
  _XMP_reduce_CLAUSE(data_addr, *count, *datatype + 500, *op);
}


//
// bcast
//
void xmpf_bcast__(void *data_addr, int *count, int *datatype,
		  _XMP_object_ref_t **on_desc, _XMP_object_ref_t **from_desc)
{
  _XMP_nodes_t *on;
  _XMP_nodes_t *from;

  size_t size = _XMP_get_datatype_size(*datatype);

  int root = 0;

  _XMP_RETURN_IF_SINGLE;

  // set up node set
  if (*((int *)on_desc) && *on_desc){
    // Now, on_desc must be a nodes arrays.
    _XMP_ASSERT((*on_desc)->ref_kind == XMP_OBJ_REF_NODES);
    on = (*on_desc)->n_desc;
    if (!on->is_member) return;
  }
  else {
    on = _XMP_get_execution_nodes();
  }

  // calc source nodes number
  if (*((int *)from_desc) && *from_desc){

    // Now, from_desc must be a nodes arrays.
    _XMP_ASSERT((*from_desc)->ref_kind == XMP_OBJ_REF_NODES);

    int acc_nodes_size = 1;

    from = (*from_desc)->n_desc;

    if (!from->is_member) {
      _XMP_fatal("broadcast failed, cannot find the source node");
    }

    for (int i = 0; i < from->dim; i++){
      //root += (acc_nodes_size * ((*from_desc)->index[i] - 1));
      root += (acc_nodes_size * ((*from_desc)->REF_INDEX[i] - 1));
      acc_nodes_size *= from->info[i].size;
    }
  }

  MPI_Bcast(data_addr, (*count)*size, MPI_BYTE, root, *((MPI_Comm *)on->comm));

/*   // setup type */
/*   MPI_Datatype mpi_datatype; */
/*   MPI_Type_contiguous(*datatype_size, MPI_BYTE, &mpi_datatype); */
/*   MPI_Type_commit(&mpi_datatype); */

/*   // bcast */
/*   MPI_Bcast(*addr, *count, mpi_datatype, root, *((MPI_Comm *)on->comm)); */

/*   MPI_Type_free(&mpi_datatype); */

}


//
// barrier
//
void xmpf_barrier__(_XMP_object_ref_t **desc)
{
  if (*((int *)desc) && *desc){
    // Now, desc must be a nodes arrays.
    _XMP_ASSERT((*desc)->ref_kind == XMP_OBJ_REF_NODES);
    _XMP_barrier_NODES_ENTIRE((*desc)->n_desc);
  }
  else {
    _XMP_barrier_EXEC();
  }

  //xmpf_dbg_printf("xmpf_barrier done\n");

}
