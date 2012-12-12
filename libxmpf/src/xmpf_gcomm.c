#include "xmpf_internal.h"

#define DBG 1

#ifdef DBG
double t0;
double t_mem = 0;
double t_copy = 0;
double t_comm = 0;
#endif

//
// reflect
//

void _XMPF_pack_shadow_NORMAL(void **lo_buffer, void **hi_buffer, void *array_addr,
			      _XMP_array_t *array_desc, int array_index,
			      _Bool is_periodic) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (!is_periodic && size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  int array_type = array_desc->type;
  int array_dim = array_desc->dim;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  // pack lo shadow
  if (is_periodic || rank != (size - 1)) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // alloc buffer
      *lo_buffer = _XMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts) * (array_desc->type_size));
#ifdef DBG
	t_mem = t_mem + MPI_Wtime() - t0;
#endif

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_upper - array_desc->info[i].shadow_size_lo + 1;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_lo - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      if (array_index == array_dim - 1){
	*lo_buffer = array_addr;
	for (int i = 0; i < array_dim; i++){
	  *lo_buffer = (void *)((char *)(*lo_buffer) + lower[i] * dim_acc[i] * (array_desc->type_size));
	}
      }
      else {
#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // pack data
	_XMP_pack_array(*lo_buffer, array_addr, array_type, array_desc->type_size,
			array_dim, lower, upper, stride, dim_acc);
#ifdef DBG
	t_copy = t_copy + MPI_Wtime() - t0;
#endif
      }

    }
  }

  // pack hi shadow
  if (is_periodic || rank != 0) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // alloc buffer
      *hi_buffer = _XMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts) * (array_desc->type_size));
#ifdef DBG
	t_mem = t_mem + MPI_Wtime() - t0;
#endif

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

      if (array_index == array_dim - 1){
	*hi_buffer = array_addr;
	for (int i = 0; i < array_dim; i++){
	  *hi_buffer = (void *)((char *)(*hi_buffer) + lower[i] * dim_acc[i] * (array_desc->type_size));
	}
      }
      else {
#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // pack data
	_XMP_pack_array(*hi_buffer, array_addr, array_type, array_desc->type_size,
			array_dim, lower, upper, stride, dim_acc);
#ifdef DBG
	t_copy = t_copy + MPI_Wtime() - t0;
#endif
      }
    }
  }
}

void _XMPF_unpack_shadow_NORMAL(void *lo_buffer, void *hi_buffer, void *array_addr,
				_XMP_array_t *array_desc, int array_index,
				_Bool is_periodic) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int size = ai->shadow_comm_size;
  if (!is_periodic && size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  int array_type = array_desc->type;
  int array_dim = array_desc->dim;

  int lower[array_dim], upper[array_dim], stride[array_dim];
  unsigned long long dim_acc[array_dim];

  if (array_index == array_dim - 1) return;

  // unpack lo shadow
  if (is_periodic || rank != 0) {
    if (ai->shadow_size_lo > 0) {
      // FIXME strict condition
      if (ai->shadow_size_lo > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = 0;
          upper[i] = array_desc->info[i].shadow_size_lo - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // unpack data
      _XMP_unpack_array(array_addr, lo_buffer, array_type, array_desc->type_size,
                        array_dim, lower, upper, stride, dim_acc);
#ifdef DBG
	t_copy = t_copy + MPI_Wtime() - t0;
#endif

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // free buffer
      _XMP_free(lo_buffer);
#ifdef DBG
	t_mem = t_mem + MPI_Wtime() - t0;
#endif
    }
  }

  // unpack hi shadow
  if (is_periodic || rank != (size - 1)) {
    if (ai->shadow_size_hi > 0) {
      // FIXME strict condition
      if (ai->shadow_size_hi > ai->par_size) {
        _XMP_fatal("shadow size is too big");
      }

      // calc index
      for (int i = 0; i < array_dim; i++) {
        if (i == array_index) {
          lower[i] = array_desc->info[i].local_upper + 1;
          upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
          stride[i] = 1;
        }
        else {
          lower[i] = array_desc->info[i].local_lower;
          upper[i] = array_desc->info[i].local_upper;
          stride[i] = array_desc->info[i].local_stride;
        }

        dim_acc[i] = array_desc->info[i].dim_acc;
      }

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // unpack data
      _XMP_unpack_array(array_addr, hi_buffer, array_type, array_desc->type_size,
                        array_dim, lower, upper, stride, dim_acc);
#ifdef DBG
	t_copy = t_copy + MPI_Wtime() - t0;
#endif

#ifdef DBG
	t0 = MPI_Wtime();
#endif
      // free buffer
      _XMP_free(hi_buffer);
#ifdef DBG
	t_mem = t_mem + MPI_Wtime() - t0;
#endif
    }
  }
}

void _XMPF_exchange_shadow_NORMAL(void **lo_recv_buffer, void **hi_recv_buffer,
				  void *lo_send_buffer, void *hi_send_buffer,
				  _XMP_array_t *array_desc, int array_index,
				  _Bool is_periodic) {
  _XMP_RETURN_IF_SINGLE;

  if (!array_desc->is_allocated) {
    return;
  }

  _XMP_array_info_t *ai = &(array_desc->info[array_index]);
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  int array_dim = array_desc->dim;

  // get communicator info
  int size = ai->shadow_comm_size;
  if (!is_periodic && size == 1) {
    return;
  }

  int rank = ai->shadow_comm_rank;
  MPI_Comm *comm = ai->shadow_comm;
  int src, dst;

  // setup type
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(array_desc->type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

  // exchange shadow
  MPI_Request send_req[2];
  MPI_Request recv_req[2];

  if (ai->shadow_size_lo > 0) {
    if (is_periodic || rank != 0) {

      if (array_index == array_dim - 1){

	int lower[array_dim];
	unsigned long long dim_acc[array_dim];

	for (int i = 0; i < array_dim; i++) {
	  if (i == array_index) {
	    lower[i] = 0;
	    //	    upper[i] = array_desc->info[i].shadow_size_lo - 1;
	    //	    stride[i] = 1;
	  }
	  else {
	    lower[i] = array_desc->info[i].local_lower;
	    //	    upper[i] = array_desc->info[i].local_upper;
	    //	    stride[i] = array_desc->info[i].local_stride;
	  }

	  dim_acc[i] = array_desc->info[i].dim_acc;
	}

	*lo_recv_buffer = array_desc->array_addr_p;
	for (int i = 0; i < array_dim; i++){
	  *lo_recv_buffer = (void *)((char *)(*lo_recv_buffer) + lower[i] * dim_acc[i] * (array_desc->type_size));
	}

      }
      else {
	*lo_recv_buffer = _XMP_alloc((ai->shadow_size_lo) * (ai->dim_elmts) * (array_desc->type_size));
      }

      src = (rank - 1 + size) % size;
      MPI_Irecv(*lo_recv_buffer, (ai->shadow_size_lo) * (ai->dim_elmts), mpi_datatype,
		src, _XMP_N_MPI_TAG_REFLECT_LO, *comm, &(recv_req[0]));
    }

    if (is_periodic || rank != (size - 1)) {
      dst = (rank + 1) % size;
      MPI_Isend(lo_send_buffer, (ai->shadow_size_lo) * (ai->dim_elmts), mpi_datatype,
                dst, _XMP_N_MPI_TAG_REFLECT_LO, *comm, &(send_req[0]));
    }
  }

  if (ai->shadow_size_hi > 0) {
    if (is_periodic || rank != (size - 1)) {

      if (array_index == array_dim - 1){

	int lower[array_dim];
	unsigned long long dim_acc[array_dim];

	for (int i = 0; i < array_dim; i++) {
	  if (i == array_index) {
	    lower[i] = array_desc->info[i].local_upper + 1;
	    //	    upper[i] = lower[i] + array_desc->info[i].shadow_size_hi - 1;
	    //	    stride[i] = 1;
	  }
	  else {
	    lower[i] = array_desc->info[i].local_lower;
	    //	    upper[i] = array_desc->info[i].local_upper;
	    //	    stride[i] = array_desc->info[i].local_stride;
	  }
	  
	  dim_acc[i] = array_desc->info[i].dim_acc;
	}

	*hi_recv_buffer = array_desc->array_addr_p;
	for (int i = 0; i < array_dim; i++){
	  *hi_recv_buffer = (void *)((char *)(*hi_recv_buffer) + lower[i] * dim_acc[i] * (array_desc->type_size));
	}
      }
      else {
	*hi_recv_buffer = _XMP_alloc((ai->shadow_size_hi) * (ai->dim_elmts) * (array_desc->type_size));
      }

      src = (rank + 1) % size;
      MPI_Irecv(*hi_recv_buffer, (ai->shadow_size_hi) * (ai->dim_elmts), mpi_datatype,
                src, _XMP_N_MPI_TAG_REFLECT_HI, *comm, &(recv_req[1]));
    }

    if (is_periodic || rank != 0) {
      dst = (rank - 1 + size) % size;
      MPI_Isend(hi_send_buffer, (ai->shadow_size_hi) * (ai->dim_elmts), mpi_datatype,
                dst, _XMP_N_MPI_TAG_REFLECT_HI, *comm, &(send_req[1]));
    }
  }

  // wait & free
  MPI_Status stat;

  if (ai->shadow_size_lo > 0) {
    if (is_periodic || rank != 0) {
      MPI_Wait(&(recv_req[0]), &stat);
    }

    if (is_periodic || rank != (size - 1)) {
      MPI_Wait(&(send_req[0]), &stat);
      if (array_index != array_dim - 1) _XMP_free(lo_send_buffer);
    }
  }

  if (ai->shadow_size_hi > 0) {
    if (is_periodic || rank != (size - 1)) {
      MPI_Wait(&(recv_req[1]), &stat);
    }

    if (is_periodic || rank != 0) {
      MPI_Wait(&(send_req[1]), &stat);
      if (array_index != array_dim - 1) _XMP_free(hi_send_buffer);
    }
  }

  MPI_Type_free(&mpi_datatype);
}

void _XMPF_reflect_(_XMP_array_t **a_desc, int lwidth[], int uwidth[],
		    _Bool is_periodic[])
{
  _XMP_array_t *a = *a_desc;
  void *l_send_buf, *u_send_buf, *l_recv_buf, *u_recv_buf;

  // NOTE: now, lwidth and uwidth are not used and is_periodic is assumed to be true.

  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NONE){
      continue;
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      if (ai->shadow_size_lo > 0 || ai->shadow_size_hi > 0){

	_XMPF_pack_shadow_NORMAL(&l_send_buf, &u_send_buf,
				 a->array_addr_p, a, i, true);
#ifdef DBG
	t0 = MPI_Wtime();
#endif
	_XMPF_exchange_shadow_NORMAL(&l_recv_buf, &u_recv_buf,
				     l_send_buf, u_send_buf, a, i, true);
#ifdef DBG
	t_comm = t_comm + MPI_Wtime() - t0;
#endif

	_XMPF_unpack_shadow_NORMAL(l_recv_buf, u_recv_buf,
				   a->array_addr_p, a, i, true);
      }
      
    }
    else { /* _XMP_N_SHADOW_FULL */
      _XMP_fatal("xmpf_reflect: not surport full shadow");
    }
  }

}


void xmpf_reflect__(_XMP_array_t **a_desc)
{
  _XMPF_reflect_(a_desc, NULL, NULL, NULL);
}


void xmpf_reflect_1__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      _Bool *periodic_0)
{
  int lwidth[1], uwidth[1];
  _Bool is_periodic[1];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  is_periodic[0] = *periodic_0;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_2__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1,
		      _Bool *periodic_0, _Bool *periodic_1)
{
  int lwidth[2], uwidth[2];
  _Bool is_periodic[2];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_3__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2,
		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2)
{
  int lwidth[3], uwidth[3];
  _Bool is_periodic[3];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  is_periodic[2] = *periodic_2;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_4__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2,
		      int *lwidth_3, int *uwidth_3,
		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2,
		      _Bool *periodic_3)
{
  int lwidth[4], uwidth[4];
  _Bool is_periodic[4];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2;
  lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  is_periodic[2] = *periodic_2;
  is_periodic[3] = *periodic_3;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_5__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2,
		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4,
		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2,
		      _Bool *periodic_3, _Bool *periodic_4)
{
  int lwidth[5], uwidth[5];
  _Bool is_periodic[5];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2;
  lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3;
  lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  is_periodic[2] = *periodic_2;
  is_periodic[3] = *periodic_3;
  is_periodic[4] = *periodic_4;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_6__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2,
		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4,
		      int *lwidth_5, int *uwidth_5,
		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2,
		      _Bool *periodic_3, _Bool *periodic_4, _Bool *periodic_5)
{
  int lwidth[6], uwidth[6];
  _Bool is_periodic[6];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2;
  lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3;
  lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4;
  lwidth[5] = *lwidth_5; uwidth[5] = *uwidth_5;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  is_periodic[2] = *periodic_2;
  is_periodic[3] = *periodic_3;
  is_periodic[4] = *periodic_4;
  is_periodic[5] = *periodic_5;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


void xmpf_reflect_7__(_XMP_array_t **a_desc, int *lwidth_0, int *uwidth_0,
		      int *lwidth_1, int *uwidth_1, int *lwidth_2, int *uwidth_2,
		      int *lwidth_3, int *uwidth_3, int *lwidth_4, int *uwidth_4,
		      int *lwidth_5, int *uwidth_5, int *lwidth_6, int *uwidth_6,
		      _Bool *periodic_0, _Bool *periodic_1, _Bool *periodic_2,
		      _Bool *periodic_3, _Bool *periodic_4, _Bool *periodic_5,
		      _Bool *periodic_6)
{
  int lwidth[7], uwidth[7];
  _Bool is_periodic[7];
  lwidth[0] = *lwidth_0; uwidth[0] = *uwidth_0;
  lwidth[1] = *lwidth_1; uwidth[1] = *uwidth_1;
  lwidth[2] = *lwidth_2; uwidth[2] = *uwidth_2;
  lwidth[3] = *lwidth_3; uwidth[3] = *uwidth_3;
  lwidth[4] = *lwidth_4; uwidth[4] = *uwidth_4;
  lwidth[5] = *lwidth_5; uwidth[5] = *uwidth_5;
  lwidth[6] = *lwidth_6; uwidth[6] = *uwidth_6;
  is_periodic[0] = *periodic_0;
  is_periodic[1] = *periodic_1;
  is_periodic[2] = *periodic_2;
  is_periodic[3] = *periodic_3;
  is_periodic[4] = *periodic_4;
  is_periodic[5] = *periodic_5;
  is_periodic[6] = *periodic_6;
  _XMPF_reflect_(a_desc, lwidth, uwidth, is_periodic);
}


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
