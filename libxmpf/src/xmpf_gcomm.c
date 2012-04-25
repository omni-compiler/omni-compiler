#include "xmpf_internal.h"

//
// reflect
//
void xmpf_reflect__(_XMP_array_t **a_desc)
{
  _XMP_array_t *a = *a_desc;
  void *l_send_buf, *u_send_buf, *l_recv_buf, *u_recv_buf;

  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NONE){
      continue;
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      if (ai->shadow_size_lo > 0 || ai->shadow_size_hi > 0){
	_XMP_pack_shadow_NORMAL(&l_send_buf, &u_send_buf,
				*(a->array_addr_p), a, i);
	_XMP_exchange_shadow_NORMAL(&l_recv_buf, &u_recv_buf,
				    l_send_buf, u_send_buf, a, i);
	_XMP_unpack_shadow_NORMAL(l_recv_buf, u_recv_buf,
				  *(a->array_addr_p), a, i);
      }
      
    }
    else { /* _XMP_N_SHADOW_FULL */
      _XMP_fatal("xmpf_reflect: not support full shadow");
    }
  }

}


// Now, all of the reflection width are ignored.
void xmpf_reflect_1__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_2__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_3__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2, int lwidth_3, int uwidth_3)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_4__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2, int lwidth_3, int uwidth_3,
		      int lwidth_4, int uwidth_4)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_5__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2, int lwidth_3, int uwidth_3,
		      int lwidth_4, int uwidth_4, int lwidth_5, int uwidth_5)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_6__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2, int lwidth_3, int uwidth_3,
		      int lwidth_4, int uwidth_4, int lwidth_5, int uwidth_5,
		      int lwidth_6, int uwidth_6)
{
  xmpf_reflect__(a_desc);
}


void xmpf_reflect_7__(_XMP_array_t **a_desc, int lwidth_1, int uwidth_1,
		      int lwidth_2, int uwidth_2, int lwidth_3, int uwidth_3,
		      int lwidth_4, int uwidth_4, int lwidth_5, int uwidth_5,
		      int lwidth_6, int uwidth_6, int lwidth_7, int uwidth_7)
{
  xmpf_reflect__(a_desc);
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
void xmpf_bcast__(void *addr, int *count, size_t *datatype_size,
		  _XMP_object_ref_t **on_desc, _XMP_object_ref_t **from_desc)
{
  _XMP_nodes_t *on;
  _XMP_nodes_t *from;

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
      root += (acc_nodes_size * ((*from_desc)->index[i] - 1));
      acc_nodes_size *= from->info[i].size;
    }
  }

  MPI_Bcast(addr, (*count)*(*datatype_size), MPI_BYTE,
	    root, *((MPI_Comm *)on->comm));

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
