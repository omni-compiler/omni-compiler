#include "xmpf_internal.h"

//#define DBG 1

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
