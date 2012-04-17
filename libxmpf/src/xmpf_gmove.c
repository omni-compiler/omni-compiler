#include "xmpf_internal.h"

void *_XMP_get_array_addr(_XMP_array_t *a, int *gidx)
{
  int ndims = a->dim;
  void *ret = *(a->array_addr_p);
  //xmpf_dbg_printf("ret = %p\n", ret);
  //xmpf_dbg_printf("*(a->array_addr_p) = %p\n", *(a->array_addr_p));

  _XMP_ASSERT(a->is_allocated);

  for (int i = 0; i < ndims; i++){

    _XMP_array_info_t *ai = &(a->info[i]);
    _XMP_template_info_t *ti = &(a->align_template->info[ai->align_template_index]);
    _XMP_template_chunk_t *tc = &(a->align_template->chunk[ai->align_template_index]);
    int lidx;
    int offset;
    int glb, t_lb, np, w;
    int l_shadow;
    size_t type_size = a->type_size;

    switch (ai->align_manner){

      case _XMP_N_ALIGN_NOT_ALIGNED:
	lidx = gidx[i] - ai->ser_lower;
	break;

      case _XMP_N_ALIGN_DUPLICATION:
	lidx = gidx[i] - ai->ser_lower;
	break;

      case _XMP_N_ALIGN_BLOCK:
	// par_lower is the index of the lower bound of the local section.
	glb = ai->par_lower;
	l_shadow = ai->shadow_size_lo;
	lidx = gidx[i] - glb + l_shadow;
	break;

      case _XMP_N_ALIGN_CYCLIC:
	// assumed that even a cyclic array is distributed equally
	offset = ai->align_subscript;
	t_lb = ti->ser_lower;
	np = ai->par_stride;
	lidx = (gidx[i] + offset - t_lb) / np;
	break;

      case _XMP_N_ALIGN_BLOCK_CYCLIC:
	// assumed that even a cyclic array is distributed equally
	offset = ai->align_subscript;
	t_lb = ti->ser_lower;
	np = ai->par_stride;
	w = tc->par_stride;
	lidx = w * ((gidx[i] + offset - t_lb) / (np * w))
	     + ((gidx[i] + offset - t_lb) % w);
	break;

      default:
	_XMP_fatal("_XMP_get_array_addr: unknown align_manner");
    }

    //xmpf_dbg_printf("a->array_addr_p = %p\n", a->array_addr_p);
    //xmpf_dbg_printf("ret = %p, lidx = %d, ai->dim_acc = %d\n", ret, lidx, ai->dim_acc);
    ret = (char *)ret + lidx * ai->dim_acc * type_size;
  }

  return ret;

}

//
// s = a(i)
//
void xmpf_gmove_scalar_array__(void *scalar,
			       _XMP_array_t **rhs_desc, int ridx[])
{
  _XMP_array_t *array = *rhs_desc;
  int type_size = array->type_size;
  void *src_addr = NULL;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();

  xmpf_dbg_printf("gmove : *(a->array_addr_p) = %p\n", *(array->array_addr_p));
  MPI_Barrier(MPI_COMM_WORLD);
/*   if (_XMP_IS_SINGLE){ */
/*     memcpy(scalar, src_addr, type_size); */
/*     return; */
/*   } */

  int root_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, ridx);
  xmpf_dbg_printf("root_rank = %d\n", root_rank);

  if (root_rank == exec_nodes->comm_rank){
    // I am the root.
    src_addr = _XMP_get_array_addr(array, ridx);
    xmpf_dbg_printf("src_addr = %p\n", src_addr);
  }

  // broadcast
  _XMP_gmove_bcast_SCALAR(scalar, src_addr, type_size, root_rank);
}

//
// a(i) = s
//
void xmpf_gmove_array_scalar__(_XMP_array_t **lhs_desc, int *lidx,
			       void *scalar)
{
  _XMP_array_t *array = *lhs_desc;
  int type_size = array->type_size;
  void *dst_addr;
  _XMP_nodes_t *exec_nodes = _XMP_get_execution_nodes();

  int owner_rank = _XMP_calc_gmove_array_owner_linear_rank_SCALAR(array, lidx);

  if (owner_rank == exec_nodes->comm_rank){
    // I am the owner.
    dst_addr = _XMP_get_array_addr(array, lidx);
    memcpy(dst_addr, scalar, type_size);
  }

}


//
// a(:) = b(:)
//
void xmpf_gmove_array_array__(_XMP_array_t **lhs_desc, int lidx[][3],
			      _XMP_array_t **rhs_desc, int ridx[][3])
{
  _XMP_array_t *dst_array = *lhs_desc;
  _XMP_array_t *src_array = *rhs_desc;

  int type = dst_array->type;
  size_t type_size = dst_array->type_size;

  _XMP_ASSERT(src_array->type == type);
  _XMP_ASSERT(src_array->type_size == type_size);

  unsigned long long gmove_total_elmts = 0;

  // get dst info
  unsigned long long dst_total_elmts = 1;
  void *dst_addr = *(dst_array->array_addr_p);
  int dst_dim = dst_array->dim;
  int dst_l[dst_dim], dst_u[dst_dim], dst_s[dst_dim];
  unsigned long long dst_d[dst_dim];
  for (int i = 0; i < dst_dim; i++) {
    dst_l[i] = lidx[i][0];
    dst_u[i] = lidx[i][1];
    dst_s[i] = lidx[i][2];
    dst_d[i] = dst_array->info[i].dim_acc;
    _XMP_normalize_array_section(&(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    dst_total_elmts *= _XMP_M_COUNT_TRIPLETi(dst_l[i], dst_u[i], dst_s[i]);
  }

  // get src info
  unsigned long long src_total_elmts = 1;
  void *src_addr = *(src_array->array_addr_p);
  int src_dim = src_array->dim;;
  int src_l[src_dim], src_u[src_dim], src_s[src_dim];
  unsigned long long src_d[src_dim];
  for (int i = 0; i < src_dim; i++) {
    src_l[i] = ridx[i][0];
    src_u[i] = ridx[i][1];
    src_s[i] = ridx[i][2];
    src_d[i] = src_array->info[i].dim_acc;
    _XMP_normalize_array_section(&(src_l[i]), &(src_u[i]), &(src_s[i]));
    src_total_elmts *= _XMP_M_COUNT_TRIPLETi(src_l[i], src_u[i], src_s[i]);
  }

  if (dst_total_elmts != src_total_elmts) {
    _XMP_fatal("bad assign statement for gmove");
  } else {
    gmove_total_elmts = dst_total_elmts;
  }

  if (_XMP_IS_SINGLE) {
    for (int i = 0; i < dst_dim; i++) {
      _XMP_gtol_array_ref_triplet(dst_array, i, &(dst_l[i]), &(dst_u[i]), &(dst_s[i]));
    }

    for (int i = 0; i < src_dim; i++) {
      _XMP_gtol_array_ref_triplet(src_array, i, &(src_l[i]), &(src_u[i]), &(src_s[i]));
    }

    _XMP_gmove_localcopy_ARRAY(type, type_size,
                               dst_addr, dst_dim, dst_l, dst_u, dst_s, dst_d,
                               src_addr, src_dim, src_l, src_u, src_s, src_d);
    return;
  }

  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(type_size, MPI_BYTE, &mpi_datatype);
  MPI_Type_commit(&mpi_datatype);

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
          _XMP_sendrecv_ARRAY(gmove_total_elmts,
                              type, type_size, &mpi_datatype,
                              dst_array, dst_array_nodes_ref,
                              recv_lower, recv_upper, recv_stride, dst_d,
                              src_array, src_array_nodes_ref,
                              send_lower, send_upper, send_stride, src_d);
        }
      } while (_XMP_calc_next_next_rank(src_array_nodes, src_array_nodes_ref));
    }
  } while (_XMP_calc_next_next_rank(dst_array_nodes, dst_array_nodes_ref));

  MPI_Type_free(&mpi_datatype);

}
