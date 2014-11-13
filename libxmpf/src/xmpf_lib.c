/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmpf_internal.h"
#include "xmp_internal.h"
#include "xmp.h"
//#include "xmpf.h"

MPI_Fint xmp_get_mpi_comm_(void) {
  MPI_Comm commc = xmp_get_mpi_comm();
  MPI_Fint commf = MPI_Comm_c2f(commc);
  return commf;
}

void xmp_init_mpi_(void) {
}

void xmp_finalize_mpi_(void) {
}

void xmp_init_() {
  _XMP_init(0, NULL);
}

void xmp_finalize_(void) {
  xmp_finalize();
}

int xmp_num_nodes_(void) {
  return _XMP_get_execution_nodes()->comm_size;
}

int xmp_node_num_(void) {
  return _XMP_get_execution_nodes()->comm_rank + 1;
}

void xmp_barrier_(void) {
  _XMP_barrier_EXEC();
}

int xmp_all_num_nodes_(void) {
  return _XMP_world_size;
}

int xmp_all_node_num_(void) {
  return _XMP_world_rank + 1;
}

double xmp_wtime_(void) {
  return MPI_Wtime();
}

double xmp_wtick_(void) {
  return MPI_Wtick();
}

int xmp_array_ndims_(xmp_desc_t **d, int *ndims) {

  return xmp_array_ndims(*d, ndims);

}

int xmp_array_lbound_(xmp_desc_t **d, int *dim, int *lbound) {

  return xmp_array_lbound(*d, *dim, lbound);

}

int xmp_array_ubound_(xmp_desc_t **d, int *dim, int *ubound) {

  return xmp_array_ubound(*d, *dim, ubound);

}


int xmp_array_gsize_(xmp_desc_t **d, int *dim) {

  _XMP_array_t *a = *(_XMP_array_t **)d;

  return a->info[*dim-1].ser_size;

}

int xmp_array_lsize_(xmp_desc_t **d, int *dim, int *lsize){

  return xmp_array_lsize(*d, *dim, lsize);

}

int xmp_array_ushadow_(xmp_desc_t **d, int *dim, int *ushadow){

  return xmp_array_ushadow(*d, *dim, ushadow);

}

int xmp_array_lshadow_(xmp_desc_t **d, int *dim, int *lshadow){

  return xmp_array_lshadow(*d, *dim, lshadow);

}

int xmp_array_lead_dim_(xmp_desc_t **d, int size[]){

  xmp_array_lead_dim(*d, size);

  return 0;
}

int xmp_align_axis_(xmp_desc_t **d, int *dim, int *axis){

  return xmp_align_axis(*d, *dim, axis);

}

int xmp_align_offset_(xmp_desc_t **d, int *dim, int *offset){

  return xmp_align_offset(*d, *dim, offset);

}

int xmp_align_replicated_(xmp_desc_t **d, int *dim, int *replicated){

  return xmp_align_replicated(*d, *dim, replicated);

}

int xmp_align_template_(xmp_desc_t **d, xmp_desc_t *dt){

  return xmp_align_template(*d, dt);
}


int xmp_template_fixed_(xmp_desc_t **d, int *fixed){

  return xmp_template_fixed(*d, fixed);

}

int xmp_template_ndims_(xmp_desc_t **d, int *ndims){

  return xmp_template_ndims(*d, ndims);

}

int xmp_template_lbound_(xmp_desc_t **d, int *dim, int *lbound) {

  return xmp_template_lbound(*d, *dim, lbound);

}

int xmp_template_ubound_(xmp_desc_t **d, int *dim, int *ubound) {

  return xmp_template_ubound(*d, *dim, ubound);

}

int xmp_dist_format_(xmp_desc_t **d, int *dim, int *format){

  return xmp_dist_format(*d, *dim, format);

}

int xmp_dist_blocksize_(xmp_desc_t **d, int *dim, int *blocksize){

  return xmp_dist_blocksize(*d, *dim, blocksize);

}

int xmp_dist_gblockmap_(xmp_desc_t **d, int *dim, int *map){

  return xmp_dist_gblockmap(*d, *dim, map);

}

int xmp_dist_nodes_(xmp_desc_t **d, xmp_desc_t *dn){

  return xmp_dist_nodes(*d, dn);

}

int xmp_dist_axis_(xmp_desc_t **d, int *dim, int *axis){

  return xmp_dist_axis(*d, *dim, axis);

}

int xmp_nodes_ndims_(xmp_desc_t **d, int *ndims){

  return xmp_nodes_ndims(*d, ndims);

}

int xmp_nodes_index_(xmp_desc_t **d, int *dim, int *index){

  return xmp_nodes_index(*d, *dim, index);

}


int xmp_nodes_size_(xmp_desc_t **d, int *dim, int *size){

  return xmp_nodes_size(*d, *dim, size);

}

int xmp_nodes_equiv_(xmp_desc_t **d, xmp_desc_t *dn, int *lb, int *ub, int *st){

  return xmp_nodes_equiv(*d, dn, lb, ub, st);

}

void xmp_transpose_(_XMP_array_t **dst_d, _XMP_array_t **src_d, int *opt){

#if 1
   xmpf_transpose(*dst_d, *src_d, *opt);
   return;
#else
  _XMP_array_t *dst_array = *(_XMP_array_t **)dst_d;
  _XMP_array_t *src_array = *(_XMP_array_t **)src_d;

  int nnodes;

  int dst_block_dim, src_block_dim;

  void *sendbuf=NULL, *recvbuf=NULL;
  unsigned long long count, bufsize;

  int dst_chunk_size, dst_ser_size, type_size;
  int src_chunk_size, src_ser_size;

  nnodes = dst_array->align_template->onto_nodes->comm_size;

  // 2-dimensional Matrix
  if (dst_array->dim != 2) {
    _XMP_fatal("bad dimension for xmp_transpose");
  }

  // No Shadow
  if (dst_array->info[0].shadow_size_lo != 0 ||
      dst_array->info[0].shadow_size_hi != 0 ||
      src_array->info[0].shadow_size_lo != 0 ||
      src_array->info[0].shadow_size_hi != 0) {
   _XMP_fatal("A global array must not have shadows");
  fflush(stdout);
  }

  // Dividable by the number of nodes
  if (dst_array->info[0].ser_size % nnodes != 0) {
   _XMP_fatal("Not dividable by the number of nodes");
  fflush(stdout);
  }

  dst_block_dim = (dst_array->info[0].align_manner == _XMP_N_ALIGN_BLOCK) ? 0 : 1;
  src_block_dim = (src_array->info[0].align_manner == _XMP_N_ALIGN_BLOCK) ? 0 : 1;

  dst_chunk_size = dst_array->info[dst_block_dim].par_size;
  dst_ser_size = dst_array->info[dst_block_dim].ser_size;
  src_chunk_size = src_array->info[src_block_dim].par_size;
  src_ser_size = src_array->info[src_block_dim].ser_size;
  type_size = dst_array->type_size;

  count =  dst_chunk_size * src_chunk_size;
  bufsize = count * nnodes * type_size;

  _XMP_check_reflect_type();

  if (src_block_dim == 1){
    if (*opt ==0){
      sendbuf = _XMP_alloc(bufsize);
    }else if (*opt==1){
      sendbuf = dst_array->array_addr_p;
    }
    // src_array -> sendbuf
    _XMP_pack_vector2((char *)sendbuf, (char *)src_array->array_addr_p ,
		      src_chunk_size, dst_chunk_size, nnodes, type_size,
		      src_block_dim);
  }
  else {
    sendbuf = src_array->array_addr_p;
  }

  if (*opt == 0){
    recvbuf = _XMP_alloc(bufsize);
  }else if (*opt ==1){
    recvbuf = src_array->array_addr_p;
  }
  MPI_Alltoall(sendbuf, count * type_size, MPI_BYTE, recvbuf, count * type_size,
               MPI_BYTE, *((MPI_Comm *)src_array->align_template->onto_nodes->comm));

  if (dst_block_dim == 1){
    _XMPF_unpack_transpose_vector((char *)dst_array->array_addr_p ,
       (char *)recvbuf , src_ser_size, dst_chunk_size, type_size, dst_block_dim);

    if (*opt==0){
      _XMP_free(recvbuf);
    }
  }

  if (src_block_dim == 1){
    if (*opt == 0){
      _XMP_free(sendbuf);
    }
  }


  return;
#endif
}


void xmp_matmul_(_XMP_array_t **x_d, _XMP_array_t **a_d, _XMP_array_t **b_d){
   xmpf_matmul(*x_d, *a_d, *b_d);
}
