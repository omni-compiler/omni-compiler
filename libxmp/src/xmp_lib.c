/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "mpi.h"
#include "stdlib.h"
#include "xmp_internal.h"
#include "xmp.h"
#include <stddef.h>

// FIXME utility functions
void xmp_MPI_comm(void **comm) {
  *comm = _XMP_get_execution_nodes()->comm;
}

int xmp_num_nodes(void) {
  return _XMP_get_execution_nodes()->comm_size;
}

int xmp_node_num(void) {
  return _XMP_get_execution_nodes()->comm_rank + 1;
}

void xmp_barrier(void) {
  _XMP_barrier_EXEC();
}

int xmp_all_num_nodes(void) {
  return _XMP_world_size;
}

int xmp_all_node_num(void) {
  return _XMP_world_rank;
}

double xmp_wtime(void) {
  return MPI_Wtime();
}

double xmp_wtick(void) {
  return MPI_Wtick();
}

int xmp_array_ndims(xmp_desc_t d, int *ndims) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  *ndims = a->dim;

  return 0;
 
}

int xmp_array_lbound(xmp_desc_t d, int dim, int *lbound) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  *lbound = a->info[dim-1].ser_lower;

  return 0;

}

int xmp_array_ubound(xmp_desc_t d, int dim, int *ubound) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  *ubound = a->info[dim-1].ser_upper;

  return 0;

}

size_t xmp_array_type_size(xmp_desc_t d) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->type_size;
 
}

int xmp_array_gsize(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].ser_size;

}

int xmp_array_lsize(xmp_desc_t d, int dim, int *lsize){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *lsize = a->info[dim-1].par_size;

  return 0;

}

int xmp_array_gcllbound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].par_lower;

}

int xmp_array_gclubound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].par_upper;

}

int xmp_array_lcllbound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].local_lower;

}

int xmp_array_lclubound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].local_upper;

}

int xmp_array_gcglbound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].ser_lower;

}

int xmp_array_gcgubound(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].ser_upper;

}

int xmp_array_laddr(xmp_desc_t d, void **laddr){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *(void **)laddr = (void *)a->array_addr_p;

  return 0;

}

int xmp_array_ushadow(xmp_desc_t d, int dim, int *ushadow){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *ushadow = a->info[dim-1].shadow_size_hi;

  return 0;

}

int xmp_array_lshadow(xmp_desc_t d, int dim, int *lshadow){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *lshadow = a->info[dim-1].shadow_size_lo;

  return 0;

}

int xmp_array_owner(xmp_desc_t d, int ndims, int index[ndims], int dim){

  int ierr, idim, ival, idistnum, t_dist_size, format;
  xmp_desc_t dt, dn;

  _XMP_array_t *a = (_XMP_array_t *)d;

  ierr = xmp_align_template(d, &dt);
  ierr = xmp_dist_nodes(dt, &dn);
  _XMP_nodes_t *n = (_XMP_nodes_t *)dn;

  ierr=xmp_dist_blocksize(dt,dim,&t_dist_size);
  idistnum=a->info[dim-1].align_subscript/t_dist_size;
 
  format = xmp_align_format(d,dim);
  ierr = xmp_align_axis(d,dim,&idim);

  if (format == _XMP_N_ALIGN_BLOCK){
    ival = index[idim-1]/t_dist_size+idistnum +1;
  }else if (format == _XMP_N_ALIGN_CYCLIC){
    ival = (index[idim-1]/t_dist_size+idistnum)%(n->info[idim-1].size)+1;
  }else if (format == _XMP_N_ALIGN_BLOCK_CYCLIC){
    ival = (index[idim-1]/t_dist_size+idistnum)%(n->info[idim-1].size)+1;
  }else
    ival = -1;

  return ival;
}

int xmp_array_lead_dim(xmp_desc_t d, int size[]){

   int i, ndims, ierr;
  _XMP_array_t *a = (_XMP_array_t *)d;

  ierr = xmp_array_ndims(d, &ndims);
  for (i=0;i<ndims;i++){
    size[i] = a->info[i].par_size;
  }

  return 0;
}

int xmp_align_axis(xmp_desc_t d, int dim, int *axis){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *axis = a->info[dim-1].align_template_index + 1;

  return 0;

}

int xmp_align_offset(xmp_desc_t d, int dim, int *offset){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *offset = a->info[dim-1].align_subscript;

  return 0;

}

int xmp_align_format(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].align_manner;

}

int xmp_align_size(xmp_desc_t d, int dim){

  int format, ival=0, idim, ierr;
  xmp_desc_t dt;

  _XMP_array_t *a = (_XMP_array_t *)d;

  format = xmp_align_format(d,dim);
  ierr = xmp_align_axis(d,dim,&idim);
  ierr = xmp_align_template(d,&dt);

  _XMP_template_t *t = (_XMP_template_t *)dt;

  if (format == _XMP_N_ALIGN_BLOCK){
    ival = t->chunk[idim-1].par_chunk_width;
  }else if (format == _XMP_N_ALIGN_CYCLIC){
    ival = t->chunk[idim-1].par_width;
  }else if (format == _XMP_N_ALIGN_BLOCK_CYCLIC){
    ival = t->chunk[idim-1].par_width;
  }else if (format == _XMP_N_ALIGN_DUPLICATION){
    ival = t->chunk[idim-1].par_chunk_width;
  }else if (format == _XMP_N_ALIGN_NOT_ALIGNED){
    ival = a->info[dim-1].par_size;
  }else
    ival = -1;

  return ival;
}

int xmp_align_template(xmp_desc_t d, xmp_desc_t *dt){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *dt = (xmp_desc_t)(a->align_template);

  return 0;

}

int xmp_template_fixed(xmp_desc_t d, int *fixed){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *fixed = t->is_fixed;

  return 0;

}

int xmp_template_ndims(xmp_desc_t d, int *ndims){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *ndims = t->dim;

  return 0;

}

int xmp_template_lbound(xmp_desc_t d, int dim, int *lbound) {

  _XMP_template_t *t = (_XMP_template_t *)d;

  *lbound = t->info[dim-1].ser_lower;

  return 0;

}

int xmp_template_ubound(xmp_desc_t d, int dim, int *ubound) {

  _XMP_template_t *t = (_XMP_template_t *)d;

  *ubound = t->info[dim-1].ser_upper;

  return 0;

}
int xmp_template_gsize(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->info[dim-1].ser_size;

}

int xmp_template_lsize(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->chunk[dim-1].par_chunk_width;

}

int xmp_dist_format(xmp_desc_t d, int dim, int *format){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *format = t->chunk[dim-1].dist_manner;
  if (*format == _XMP_N_DIST_BLOCK){
    *format = XMP_BLOCK;
  }else if (*format == _XMP_N_DIST_CYCLIC){
    *format = XMP_CYCLIC;
  }else if (*format == _XMP_N_DIST_BLOCK_CYCLIC){
    *format = XMP_CYCLIC;
  }else{
    *format = XMP_NOT_DISTRIBUTED;
  }

  return 0;

}

int xmp_dist_blocksize(xmp_desc_t d, int dim, int *blocksize){

  int format,ierr;

  _XMP_template_t *t = (_XMP_template_t *)d;

  ierr = xmp_dist_format(d,dim,&format);

  if (format == XMP_BLOCK){
    *blocksize = t->chunk[dim-1].par_chunk_width;
  }else if (format == XMP_CYCLIC){
    *blocksize = t->chunk[dim-1].par_width;
  }else if (format == XMP_NOT_DISTRIBUTED){
    *blocksize = t->chunk[dim-1].par_chunk_width;
  }else{
    *blocksize = -1;
  }

  return 0;
}

int xmp_dist_stride(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->chunk[dim-1].par_stride;

}

int xmp_dist_nodes(xmp_desc_t d, xmp_desc_t *dn){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *dn = (xmp_desc_t)(t->onto_nodes);

  return 0;

}

int xmp_nodes_ndims(xmp_desc_t d, int *ndims){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *ndims = n->dim;

  return 0;

}

int xmp_nodes_index(xmp_desc_t d, int dim, int *index){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *index = n->info[dim-1].rank;

  return 0;

}

int xmp_nodes_size(xmp_desc_t d, int dim, int *size){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *size = n->info[dim-1].size;

  return 0;

}

extern void _XMP_sched_loop_template_BLOCK(int, int, int, int *, int *, int *, void *, int);
extern void _XMP_sched_loop_template_CYCLIC(int, int, int, int *, int *, int *, void *, int);
extern void _XMP_sched_loop_template_BLOCK_CYCLIC(int, int, int, int *, int *, int *, void *, int);
void xmp_sched_template_index(int* local_start_index, int* local_end_index, 
			     const int global_start_index, const int global_end_index, const int step, 
			     const xmp_desc_t template, const int template_dim)
{
  int tmp;
  _XMP_template_chunk_t *chunk = &(((_XMP_template_t*)template)->chunk[template_dim]);

/*   if(chunk->dist_manner == NULL){ */
/*     _XMP_fatal("Invalid template descriptor in xmp_sched_template_index()"); */
/*   } */

  switch(chunk->dist_manner){
  case _XMP_N_DIST_BLOCK:
    _XMP_sched_loop_template_BLOCK(global_start_index, global_end_index, step, 
				   local_start_index, local_end_index, &tmp, template, template_dim);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_sched_loop_template_CYCLIC(global_start_index, global_end_index, step, 
				    local_start_index, local_end_index, &tmp, template, template_dim);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC: 
    _XMP_sched_loop_template_BLOCK_CYCLIC(global_start_index, global_end_index, step, 
					  local_start_index, local_end_index, &tmp, template, template_dim);
    break;
  default:
    _XMP_fatal("does not support distribution in xmp_sched_template_index()");
    break;
  }
}


void *xmp_malloc(xmp_desc_t d){

  _XMP_array_t *array_desc = (_XMP_array_t *)d;

  void *array_addr;

  if (!array_desc->is_allocated) {
    return NULL;
  }
  
  _XMP_ASSERT(array_desc->dim == 1);

  unsigned long long total_elmts = 1;
  int ndims = array_desc->dim;
  for (int i = ndims - 1; i >= 0; i--) {
    array_desc->info[i].dim_acc = total_elmts;
    total_elmts *= array_desc->info[i].alloc_size;
  }

  for (int i = 0; i < ndims; i++) {
    _XMP_calc_array_dim_elmts(array_desc, i);
  }

  array_addr = _XMP_alloc(total_elmts * (array_desc->type_size));

  // set members
  array_desc->array_addr_p = array_addr;
  array_desc->total_elmts = total_elmts;

  return array_addr;

}
