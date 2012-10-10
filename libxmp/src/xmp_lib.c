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

int xmp_array_ndim(xmp_desc_t d) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->dim;
 
}

size_t xmp_array_type_size(xmp_desc_t d) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->type_size;
 
}

int xmp_array_gsize(xmp_desc_t d, int dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].ser_size;

}

int xmp_array_lsize(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].par_size;

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

void **xmp_array_laddr(xmp_desc_t d){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return (void *)a->array_addr_p;

}

int xmp_array_ushadow(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].shadow_size_hi;

}

int xmp_array_lshadow(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].shadow_size_lo;

}

int xmp_array_owner(xmp_desc_t d, int ndims, int index[ndims], int dim){

  int idim, ival, idistnum, t_dist_size, format;
  xmp_desc_t dt, dn;

  _XMP_array_t *a = (_XMP_array_t *)d;

  dt = xmp_align_template(d);
  dn = xmp_dist_nodes(dt);
  _XMP_nodes_t *n = (_XMP_nodes_t *)dn;

  t_dist_size=xmp_dist_size(dt,dim);
  idistnum=a->info[dim-1].align_subscript/t_dist_size;
 
  format = xmp_align_format(d,dim);
  idim = xmp_align_axis(d,dim);

  if (format == _XMP_N_ALIGN_BLOCK){
    ival = index[idim-1]/t_dist_size+idistnum;
  }else if (format == _XMP_N_ALIGN_CYCLIC){
    ival = (index[idim-1]/t_dist_size+idistnum)%(n->info[idim-1].size);
  }else if (format == _XMP_N_ALIGN_BLOCK_CYCLIC){
    ival = (index[idim-1]/t_dist_size+idistnum)%(n->info[idim-1].size);
  }else
    ival = -1;

  return ival;
}

int xmp_array_lead_dim(xmp_desc_t d){

   int ival = 0;
  _XMP_array_t *a = (_XMP_array_t *)d;

  if (a->dim == 1){
     ival = a->info[0].par_size;
  } else if (a->dim > 1){
     ival = a->info[a->dim -1].par_size;
  }

  return ival;
}

int xmp_align_axis(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].align_template_index + 1;

}

int xmp_align_offset(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].align_subscript;

}

int xmp_align_format(xmp_desc_t d, int dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return a->info[dim-1].align_manner;

}

int xmp_align_size(xmp_desc_t d, int dim){

  int format, ival=0, idim;
  xmp_desc_t dt;

  _XMP_array_t *a = (_XMP_array_t *)d;

  format = xmp_align_format(d,dim);
  idim = xmp_align_axis(d,dim);
  dt = xmp_align_template(d);

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

xmp_desc_t xmp_align_template(xmp_desc_t d){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return (xmp_desc_t)(a->align_template);

}

_Bool xmp_template_fixed(xmp_desc_t d){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->is_fixed;

}

int xmp_template_ndim(xmp_desc_t d){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->dim;

}

int xmp_template_gsize(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->info[dim-1].ser_size;

}

int xmp_template_lsize(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->chunk[dim-1].par_chunk_width;

}

int xmp_dist_format(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->chunk[dim-1].dist_manner;

}

int xmp_dist_size(xmp_desc_t d, int dim){

  int format,ival=0;

  _XMP_template_t *t = (_XMP_template_t *)d;

  format = xmp_dist_format(d,dim);

  if (format == _XMP_N_DIST_BLOCK){
    ival = t->chunk[dim-1].par_chunk_width;
  }else if (format == _XMP_N_DIST_CYCLIC){
    ival = t->chunk[dim-1].par_width;
  }else if (format == _XMP_N_DIST_BLOCK_CYCLIC){
    ival = t->chunk[dim-1].par_width;
  }else if (format == _XMP_N_DIST_DUPLICATION){
    ival = t->chunk[dim-1].par_chunk_width;
  }else{
    ival = -1;
  }

  return ival;
}

int xmp_dist_stride(xmp_desc_t d, int dim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->chunk[dim-1].par_stride;

}

xmp_desc_t xmp_dist_nodes(xmp_desc_t d){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return (xmp_desc_t)(t->onto_nodes);

}

int xmp_nodes_ndim(xmp_desc_t d){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

    return n->dim;

}

int xmp_nodes_index(xmp_desc_t d, int dim){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  return n->info[dim-1].rank;

}

int xmp_nodes_size(xmp_desc_t d, int dim){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  return n->info[dim-1].size;

}

extern void _XMP_sched_loop_template_BLOCK(int, int, int, int *, int *, int *, void *, int);
extern void _XMP_sched_loop_template_CYCLIC(int, int, int, int *, int *, int *, void *, int);
extern void _XMP_sched_loop_template_BLOCK_CYCLIC(int, int, int, int *, int *, int *, void *, int);
void xmp_sched_template_index(int* local_start_index, int* local_end_index, 
			     const int global_start_index, const int global_end_index, const int step, 
			     const xmp_desc_t template, const int template_index)
{
  int tmp;
  _XMP_template_chunk_t *chunk = &(((_XMP_template_t*)template)->chunk[template_index]);

  if(chunk->dist_manner == NULL){
    _XMP_fatal("Invalid template descriptor in xmp_sched_template_index()");
  }

  switch(chunk->dist_manner){
  case _XMP_N_DIST_BLOCK:
    _XMP_sched_loop_template_BLOCK(global_start_index, global_end_index, step, 
				   local_start_index, local_end_index, &tmp, template, template_index);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_sched_loop_template_CYCLIC(global_start_index, global_end_index, step, 
				    local_start_index, local_end_index, &tmp, template, template_index);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC: 
    _XMP_sched_loop_template_BLOCK_CYCLIC(global_start_index, global_end_index, step, 
					  local_start_index, local_end_index, &tmp, template, template_index);
    break;
  default:
    _XMP_fatal("does not support distribution in xmp_sched_template_index()");
    break;
  }
}
