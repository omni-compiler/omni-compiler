/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "stdlib.h"
#include "xmp_internal.h"
#include "xmp.h"
#include <stddef.h>

MPI_Comm xmp_get_mpi_comm(void) {
  MPI_Comm *comm;
  comm=_XMP_get_execution_nodes()->comm;
  return *comm;
}

void xmp_init_mpi(int *argc, char ***argv) {
}

void xmp_finalize_mpi(void) {
}

void xmp_init(int *argc, char ***argv) {
  _XMP_init(*argc, *argv);
}

void xmp_finalize(void) {
  _XMP_finalize(0);
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
  return _XMP_world_rank + 1;
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

  int idim, ival, idistnum, t_dist_size, format;
  xmp_desc_t dt, dn;

  _XMP_array_t *a = (_XMP_array_t *)d;

  xmp_align_template(d, &dt);
  xmp_dist_nodes(dt, &dn);
  _XMP_nodes_t *n = (_XMP_nodes_t *)dn;

  xmp_dist_blocksize(dt,dim,&t_dist_size);
  idistnum=a->info[dim-1].align_subscript/t_dist_size;
 
  format = xmp_align_format(d,dim);
  xmp_align_axis(d,dim,&idim);

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

   int i, ndims;
  _XMP_array_t *a = (_XMP_array_t *)d;

  xmp_array_ndims(d, &ndims);
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

  int format, ival=0, idim;
  xmp_desc_t dt;

  _XMP_array_t *a = (_XMP_array_t *)d;

  format = xmp_align_format(d,dim);
  xmp_align_axis(d,dim,&idim);
  xmp_align_template(d,&dt);

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

int xmp_align_replicated(xmp_desc_t d, int dim, int *replicated){

  _XMP_array_t *a = (_XMP_array_t *)d;

  int andims, nndims, axis, counter=0; 
  xmp_desc_t dt, dn;

  xmp_align_template(d, &dt);
  xmp_dist_nodes(dt, &dn);
  xmp_array_ndims(d, &andims);
  xmp_nodes_ndims(dn, &nndims);

  for(int i=0; i<andims; i++){
    xmp_align_axis(d, i+1, &axis);
    if (axis <= 0){
      counter = counter +1;
    }
  }

  if (counter != nndims){
    *replicated=1;
    for(int i=0; i<andims; i++){
      xmp_align_axis(d, i+1, &axis);
      if (dim == axis){
        *replicated=0;
        break;
      }
    }
  }else{
    *replicated=0;
  }

  return 0;

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

  int format;

  _XMP_template_t *t = (_XMP_template_t *)d;

  xmp_dist_format(d,dim,&format);

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

int xmp_dist_axis(xmp_desc_t d, int dim, int *axis){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *axis = t->chunk[dim-1].onto_nodes_index + 1;

  return 0;

}

int xmp_dist_gblockmap(xmp_desc_t d, int dim, int *map){

  _XMP_template_t *t = (_XMP_template_t *)d;
  _XMP_template_chunk_t *chunk = &(t->chunk[dim-1]);

  int axis, size;
  xmp_desc_t dn;
  xmp_dist_nodes(d, &dn);
  xmp_dist_axis(d, dim, &axis);
  xmp_nodes_size(dn, axis, &size);

  for (int i=0; i<size-1; i++){
    map[i] = chunk->mapping_array[i+1]-chunk->mapping_array[i];
  }
  if (size > 1){
    map[size-1] = t->info[dim-1].ser_size - map[size-2];
  }else if (size == 1){
    map[0] =  t->info[dim-1].ser_size;
  }else{
    return -1;
  }

  return 0;

}

int xmp_nodes_ndims(xmp_desc_t d, int *ndims){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *ndims = n->dim;

  return 0;

}

int xmp_nodes_index(xmp_desc_t d, int dim, int *index){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *index = n->info[dim-1].rank + 1;

  return 0;

}

int xmp_nodes_size(xmp_desc_t d, int dim, int *size){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *size = n->info[dim-1].size;

  return 0;

}

int xmp_nodes_rank(xmp_desc_t d, int *rank){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *rank = n->comm_rank + 1;

  return 0;

}

int xmp_nodes_comm(xmp_desc_t d, void **comm){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *(void **)comm = (void *)n->comm;

  return 0;

}

int xmp_nodes_equiv(xmp_desc_t d, xmp_desc_t *dn, int lb[], int ub[], int st[]){

  int i, ndims;
  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  *dn = (xmp_desc_t)(n->inherit_nodes);
  if (*dn != NULL){
    xmp_nodes_ndims(*dn, &ndims);

    for (i=0; i<ndims; i++){
      lb[i]= n ->inherit_info[i].lower+1;
      ub[i]= n ->inherit_info[i].upper+1;
      st[i]= n ->inherit_info[i].stride;
    }
    return 0;

  }else{
    return -1;
  }
}
/* extern void _XMP_sched_loop_template_BLOCK(int, int, int, int *, int *, int *, void *, int); */
/* extern void _XMP_sched_loop_template_CYCLIC(int, int, int, int *, int *, int *, void *, int); */
/* extern void _XMP_sched_loop_template_BLOCK_CYCLIC(int, int, int, int *, int *, int *, void *, int); */
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


void *xmp_malloc(xmp_desc_t d, int size){

  _XMP_array_t *a = (_XMP_array_t *)d;

  _XMP_ASSERT(a->dim == 1);

  _XMP_array_info_t *ai = &(a->info[0]);

  _XMP_template_t *t = a->align_template;

  if (!t->is_fixed) _XMP_fatal("target template is not fixed");

  int tdim = ai->align_template_index;
  _XMP_template_info_t *info = &(t->info[tdim]);

  a->is_allocated = t->is_owner;

  /* Now, normalize align_subscript and size */
  size += (ai->align_subscript - info->ser_lower);
  ai->align_subscript = info->ser_lower;

  ai->ser_upper = size - 1;
  ai->ser_size = size;

  switch (t->chunk[tdim].dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    _XMP_align_array_DUPLICATION(a, 0, ai->align_template_index, ai->align_subscript);
    break;
  case _XMP_N_DIST_BLOCK:
    _XMP_align_array_BLOCK(a, 0, ai->align_template_index, ai->align_subscript, ai->temp0);
    break;
  case _XMP_N_DIST_CYCLIC:
    _XMP_align_array_CYCLIC(a, 0, ai->align_template_index, ai->align_subscript, ai->temp0);
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    _XMP_align_array_BLOCK_CYCLIC(a, 0, ai->align_template_index, ai->align_subscript, ai->temp0);
    break;
  case _XMP_N_DIST_GBLOCK:
    _XMP_align_array_GBLOCK(a, 0, ai->align_template_index, ai->align_subscript, ai->temp0);
    break;
  default:
    _XMP_fatal("unknown distribution manner 1");
    return NULL;
  }

  int ntdims = t->dim;
  int args[ntdims];
  for (int i = 0; i < ntdims; i++) args[i] = 1;
  args[tdim] = 0;
  _XMP_init_array_comm2(a, args);

  _XMP_init_array_nodes(a);

  _XMP_init_shadow(a, ai->shadow_type, ai->shadow_size_lo, ai->shadow_size_hi);

  void *array_addr;
  _XMP_alloc_array(&array_addr, a, (unsigned long long *)a->array_addr_p);

  return array_addr;
}


void xmp_free(xmp_desc_t d){
  if (((_XMP_array_t *)d)->is_allocated)
    _XMP_dealloc_array((_XMP_array_t *)d);
}


void xmp_exit(int status){
  _XMP_finalize(0);
  exit(status);
}
