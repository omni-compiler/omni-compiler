/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "xmpf_internal.h"
//#include "xmpf.h"


/*
void xmp_MPI_comm(void **comm) {
  *comm = _XMP_get_execution_nodes()->comm;
}
*/

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
  return _XMP_world_rank;
}

double xmp_wtime_(void) {
  return MPI_Wtime();
}

double xmp_wtick_(void) {
  return MPI_Wtick();
}

/*
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

void *xmp_array_laddr(xmp_desc_t d){

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

void xmp_array_first_idx_node_index(xmp_desc_t d, int idx[]){

  int i, a_dim, t_dim, *a_lower, *t_lower, t_dist_size;
  xmp_desc_t dt;

  _XMP_array_t *a = (_XMP_array_t *)d;

  a_dim = a->dim;
  t_dim = a->align_template->dim;
  a_lower = (int *)malloc(sizeof(int)* t_dim);
  t_lower = (int *)malloc(sizeof(int)* t_dim);

  if (t_dim == a_dim){
     for (i = 0; i< t_dim; i++){
       t_lower[i] = a->align_template->chunk[i].par_lower;
       a_lower[i] = a->info[i].par_lower;
     }
  }else{
     for (i = t_dim-1; i>= t_dim - a_dim -1; i--){
       t_lower[i] = a->align_template->chunk[i].par_lower;
       if (i >= t_dim-1-(a_dim-1)){
          a_lower[i] = a->info[i-(t_dim-a_dim)].par_lower;
       } else {
          a_lower[i] = t_lower[i];
       }
     }
  }

  dt = xmp_align_template(d);

  for (i = 0; i< t_dim; i++){
    t_dist_size=xmp_dist_size(dt,i+1);
    idx[i]=(t_lower[i]-a_lower[i])/t_dist_size; 
  }

  free(a_lower);
  free(t_lower);
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
*/
