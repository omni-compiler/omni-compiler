/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "mpi.h"
#include "stdlib.h"
#include "xmp_internal.h"
#include "xmp.h"

// FIXME utility functions
void xmp_MPI_comm(void **comm) {
  *comm = _XMP_get_execution_nodes()->comm;
}

int xmp_num_nodes(void) {
  return _XMP_get_execution_nodes()->comm_size;
}

int xmp_node_num(void) {
  return _XMP_get_execution_nodes()->comm_rank;
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

void xmp_array_ndim(xmp_desc_t d, int *ndim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  *ndim = a->dim;
 
}
void xmp_array_gsize(xmp_desc_t d, int size[]) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    size[i] = a->info[i].ser_size;
  }

}

void xmp_array_lsize(xmp_desc_t d, int size[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    size[i] = a->info[i].par_size;
  }

}

void xmp_array_laddr(xmp_desc_t d, void **laddr){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *laddr = (void *)a->array_addr_p;

}

void xmp_array_shadow(xmp_desc_t d, int ushadow[], int lshadow[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    ushadow[i] = a->info[i].shadow_size_hi;
    lshadow[i] = a->info[i].shadow_size_lo;
  }

}

void xmp_array_first_idx_node_index(xmp_desc_t d, int idx[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    idx[i] = 0;
  }
}

void xmp_array_lead_dim(xmp_desc_t d, int *lead_dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  if (a->dim == 1){
     *lead_dim = a->info[0].par_size;
  } else if (a->dim > 1){
     *lead_dim = a->info[a->dim -1].par_size;
  }

}

void xmp_align_axis(xmp_desc_t d, int axis[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    axis[i] = a->info[i].align_template_index;
  }

}

void xmp_align_offset(xmp_desc_t d, int offset[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    offset[i] = a->info[i].align_subscript;
  }

}

xmp_desc_t xmp_align_template(xmp_desc_t d){

  _XMP_array_t *a = (_XMP_array_t *)d;

  return (xmp_desc_t)(a->align_template);

}

_Bool xmp_template_fixed(xmp_desc_t d){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return t->is_fixed;

}

void xmp_template_ndim(xmp_desc_t d, int *ndim){

  _XMP_template_t *t = (_XMP_template_t *)d;

  *ndim = t->dim;

}

void xmp_template_gsize(xmp_desc_t d, int size[]){

  _XMP_template_t *t = (_XMP_template_t *)d;

  for (int i = 0; i < t->dim; i++){
    size[i] = t->info[i].ser_size;
  }

}

void xmp_template_lsize(xmp_desc_t d, int size[]){

  _XMP_template_t *t = (_XMP_template_t *)d;

  for (int i = 0; i < t->dim; i++){
    size[i] = t->chunk[i].par_chunk_width;
  }

}

void xmp_dist_format(xmp_desc_t d, int dist_format[]){

  _XMP_template_t *t = (_XMP_template_t *)d;

  for (int i = 0; i < t->dim; i++){
    dist_format[i] = t->chunk[i].dist_manner;
  }

}

void xmp_dist_size(xmp_desc_t d, int size[]){

  int *dist_format;

  _XMP_template_t *t = (_XMP_template_t *)d;

  dist_format = (int *)malloc(sizeof(int)* t->dim);

  xmp_dist_format(d,dist_format);

  for (int i = 0; i < t->dim; i++){
     if (dist_format[i] == _XMP_N_DIST_BLOCK){
       size[i] = t->chunk[i].par_chunk_width;
     }else if (dist_format[i] == _XMP_N_DIST_CYCLIC){
       size[i] = t->chunk[i].par_width;
     }else if (dist_format[i] == _XMP_N_DIST_BLOCK_CYCLIC){
       size[i] = t->chunk[i].par_width;
     }else if (dist_format[i] == _XMP_N_DIST_DUPLICATION){
       size[i] = t->chunk[i].par_chunk_width;
     }
  }

  free(dist_format);

}

xmp_desc_t xmp_dist_nodes(xmp_desc_t d){

  _XMP_template_t *t = (_XMP_template_t *)d;

  return (xmp_desc_t)(t->onto_nodes);

}

void xmp_nodes_ndim(xmp_desc_t d, int *ndim){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

    *ndim = n->dim;

}

void xmp_nodes_index(xmp_desc_t d, int idx[]){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  for (int i = 0; i < n->dim; i++){
    idx[i] = n->info[i].rank;
  }

}

void xmp_nodes_size(xmp_desc_t d, int size[]){

  _XMP_nodes_t *n = (_XMP_nodes_t *)d;

  for (int i = 0; i < n->dim; i++){
    size[i] = n->info[i].size;
  }

}
