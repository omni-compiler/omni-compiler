/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#include "mpi.h"
//#include "stdbool.h"
#include "xmp_internal.h"
#include "xmp_constant.h"
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

void xmp_array_dim(xmp_desc_t d, int *dim) {

  _XMP_array_t *a = (_XMP_array_t *)d;

  *dim = a->dim;
 
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

//void xmp_array_laddr(xmp_desc_t d, void **laddr){
//}

//void xmp_array_shadow(xmp_desc_t d, int ushadow[], int lshadow[]){
//}

void xmp_array_first_idx_node_index(xmp_desc_t d, int idx[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    idx[i] = 0;
  }
}

void xmp_array_lead_dim(xmp_desc_t d, int *lead_dim){

  _XMP_array_t *a = (_XMP_array_t *)d;

  *lead_dim = a->info[1].par_size;

}

//void xmp_align_axis(xmp_desc_t d, int axis[]){
//}

//void xmp_align_offset(xmp_desc_t d, int offest[]){
//}

//xmp_desc_t *xmp_align_template(xmp_desc_t d){
//}

//bool xmp_template_fixed(xmp_desc_t d){
//}

//void xmp_template_rank(xmp_desc_t d, int *rank){
//}

//void xmp_template_gsize(xmp_desc_t d, int size[]){
//}

//void xmp_template_lsize(xmp_desc_t d, int size[]){
//}

void xmp_dist_format(xmp_desc_t d, int dist_format[]){

  _XMP_array_t *a = (_XMP_array_t *)d;

  for (int i = 0; i < a->dim; i++){
    dist_format[i] = a->align_template->chunk->dist_manner;
  }

}

void xmp_dist_size(xmp_desc_t d, int size[]){

  int *dist_format;

  _XMP_array_t *a = (_XMP_array_t *)d;

  dist_format = (int *)malloc(sizeof(int)* a->dim);

  xmp_dist_format(d,dist_format);

  for (int i = 0; i < a->dim; i++){
     if (dist_format[i] == _XMP_N_DIST_BLOCK){
       size[i] = a->info[i].par_size;
     }else if (dist_format[i] == _XMP_N_DIST_CYCLIC){
       size[i] = a->align_template->chunk->par_width;
     }else if (dist_format[i] == _XMP_N_DIST_BLOCK_CYCLIC){
       size[i] = a->align_template->chunk->par_width;
     }else if (dist_format[i] == _XMP_N_DIST_DUPLICATION){
       size[i] = a->info[i].par_size;
     }
  }

  free(dist_format);

}

//xmp_desc_t *xmp_dist_nodes(xmp_desc_t d){
//}

//void xmp_nodes_rank(xmp_desc_t d, int *rank){
//}

//void xmp_nodes_size(xmp_desc_t d, int size[]){
//}
