#include "xmpf_internal.h"

/* index conversion from Local to Global */
void _XMP_L2G(int local_idx, long long int *global_idx,
	      _XMP_template_t *template, int template_index)
{
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_nodes_info_t *n_info = chunk->onto_nodes_info;
  long long base = template->info[template_index].ser_lower;

  switch(chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    *global_idx = base + local_idx ;
    break;
  case _XMP_N_DIST_BLOCK:
    *global_idx = base + n_info->rank*chunk->par_chunk_width+local_idx;
    break;
  case _XMP_N_DIST_CYCLIC:
    *global_idx = base+n_info->size*local_idx;
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    {
      int w = chunk->par_chunk_width;
      *global_idx = base+(local_idx/w)*w*n_info->size+local_idx%w;
    }
    break;
  default:
    _XMP_fatal("_XMP_: unknown chunk dist_manner");
  }
}


/* index conversion from Global to local */
void _XMP_G2L(long long int global_idx,int *local_idx,
	      _XMP_template_t *template, int template_index)
{
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_nodes_info_t *n_info = chunk->onto_nodes_info;
  long long base = template->info[template_index].ser_lower;

  switch(chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    *local_idx = global_idx-base;
    break;
  case _XMP_N_DIST_BLOCK:
    //*local_idx = (global_idx - base)%chunk->par_chunk_width;
    *local_idx = (global_idx - base) - chunk->onto_nodes_info->rank * chunk->par_chunk_width;
    break;
  case _XMP_N_DIST_CYCLIC:
    *local_idx = (global_idx - base)/n_info->size;
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    {
      int off = global_idx-base;
      int w = chunk->par_chunk_width;
      *local_idx = (off/(n_info->size*w))*w+off%w;
    }
    break;
  default:
    _XMP_fatal("_XMP_: unknown chunk dist_manner");
  }
}


int xmpf_local_idx__(_XMP_array_t **a_desc, int *i_dim, int *global_idx)
{
    _XMP_array_t *a = *a_desc;
    int l_idx;
    _XMP_G2L(*global_idx,&l_idx,a->align_template,*i_dim);
    // how to get offset
    return l_idx;
}


/* calc global index by local index */
void xmpf_l2g_(int *global_idx, int *local_idx,
	       _XMP_template_t **t_desc, int *t_idx)
{
  long long int g_idx;
  _XMP_L2G(*local_idx,&g_idx,*t_desc,*t_idx);
  *global_idx = g_idx;  // must be long long
  //xmpf_dbg_printf("L2G(%d->%d)\n",*local_idx,*global_idx);
}
