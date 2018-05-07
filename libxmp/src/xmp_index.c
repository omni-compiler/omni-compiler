#include "xmp_internal.h"

/* index conversion from Local to Global */
void _XMP_L2G(int local_idx, long long int *global_idx,
	      _XMP_template_t *template, int template_index)
{
  if(! template->is_owner){
    return;
  }

  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_nodes_info_t *n_info = chunk->onto_nodes_info;
  long long base = template->info[template_index].ser_lower;

  switch(chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    //*global_idx = base + local_idx ;
    *global_idx = local_idx ;
    break;
  case _XMP_N_DIST_BLOCK:
    *global_idx = base + n_info->rank * chunk->par_chunk_width + local_idx;
    //xmpf_dbg_printf("%d -> %d, base = %d, rank = %d, chunk = %d\n", local_idx, *global_idx, base, n_info->rank, chunk->par_chunk_width);
    break;
  case _XMP_N_DIST_CYCLIC:
    *global_idx = base + n_info->rank + n_info->size * local_idx;
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    {
      int w = chunk->par_width;
      *global_idx = base + n_info->rank * w
	          + (local_idx/w) * w * n_info->size + local_idx%w;
    }
    break;
  case _XMP_N_DIST_GBLOCK:
    *global_idx = local_idx + chunk->mapping_array[n_info->rank];
    break;
  default:
    _XMP_fatal("_XMP_: unknown chunk dist_manner");
  }
}


/* index conversion from Global to local */
void _XMP_G2L(long long int global_idx, int *local_idx,
	      _XMP_template_t *template, int template_index)
{
  _XMP_template_chunk_t *chunk = &(template->chunk[template_index]);
  _XMP_nodes_info_t *n_info = chunk->onto_nodes_info;
  long long base = template->info[template_index].ser_lower;

  // NOTE: local_idx is 0-origin.

  switch(chunk->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    //*local_idx = global_idx - base;
    *local_idx = global_idx;
    break;
  case _XMP_N_DIST_BLOCK:
    *local_idx = (global_idx - base) - n_info->rank * chunk->par_chunk_width;
    break;
  case _XMP_N_DIST_CYCLIC:
    *local_idx = (global_idx - base) / n_info->size;
    break;
  case _XMP_N_DIST_BLOCK_CYCLIC:
    {
      int off = global_idx - base;
      int w = chunk->par_width;
      *local_idx = (off / (n_info->size*w)) * w + off%w;
    }
    break;
  case _XMP_N_DIST_GBLOCK:
    *local_idx = global_idx - chunk->mapping_array[n_info->rank];
    break;
  default:
    _XMP_fatal("_XMP_: unknown chunk dist_manner");
  }
}
