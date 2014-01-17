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
    *local_idx = global_idx - base;
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


int xmpf_local_idx__(_XMP_array_t **a_desc, int *i_dim, int *global_idx)
{
    _XMP_array_t *a = *a_desc;
    int l_idx, l_base;
    _XMP_array_info_t *ai = &a->info[*i_dim];
    int off = ai->align_subscript;
    int tdim = ai->align_template_index;
    int base = ai->ser_lower;

    // NOTE: base should be cached in the descriptor.

    _XMP_G2L(*global_idx + off, &l_idx, a->align_template, tdim);
    // NOTE: par_lower is 0-origin.
    _XMP_G2L(ai->par_lower + base + off, &l_base, a->align_template, tdim);
    //xmpf_dbg_printf("%d, %d, %d, %d\n", *global_idx, l_idx, base, ai->par_lower);
    l_idx = l_idx - l_base + 1; // the local lower bound is always 1 in Fortran

    return l_idx;
}


/* int xmpf_global_idx__(_XMP_array_t **a_desc, int *i_dim, int *local_idx) */
/* { */
/*     _XMP_array_t *a = *a_desc; */
/*     long long g_idx; */

/*     _XMP_array_info_t ai = &a->info[*i_dim]; */
/*     int off = ai->align_subscript; */
/*     int tdim = ai->align_template_index; */

/*     int ret; */

/*     _XMP_L2G(*local_idx, &g_idx, a->align_template, tdim); */

/*     ret = g_idx - off; */

/*     return ret; */
/* } */


/* calc global index by local index */
void xmpf_l2g__(int *global_idx, int *local_idx,
	        int *rdim, _XMP_object_ref_t **r_desc)
{
  long long int g_idx;
  _XMP_object_ref_t *rp = *r_desc;

/*   int off = rp->offset[*rdim]; */
/*   int tdim = rp->index[*rdim]; */
  int off = rp->REF_OFFSET[*rdim];
  int tdim = rp->REF_INDEX[*rdim];

  _XMP_L2G(*local_idx, &g_idx, rp->t_desc, tdim);

  *global_idx = g_idx - off;  // must be long long

  //xmpf_dbg_printf("L2G(%d->%d)\n",*local_idx,*global_idx);
}
