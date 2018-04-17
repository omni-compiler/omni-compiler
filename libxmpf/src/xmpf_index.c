#include "xmpf_internal.h"

int xmpf_local_idx__(_XMP_array_t **a_desc, int *i_dim, int *global_idx)
{
    _XMP_array_t *a = *a_desc;
    int l_idx = 0, l_base = 0;
    _XMP_array_info_t *ai = &a->info[*i_dim];
    long long off = ai->align_subscript;
    int tdim = ai->align_template_index;
    int lshadow = ai->shadow_size_lo;

    _XMP_G2L((long long)(*global_idx) + off, &l_idx, a->align_template, tdim);
    _XMP_G2L((long long)ai->par_lower + off, &l_base, a->align_template, tdim);
    l_idx = l_idx - l_base + lshadow;

    return l_idx;
}

/* calc global index by local index */
void xmpf_l2g__(int *global_idx, int *local_idx,
	        int *rdim, _XMP_object_ref_t **r_desc)
{
  long long int g_idx = 0;
  _XMP_object_ref_t *rp = *r_desc;
  int off = rp->REF_OFFSET[*rdim];
  int tdim = rp->REF_INDEX[*rdim];

  _XMP_L2G(*local_idx, &g_idx, rp->t_desc, tdim);

  *global_idx = g_idx - off;  // must be long long
}
