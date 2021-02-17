#include "xmp_api.h"
#include "xmp_internal.h"

/* allcoate range structure */
xmp_array_section_t *xmp_new_array_section(int n_dims)
{
  xmp_array_section_t *ap;
  int i;
  
  /* if(n_dims <=0) ... */
  ap = (xmp_array_section_t *)_XMP_alloc(sizeof(struct _xmp_array_section_t)
					 +(n_dims-1)*sizeof(struct _xmp_asection_triplet));
  ap->n_dims = n_dims;
  for(i = 0; i < n_dims; i++){
    ap->dim_info[i].start = 0;
    ap->dim_info[i].length = 0;
    ap->dim_info[i].stride = 0;
  }
  return ap;
}

int xmp_array_section_set_info(xmp_array_section_t *ap, int dim_idx,
			       long start, long length)
{
  if(ap == NULL) return XMP_ERR_ARG;
  if(dim_idx < 0 || dim_idx >= ap->n_dims) return XMP_ERR_DIMS;
  ap->dim_info[dim_idx].start = start;
  ap->dim_info[dim_idx].length = length;
  ap->dim_info[dim_idx].stride = 1;
  return XMP_SUCCESS;
}

int xmp_array_section_set_triplet(xmp_array_section_t *ap, int dim_idx,  
				  long start, long length, int stride)
{
  if(ap == NULL) return XMP_ERR_ARG;
  if(dim_idx < 0 || dim_idx >= ap->n_dims) return XMP_ERR_DIMS;
  ap->dim_info[dim_idx].start = start;
  ap->dim_info[dim_idx].length = length;
  ap->dim_info[dim_idx].stride = stride;
  return XMP_SUCCESS;
}

void xmp_free_array_section(xmp_array_section_t *ap)
{
  _XMP_free(ap);
}

xmp_local_array_t *xmp_new_local_array(size_t elmt_size, int n_dims, long dim_size[], void *loc)
{
  xmp_local_array_t *ap;
  int i;
  
  ap = (xmp_local_array_t *)_XMP_alloc(sizeof(struct _xmp_local_array_t));
  ap->dim_size = (long *)_XMP_alloc(sizeof(long)*n_dims);
  ap->n_dims = n_dims;
  for(i = 0; i < n_dims; i++){
    ap->dim_size[i] = dim_size[i];
  }
  ap->dim_f_offset = NULL;
  ap->addr = loc;
  return ap;
}

void xmp_free_local_array(xmp_local_array_t *ap)
{
  _XMP_free(ap->dim_size);
  _XMP_free(ap);
}

/* 
 * Fortran interface
 */
void xmp_new_array_section_(xmp_desc_t *desc,int *_n_dims)
{
  xmp_array_section_t *ap;
  ap = xmp_new_array_section(*_n_dims);
  *desc = (xmp_array_section_t *)ap;
}

void xmp_array_section_set_info_(xmp_desc_t *desc, int *_dim_idx,
				 long *_start, long *_end, int *status)
{
  xmp_array_section_t *ap = (xmp_array_section_t *)*desc;
  int dim_idx = *_dim_idx; // start from 1 in fortran
  int idx;
  
  if(ap == NULL){
    *status = XMP_ERR_ARG;
    return;
  }
  if(dim_idx <= 0 || dim_idx > ap->n_dims){
    *status = XMP_ERR_DIMS;
    return;
  }
  idx = ap->n_dims - dim_idx;

  /* (3:4) = [3:2], len=2=4-3+1*/
  ap->dim_info[idx].start = *_start;
  ap->dim_info[idx].length = *_end - *_start + 1;
  ap->dim_info[idx].stride = 1;
  *status = XMP_SUCCESS;
  return;
}

void xmp_array_section_set_triplet_(xmp_desc_t *desc, int *_dim_idx,
				    long *_start, long *_end, int *_stride, int *status)
{
  xmp_array_section_t *ap = (xmp_array_section_t *)*desc;
  int dim_idx = *_dim_idx; // start from 1 in fortran
  int idx;
  
  if(ap == NULL){
    *status = XMP_ERR_ARG;
    return;
  }
  if(dim_idx <= 0 || dim_idx > ap->n_dims){
    *status = XMP_ERR_DIMS;
    return;
  }
  idx = ap->n_dims - dim_idx;

  /* (1:6:2) = (1), (3), (5) = [1:3:2] 3=(6-1+1)/2 */
  ap->dim_info[idx].start = *_start;
  ap->dim_info[idx].length = (*_end - *_start + 1)/(*_stride);
  ap->dim_info[idx].stride = *_stride;

  /* printf("triplet[%d]=(%ld,%ld,%ld) <- (%ld,%ld,%d)\n",idx, */
  /* 	 ap->dim_info[idx].start,ap->dim_info[idx].length, ap->dim_info[idx].stride, */
  /* 	 *_start,*_end,*_stride); */

  *status = XMP_SUCCESS;
  return;
}

void xmp_free_array_section_(xmp_array_section_t **ap)
{
  xmp_free_array_section(*ap);
}

void xmp_new_local_array_(xmp_desc_t *desc, size_t *_elmt_size,
			  int *_n_dims, long *dim_lb, long *dim_ub, void *loc)
{
  xmp_local_array_t *ap;
  int i, n_dims;
  
  n_dims = *_n_dims;
  ap = (xmp_local_array_t *)_XMP_alloc(sizeof(struct _xmp_local_array_t));
  ap->dim_size = (long *)_XMP_alloc(sizeof(long)*n_dims);
  ap->dim_f_offset = (long *)_XMP_alloc(sizeof(long)*n_dims);
  ap->n_dims = n_dims;

  for(i = 0; i < n_dims; i++){   /* reverse order */
    ap->dim_f_offset[i] = dim_lb[n_dims-1-i];
    ap->dim_size[i] = dim_ub[n_dims-1-i] - dim_lb[n_dims-1-1] + 1;
  }
  ap->addr = loc;
  *desc = (xmp_desc_t *)ap;
}

void xmp_free_local_array_(xmp_local_array_t **app)
{
  xmp_local_array_t *ap = *app;
  _XMP_free(ap->dim_size);
  _XMP_free(ap->dim_f_offset);
  _XMP_free(ap);
}

/* init/finalize */
void xmp_init_all_()
{
  char *dummy_argv[1];
  dummy_argv[0] = NULL;
  xmp_init_all(0,dummy_argv);
}

void xmp_finalize_all_()
{
  xmp_finalize_all();
}

/* dummy functions */
void xmpc_traverse_init() { }
void xmpc_traverse_finalize() { }

