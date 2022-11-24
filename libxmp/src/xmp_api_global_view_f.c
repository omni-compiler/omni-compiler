#include <stdarg.h>
#include "xmp_api.h"
#include "xmp_internal.h"

 /* CALL xmpf_array_alloc_ ( XMP_DESC_u , 2 , 514 , 0 , XMP_DESC_t ) */
 /* CALL xmp_f_init_allocated_ ( XMP_DESC_u ) */
 /* CALL xmpf_align_info_ ( XMP_DESC_u , 0 , 1 , 100 , 0 , 0 ) */
 /* CALL xmpf_align_info_ ( XMP_DESC_u , 1 , 1 , 200 , 1 , 0 ) */
 /* CALL xmpf_array_init_ ( XMP_DESC_u ) */
 /* CALL xmpf_array_get_local_size_off_ ( XMP_DESC_u , 0 , XMP_DESC_u_size_0 , XMP_DESC_u_off_0 , XMP_DESC_u_blkoff_0 ) */
 /* CALL xmpf_array_get_local_size_off_ ( XMP_DESC_u , 1 , XMP_DESC_u_size_1 , XMP_DESC_u_off_1 , XMP_DESC_u_blkoff_1 ) */
 /* ALLOCATE ( XMP__u ( 0 : XMP_DESC_u_size_0 - 1 , 0 : XMP_DESC_u_size_1 - 1 ) ) */
 /* CALL xmpf_array_set_local_array_ ( XMP_DESC_u , XMP__u , 0 ) */

extern void xmpf_array_alloc__(_XMP_array_t **a_desc, int *n_dim, int *type, size_t *type_size,
			       _XMP_template_t **t_desc);
extern void xmp_f_init_allocated__(_XMP_array_t **a_desc);

void xmp_new_array_(xmp_desc_t *a_desc, xmp_desc_t *t_desc, int *type, int *n_dims,
		    long long int *dim_lb, long long int *dim_ub)
{
  size_t type_size = 0;

  // for(int i = 0; i < *n_dims; i++) printf("dim(i=%d)[%lld:%lld]\n",i,dim_lb[i],dim_ub[i]);
  xmpf_array_alloc__((_XMP_array_t **)a_desc, n_dims, type, &type_size,(_XMP_template_t **) t_desc);
  xmp_f_init_allocated__((_XMP_array_t **)a_desc);
  {
    _XMP_array_t *a = (_XMP_array_t *)*a_desc;
    for(int i = 0; i < *n_dims; i++){
      _XMP_array_info_t *ai = &(a->info[i]);
      ai->ser_lower = dim_lb[i];
      ai->ser_upper = dim_ub[i];
    }
  }
}

extern void xmpf_align_info__(_XMP_array_t **a_desc, int *a_idx, 
			      int *lower, int *upper, int *t_idx, int *off);

void xmp_align_array_(xmp_desc_t *a_desc, int *array_dim_idx, int *template_dim_idx, int *offset,
		      int *status)
{
  int a_idx = *array_dim_idx - 1;
  int t_idx = *template_dim_idx - 1;
  _XMP_array_t *a = (_XMP_array_t *)*a_desc;
  _XMP_array_info_t *ai = &(a->info[a_idx]);
  xmpf_align_info__((_XMP_array_t **)a_desc, &a_idx,
		    &ai->ser_lower, &ai->ser_upper,
		    &t_idx, offset);
  *status =  XMP_SUCCESS;
}

extern void xmpf_array_set_local_array__(_XMP_array_t **a_desc, void **array_adr, int *is_coarray);

void xmp_allocate_array_(xmp_desc_t *a_desc, void **addr, int *status)
{
  int is_corray = 0;
  _XMP_init_array_nodes(*a_desc);
  xmpf_array_set_local_array__((_XMP_array_t **)a_desc, *addr, &is_corray);
  *status = XMP_SUCCESS;
}

extern void xmpf_array_get_local_size_off__(_XMP_array_t **a_desc, int *i_dim,
					    int *size, int *off, int *blk_off);

void xmp_get_array_local_dim_(xmp_desc_t *a_desc, int *dim_lb, int *dim_ub, int *status)
{
  _XMP_array_t *a = (_XMP_array_t *)*a_desc;
  int i, size, off,blkoff;
  
  for(i = 0; i < a->dim; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    xmpf_array_get_local_size_off__((_XMP_array_t **)a_desc,&i,&size,&off,&blkoff);    
    dim_lb[i] = -ai->shadow_size_lo;
    dim_ub[i] = size - 1 - ai->shadow_size_lo;
    
  }
  *status = XMP_SUCCESS;
}
