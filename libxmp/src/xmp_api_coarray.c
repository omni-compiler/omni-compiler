#include "xmp_api.h"
#include "xmp_data_struct.h"
#include "xmp_func_decl.h"                                                                    
#include <stdio.h>

int _xmp_coarray_put_get(int is_put, int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp,
			 xmp_local_array_t *local_ap, xmp_desc_t local_desc, xmp_array_section_t *local_asp);

int _xmp_coarray_put_get_scalar(int is_put, int img_dims[], xmp_desc_t remote_desc, 
				xmp_array_section_t *remote_asp,void *addr);

/* allocate coarray */
xmp_desc_t xmp_new_coarray(int elmt_size, int ndims, long dim_size[],
			  int img_ndims, int img_dim_size[], void **loc)
{
  xmp_desc_t desc;
  
  _XMP_coarray_malloc_info_n(dim_size, ndims, elmt_size);
  _XMP_coarray_malloc_image_info_n(img_dim_size,img_ndims);
  _XMP_coarray_malloc(&desc,loc);
  ((_XMP_coarray_t *)desc)->f_coarray_offset = NULL;
  return desc;
}

int _xmp_coarray_put_get(int is_put, int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp,
			 xmp_local_array_t *local_ap, xmp_desc_t local_desc, xmp_array_section_t *local_asp)
{
  int i, n_dims, element_size,i_distance;
  _XMP_coarray_t *remote_cp = (_XMP_coarray_t *)remote_desc;
  _XMP_coarray_t *local_cp = (_XMP_coarray_t *)local_desc;
  long start[_XMP_N_MAX_DIM], length[_XMP_N_MAX_DIM], stride[_XMP_N_MAX_DIM];
  long elmts[_XMP_N_MAX_DIM], distance[_XMP_N_MAX_DIM];
  int img_n[_XMP_N_MAX_DIM];
  void *a_addr;

  if(remote_cp == NULL) return XMP_ERR_ARG;
  element_size = remote_cp->elmt_size;
  n_dims = remote_cp->coarray_dims; // number of dimensions
  
  // _XMP_coarray_rdma_coarray_set_n
  for(i = 0; i < n_dims ; i++){
    start[i] = remote_asp->dim_info[i].start;
    length[i] = remote_asp->dim_info[i].length;
    stride[i] = remote_asp->dim_info[i].stride;
  }
  if(remote_cp->f_coarray_offset != NULL){
    for(i = 0; i < n_dims ; i++)
      start[i] -= remote_cp->f_coarray_offset[i];
  }
  /* for(i=0; i < n_dims;i++) printf("remote_co[%d]=(%ld,%ld,%ld)\n",i,start[i],length[i], stride[i]); */

  _XMP_coarray_rdma_coarray_set_n(n_dims,start,length,stride);
  
  // printf("_XMP_coarray_rdma_coarray_set_n ...\n");

  // _XMP_coarray_rdma_array_set_n
  if(local_ap == NULL){ // local array is coarray
    if(local_cp == NULL) return XMP_ERR_ARG;
    a_addr = local_cp->real_addr;
    n_dims = local_cp->coarray_dims;
    for(i=0; i < n_dims; i++){
      distance[i] = local_cp->distance_of_coarray_elmts[i];
      elmts[i] = local_cp->coarray_elmts[i];
      start[i] = local_asp->dim_info[i].start;
      length[i] = local_asp->dim_info[i].length;
      stride[i] = local_asp->dim_info[i].stride;

    }

    if(local_cp->f_coarray_offset != NULL){
      for(i = 0; i < n_dims; i++)
	start[i] -= local_cp->f_coarray_offset[i];
    }

    /* for(i=0; i < n_dims;i++) */
    /*   printf("local_co[%d]=(%ld,%ld,%ld,%ld,%ld)\n",i, */
    /* 	     start[i],length[i], stride[i],elmts[i],distance[i]); */
  } else {
    a_addr = local_ap->addr;
    n_dims = local_ap->n_dims;
    for(i=0; i < n_dims; i++){
      elmts[i] = local_ap->dim_size[i];
      start[i] = local_asp->dim_info[i].start;
      length[i] = local_asp->dim_info[i].length;
      stride[i] = local_asp->dim_info[i].stride;

    }

    if(local_ap->dim_f_offset != NULL){
      for(i = 0; i < n_dims; i++)
	start[i] -= local_ap->dim_f_offset[i];
    }
    
    i_distance = element_size;
    for(i=n_dims-1;i >=0; i--){
      distance[i] = i_distance;
      i_distance = i_distance*local_ap->dim_size[i];
    }
  }

  _XMP_coarray_rdma_array_set_n(n_dims,start,length,stride,elmts,distance);

  // printf("_XMP_coarray_rdma_array_set_n ...\n");

  // _XMP_rdma_image_set_n
  n_dims = remote_cp->image_dims;
  for(i = 0; i < n_dims; i++)
    img_n[i] = img_dims[i];
  
  if(remote_cp->f_coarray_offset != NULL){
    for(i = 0; i < n_dims; i++) img_n[i] += -1;
  }
  /* for(i = 0; i < n_dims; i++) printf("img_dims[%d]=%d\n",i,img_n[i]); */
  
  _XMP_coarray_rdma_image_set_n(n_dims,img_n);
  
  if(is_put) _XMP_coarray_put(remote_desc,a_addr,local_desc);
  else _XMP_coarray_get(remote_desc,a_addr,local_desc);

  return XMP_SUCCESS;
}

int xmp_coarray_put(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
		    xmp_desc_t local_desc, xmp_array_section_t *local_asp)
{
  return _xmp_coarray_put_get(TRUE, img_dims, remote_desc, remote_asp,
			      NULL, local_desc, local_asp);
}

int xmp_coarray_put_local(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			  xmp_local_array_t *local_ap, xmp_array_section_t *local_asp)
{
  return _xmp_coarray_put_get(TRUE, img_dims, remote_desc, remote_asp,
			      local_ap, NULL, local_asp);
}

int xmp_coarray_get(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
		    xmp_desc_t local_desc, xmp_array_section_t *local_asp)
{
  return _xmp_coarray_put_get(FALSE, img_dims, remote_desc, remote_asp, NULL, local_desc, local_asp);
}

int xmp_coarray_get_local(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			  xmp_local_array_t *local_ap, xmp_array_section_t *local_asp)
{
  return _xmp_coarray_put_get(FALSE, img_dims, remote_desc, remote_asp, local_ap, NULL, local_asp);
}

int _xmp_coarray_put_get_scalar(int is_put, int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp,
				 void *addr)
{
  int i, n_dims, element_size;
  _XMP_coarray_t *remote_cp = (_XMP_coarray_t *)remote_desc;
  long start[_XMP_N_MAX_DIM], length[_XMP_N_MAX_DIM], stride[_XMP_N_MAX_DIM];
  long elmts[_XMP_N_MAX_DIM], distance[_XMP_N_MAX_DIM];
  int img_n[_XMP_N_MAX_DIM];
  void *a_addr;

  if(remote_cp == NULL) return XMP_ERR_ARG;
  element_size = remote_cp->elmt_size;
  n_dims = remote_cp->coarray_dims; // number of dimensions
  
  // _XMP_coarray_rdma_coarray_set_n
  for(i = 0; i < n_dims ; i++){
    start[i] = remote_asp->dim_info[i].start;
    length[i] = 1;
    stride[i] = 1;
  }
  if(remote_cp->f_coarray_offset != NULL){
    for(i = 0; i < n_dims ; i++)
      start[i] -= remote_cp->f_coarray_offset[i];
  }

  _XMP_coarray_rdma_coarray_set_n(n_dims,start,length,stride);
  
  // _XMP_coarray_rdma_array_set_n
  a_addr = addr;
  n_dims = 1;
  distance[0] = 1;
  elmts[0] = element_size;
  start[0] = 0;
  length[0] = 1;
  stride[0] = 1;

  _XMP_coarray_rdma_array_set_n(n_dims,start,length,stride,elmts,distance);

  // _XMP_rdma_image_set_n
  n_dims = remote_cp->image_dims;
  for(i = 0; i < n_dims; i++)
    img_n[i] = img_dims[i];
  if(remote_cp->f_coarray_offset != NULL){
    for(i = 0; i < n_dims; i++) img_n[i] += -1;
  }
  _XMP_coarray_rdma_image_set_n(n_dims,img_n);
  
  if(is_put) _XMP_coarray_put(remote_desc,a_addr,NULL);
  else _XMP_coarray_get(remote_desc,a_addr,NULL);
  return XMP_SUCCESS;
}

int xmp_coarray_put_scalar(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp, 
			   void *addr)
{
  return _xmp_coarray_put_get_scalar(TRUE,img_dims,remote_desc,remote_asp,addr);
}

int xmp_coarray_get_scalar(int img_dims[], xmp_desc_t remote_desc, xmp_array_section_t *remote_asp,
			   void *addr)
{
  return _xmp_coarray_put_get_scalar(FALSE,img_dims,remote_desc,remote_asp,addr);
}

int xmp_coarray_deallocate(xmp_desc_t desc)
{
  _XMP_coarray_deallocate((_XMP_coarray_t *)desc);
  return XMP_SUCCESS;
}

#ifdef not
　　　　xmp_sync_memory()
　　　　xmp_sync_all()
　　　　xmp_sync_image()
　　　　xmp_sync_images()
　　　　xmp_sync_imaegs_all()
#endif

/**
   fortran interface
*/
void xmp_new_coarray_(xmp_desc_t *desc, int *_elmt_size,
		      int *_ndims, long dim_lb[], long dim_ub[],
		      int *_img_ndims, int img_dim_size[])
{
  int i;
  int ndims = *_ndims;
  long dim_size[_XMP_N_MAX_DIM];
  long *dim_offset;
  _XMP_coarray_t *ap;
  void *loc = NULL;

  dim_offset = (long *)_XMP_alloc(sizeof(long)*ndims);

  for(i = 0; i < ndims; i++){
    dim_offset[i] = dim_lb[ndims-1-i];
    dim_size[i] = dim_ub[ndims-1-i] - dim_lb[ndims-1-i] + 1;
  }
  /* printf("ndim=%d, img_ndim=%d\n",ndims,*_img_ndims); */

  ap = (_XMP_coarray_t *)xmp_new_coarray(*_elmt_size,ndims,dim_size,
					 *_img_ndims,img_dim_size,&loc); // call C interface
  ap->f_coarray_offset = dim_offset;

  /* printf("new coarray=%p, addr=%p, loc=%p\n",ap,ap->real_addr,loc); */
  *desc = (xmp_desc_t)ap;
}

void xmp_coarray_deallocate_(xmp_desc_t *desc, int *status)
{
  _XMP_coarray_t *ap = (_XMP_coarray_t *)*desc;
  _XMP_free(ap->f_coarray_offset);
  _XMP_coarray_deallocate(ap);
}

void xmp_coarray_put_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec,
		      xmp_desc_t *_local_desc, xmp_desc_t *_local_asec, int *status)
{
  *status = _xmp_coarray_put_get(TRUE,img_dims,*_remote_desc,*_remote_asec,
				 NULL,*_local_desc,*_local_asec);
}

void xmp_coarray_put_local_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec,
			    xmp_desc_t *_local_desc, xmp_desc_t *_local_asec, int *status)
{
  *status = _xmp_coarray_put_get(TRUE,img_dims,*_remote_desc,*_remote_asec,
				 *_local_desc,NULL, *_local_asec);
}

void xmp_coarray_get_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec,
		     xmp_desc_t *_local_desc, xmp_desc_t *_local_asec,int *status)
{
  *status = _xmp_coarray_put_get(FALSE,img_dims,*_remote_desc,*_remote_asec,
				 NULL,*_local_desc,*_local_asec);
}

void xmp_coarray_get_local_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec,
			    xmp_desc_t *_local_desc, xmp_desc_t *_local_asec, int *status)
{
  *status = _xmp_coarray_put_get(FALSE,img_dims,*_remote_desc,*_remote_asec,
				 *_local_desc,NULL, *_local_asec);
}

void xmp_coarray_put_scalar_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec, 
			     void *addr, int *status)
{
  *status = _xmp_coarray_put_get_scalar(TRUE,img_dims,*_remote_desc,*_remote_asec,addr);
}

void xmp_coarray_get_scalar_(int img_dims[], xmp_desc_t *_remote_desc, xmp_desc_t *_remote_asec, 
			     void *addr, int *status)
{
  *status = _xmp_coarray_put_get_scalar(FALSE,img_dims,*_remote_desc,*_remote_asec,addr);
}

int xmp_this_image_()
{
  return xmp_this_image();
}

void xmp_sync_memory_(int *status)
{
  xmp_sync_memory(status);
}

void xmp_sync_all_(int *status)
{
  xmp_sync_all(status);
}

void xmp_sync_image_(int *_image, int *status)
{
  xmp_sync_image(*_image, status);
}

void xmp_sync_images_(int *_num, int *image_set, int *status)
{
  xmp_sync_images(*_num,image_set,status);
}

void xmp_sync_images_all_(int *status)
{
  xmp_sync_images_all(status);
}

/*
 *  called by runtime by cray pointer runtime
*/
void xmp_coarray_bind_set_dim_info_(xmp_desc_t *desc,int *lb, int *ub, void **addr)
{
  _XMP_coarray_t *ap = (_XMP_coarray_t *)*desc;
  for(int i=0; i < ap->coarray_dims; i++){
    lb[i] = ap->f_coarray_offset[i];
    ub[i] = lb[i]+ap->coarray_elmts[i]-1;
  }
  *addr = ap->real_addr;
}

void xmp_assign_cray_pointer_(void **ptr, void **addr)
{
  *ptr = *addr;
}
