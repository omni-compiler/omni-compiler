#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"

static int _coarray_dims, _image_dims, *_image_num, _array_dims;
static int _transfer_coarray_elmts, _transfer_array_elmts;
static _XMP_array_section_t *_coarray, *_array;

void _XMP_coarray_rdma_set(const int coarray_dims, const int array_dims, const int image_dims)
{
  _coarray      = malloc(sizeof(_XMP_array_section_t) * coarray_dims);
  _array        = malloc(sizeof(_XMP_array_section_t) * array_dims);
  _coarray_dims = coarray_dims;
  _array_dims   = array_dims;
  _image_num    = malloc(sizeof(int) * image_dims);
  _image_dims   = image_dims;
  _transfer_coarray_elmts = 1;
  _transfer_array_elmts   = 1;
}

void _XMP_coarray_rdma_set_f(const int *coarray_dims, const int *array_dims, const int *image_dims)
{
  _XMP_coarray_rdma_set(*coarray_dims, *array_dims, *image_dims);
}

void _XMP_coarray_rdma_coarray_set(const int dim, const int start, const int length, const int stride)
{
  _coarray[dim].start    = start;
  _coarray[dim].length   = length;
  _transfer_coarray_elmts *= length;
  _coarray[dim].stride = ((length == 1)? 1 : stride);
}

void _XMP_coarray_rdma_coarray_set_f(const int *dim, const int *start, const int *length, const int *stride)
{
  _XMP_coarray_rdma_coarray_set(*dim, *start, *length, *stride);
}

void _XMP_coarray_rdma_array_set(const int dim, const int start, const int length, const int stride, const int elmts, const int distance)
{
  _array[dim].start    = start;
  _array[dim].length   = length;
  _transfer_array_elmts *= length;
  _array[dim].stride = ((length == 1)? 1 : stride);
  _array[dim].elmts    = elmts;
  _array[dim].distance = distance;
}

void _XMP_coarray_rdma_array_set_f(const int *dim, const int *start, const int *length, const int *stride, const int *elmts, const int *distance)
{
  _XMP_coarray_rdma_array_set(*dim, *start, *length, *stride, *elmts, *distance);
}

void _XMP_coarray_rdma_node_set(const int dim, const int image_num)
{
  _image_num[dim]  = image_num;
}

void _XMP_coarray_rdma_node_set_f(const int *dim, const int *image_num)
{
  _XMP_coarray_rdma_node_set(*dim, *image_num);
}

// If array a is continuous, retrun _XMP_N_INT_TRUE.
// If array a is non-continuous (e.g. stride access), return _XMP_N_INT_FALSE.
static int check_continuous(const _XMP_array_section_t *a, const int dims)
{
  // Only 1 elements is transferred.
  // ex) a[2]
  // ex) b
  if(_transfer_coarray_elmts == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension length is transferred.
  // ex) a[1][2][2:3]
  if(_transfer_coarray_elmts == (a+dims-1)->length)
    if((a+dims-1)->stride == 1)
      return _XMP_N_INT_TRUE;

  // Is the dimension not continuous ?
  for(int i=0;i<dims;i++)
    if((a+i)->stride != 1)
      return _XMP_N_INT_FALSE;

  // (.., i-2, i-1)-th dimension's length is "1" &&
  // i-th dimension's stride is "1" && 
  // (i+1, i+2, ..)-th dimensions are ":".
  // ex) a[1][3][1:2][:]   // i = 2
  // ex) a[2][:][:][:]     // i = 0
  // ex) a[:][:][:][:]     // i = -1
  // Note that: the last dimension must be continuous ((a+dims-1)->stride != 1)
  int i;
  for(i=dims-1;i>=0;i--)
    if((a+i)->length != (a+i)->elmts)
      break;

  if(i == -1 || i == 0){
    return _XMP_N_INT_TRUE;     // Note that (a+i)->stride != 1
  }
  else{  // i != 0
    if(a->length != 1){         // a[:][1:2][:]  i == 1
      return _XMP_N_INT_FALSE;
    }
    else{                       // a[1][2][:][:] ?  i == 2
      for(int j=0;j<i;j++)
	if((a+j)->length != 1)
	  return _XMP_N_INT_FALSE;
    
      return _XMP_N_INT_TRUE;
    }
  }
}

void _XMP_coarray_rdma_do(const int rdma_code, const void *remote_coarray, const void *local_array, const void *local_coarray)
/* If a local array is a coarray, local_coarray != NULL. */
{
  if(_transfer_coarray_elmts == 0) return;

  if(_transfer_coarray_elmts != _transfer_array_elmts)
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;

  int target_image = 0;
  for(int i=0;i<_image_dims;i++)
    target_image += ((_XMP_coarray_t*)remote_coarray)->distance_of_image_elmts[i] * (_image_num[i] - 1);

  for(int i=0;i<_coarray_dims;i++){
    _coarray[i].elmts    = ((_XMP_coarray_t*)remote_coarray)->coarray_elmts[i];
    _coarray[i].distance = ((_XMP_coarray_t*)remote_coarray)->distance_of_coarray_elmts[i];
  }

  int remote_coarray_is_continuous, local_array_is_continuous;
  remote_coarray_is_continuous = check_continuous(_coarray, _coarray_dims);
  local_array_is_continuous    = check_continuous(_array, _array_dims); 

#ifdef _XMP_COARRAY_FJRDMA
  if(remote_coarray_is_continuous == _XMP_N_INT_FALSE || local_array_is_continuous == _XMP_N_INT_FALSE)
    _XMP_fatal("Sorry! Not continuous array is not supported.");
#endif

  if(_XMP_N_COARRAY_PUT == rdma_code){
#ifdef _XMP_COARRAY_GASNET
    _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_image,
		    _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts);
#elif _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_image, 
		    _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts);
#endif
  }
  else if(_XMP_N_COARRAY_GET == rdma_code){
#ifdef _XMP_COARRAY_GASNET
    _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_image,
                    _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts);
#elif _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_image, 
		    _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts);
#endif
  }
  else{
    _XMP_fatal("Unexpected Operation !!");
  }

  free(_coarray);
  free(_array);
  free(_image_num);
}

void _XMP_coarray_rdma_do_f(const int *rdma_code, const void *remote_coarray, const void *local_array, const void *local_coarray)
/* If a local array is a coarray, local_coarray != NULL. */
{
  _XMP_coarray_rdma_do(*rdma_code, remote_coarray, local_array, local_coarray);
}

void _XMP_coarray_sync_all()
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

void _XMP_coarray_sync_memory()
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

void xmp_sync_memory(const int* status)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

void xmp_sync_all(const int* status)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

void xmp_sync_image(int image, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images()");
}

void xmp_sync_image_f(int *image, int* status)
{
  xmp_sync_image(*image, status);
}

void xmp_sync_images(int num, int* image_set, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_images()");
}

void xmp_sync_images_f(int *num, int* image_set, int* status)
{
  xmp_sync_images(*num, image_set, status);
}

void xmp_sync_images_all(int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_all()");
}

size_t get_offset(const _XMP_array_section_t *array, const int dims)
{
  size_t offset = 0;
  for(int i=0;i<dims;i++)
    offset += (array+i)->start * (array+i)->distance;

  return offset;
}

void _XMP_coarray_shortcut_put(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src, 
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;

#ifdef _XMP_COARRAY_GASNET
  gasnet_put_nbi_bulk(rank, dst->addr[rank]+dst_offset,
		      src->addr[_XMP_world_rank]+src_offset, transfer_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_shortcut_put(rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst, src, transfer_size);
#endif
}

void _XMP_coarray_shortcut_get(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src,
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;
  
#ifdef _XMP_COARRAY_GASNET
  gasnet_get_bulk(dst->addr[_XMP_world_rank]+dst_offset, rank, src->addr[rank]+src_offset, transfer_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_shortcut_get(rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst, src, transfer_size);
#endif
}

void _XMP_coarray_shortcut_put_f(const int *target, const void *dst, const void *src, const size_t *dst_offset, 
				 const size_t *src_offset, const size_t *transfer_size)
{
  _XMP_coarray_shortcut_put(*target, dst, src, *dst_offset, *src_offset, *transfer_size);
}

void _XMP_coarray_shortcut_get_f(const int *target, const void *dst, const void *src, const size_t *dst_offset, 
				 const size_t *src_offset, const size_t *transfer_size)
{
  _XMP_coarray_shortcut_get(*target, dst, src, *dst_offset, *src_offset, *transfer_size);
}
