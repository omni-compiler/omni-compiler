#include <string.h>
#include "xmp_internal.h"
#include "xmp_math_function.h"

/********************************************************************/
/* DESCRIPTION : Check shape of two arrays                          */
/* ARGUMENT    : [IN] array1_dims : Number of dimensions of array 1 */
/*               [IN] array2_dims : Number of dimensions of array 2 */
/*               [IN] *array1     : Information of array1           */
/*               [IN] *array2     : Information of array2           */
/* RETRUN     : If the shape of two arrays is the same, return TRUE */
/* EXAMPLE    : int a[100]:[*], b[100]:[*];                         */
/*              a[0:10:2], b[0:10:2] -> TRUE                        */
/*              a[0:10:2], b[1:10:2] -> FALSE                       */
/*              a[0:10:2], b[0:11:2] -> FALSE                       */
/*              a[0:10:2], b[0:10:3] -> FALSE                       */
/********************************************************************/
static int _is_the_same_shape(const int array1_dims, const int array2_dims, 
			      const _XMP_array_section_t *array1, const _XMP_array_section_t *array2)
{
  if(array1_dims != array2_dims)
    return _XMP_N_INT_FALSE;
  
  for(int i=0;i<array1_dims;i++)
    if(array1[i].start != array2[i].start || array1[i].length != array2[i].length ||
       array1[i].elmts != array2[i].elmts || array1[i].stride != array2[i].stride)
      return _XMP_N_INT_FALSE;

  return _XMP_N_INT_TRUE;
}

/************************************************************************************/
/* DESCRIPTION : Execute copy operation in only local node for NON-continuous array */
/* ARGUMENT    : [OUT] *dst     : Pointer of destination array                      */
/*               [IN] *src      : Pointer of source array                           */
/*               [IN] dst_dims  : Number of dimensions of destination array         */
/*               [IN] src_dims  : Number of dimensions of source array              */
/*               [IN] *dst_info : Information of destination array                  */
/*               [IN] *src_info : Information of source array                       */
/*               [IN] dst_elmts : Number of elements of destination array           */
/*               [IN] src_elmts : Number of elements of source array                */
/*               [IN] elmt_size : Element size                                      */
/* NOTE       : This function is called by both put and get functions               */
/* EXAMPLE    :                                                                     */
/*   if(xmp_node_num() == 1)                                                        */
/*     a[0:100:2]:[1] = b[0:100:3]; // a[] is a dst, b[] is a src                   */
/************************************************************************************/
static void _local_NON_continuous_copy(char *dst, const char *src, const int dst_dims, const int src_dims,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  if(dst_elmts == src_elmts){
    size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
    size_t copy_elmts = dst_elmts/(copy_chunk/elmt_size);
    size_t dst_stride[copy_elmts], src_stride[copy_elmts];

    // Set stride
    _XMP_set_stride(dst_stride, dst_info, dst_dims, copy_chunk, copy_elmts);
    // The _is_the_same_shape() is used to reduce cost of the second _XMP_set_stride()
    if(_is_the_same_shape(dst_dims, src_dims, dst_info, src_info))
      for(int i=0;i<copy_elmts;i++)
	src_stride[i] = dst_stride[i];
    else
      _XMP_set_stride(src_stride, src_info, src_dims, copy_chunk, copy_elmts);

    // Execute local memory copy
    for(int i=0;i<copy_elmts;i++)
      memcpy(dst+dst_stride[i], src+src_stride[i], copy_chunk);
  }
  else if(src_elmts == 1){     /* a[0:100:2]:[1] = b[2]; or a[0:100:2] = b[2]:[1]; */
    switch (dst_dims){
    case 1:
      _XMP_stride_memcpy_1dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 2:
      _XMP_stride_memcpy_2dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 3:
      _XMP_stride_memcpy_3dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 4:
      _XMP_stride_memcpy_4dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 5:
      _XMP_stride_memcpy_5dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 6:
      _XMP_stride_memcpy_6dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    case 7:
      _XMP_stride_memcpy_7dim(dst, src, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
      break;
    default:
      _XMP_fatal("Coarray Error ! Dimension is too big.\n");
      break;
    }
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/***************************************************************************************/
/* DESCRIPTION : Execute put operation in only local node                              */
/* ARGUMENT    : [OUT] *dst_desc     : Descriptor of destination coarray               */
/*               [IN] *src           : Pointer of source array                         */
/*               [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE) */
/*               [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)      */
/*               [IN] dst_dims       : Number of dimensions of destination array       */
/*               [IN] src_dims       : Number of dimensions of source array            */
/*               [IN] *dst_info      : Information of destination array                */
/*               [IN] *src_info      : Information of source array                     */
/*               [IN] dst_elmts      : Number of elements of destination array         */
/*               [IN] src_elmts      : Number of elements of source array              */
/* NOTE       : Destination array must be a coarray, and source array is a coarray or  */
/*              a normal array                                                         */
/* EXAMPLE    :                                                                        */
/*   if(xmp_node_num() == 1)                                                           */
/*     a[:]:[1] = b[:];  // a[] is a dst, b[] is a src.                                */
/***************************************************************************************/
void _XMP_local_put(_XMP_coarray_t *dst_desc, const void *src, const int dst_continuous, const int src_continuous, 
		    const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		    const _XMP_array_section_t *src_info, const size_t dst_elmts, const size_t src_elmts)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t elmt_size  = dst_desc->elmt_size;

  if(dst_continuous && src_continuous)
    _XMP_local_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src+src_offset,
			       dst_elmts, src_elmts, elmt_size);
  else
    _local_NON_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src+src_offset,
			       dst_dims, src_dims, dst_info, src_info, dst_elmts, src_elmts, elmt_size);
}

/****************************************************************************************/
/* DESCRIPTION : Execute get operation in only local node                               */
/* ARGUMENT    : [OUT] *dst          : Pointer of destination array                     */
/*               [IN] *src_desc      : Descriptor of source coarray                     */
/*               [IN] dst_continuous : Is destination region continuous ? (TRUE/FALSE)  */
/*               [IN] src_continuous : Is source region continuous ? (TRUE/FALSE)       */
/*               [IN] dst_dims       : Number of dimensions of destination array        */
/*               [IN] src_dims       : Number of dimensions of source array             */
/*               [IN] *dst_info      : Information of destination array                 */
/*               [IN] *src_info      : Information of source array                      */
/*               [IN] dst_elmts      : Number of elements of destination array          */
/*               [IN] src_elmts      : Number of elements of source array               */
/* NOTE       : Source array must be a coarray, and destination array is a coarray or   */
/*              a normal array                                                          */
/* EXAMPLE    :                                                                         */
/*   if(xmp_node_num() == 1)                                                            */
/*     a[:] = b[:]:[1];  // a[] is a dst, b[] is a src.                                 */
/****************************************************************************************/
void _XMP_local_get(void *dst, const _XMP_coarray_t *src_desc, const int dst_continuous, const int src_continuous, 
		    const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		    const _XMP_array_section_t *src_info, const size_t dst_elmts, const size_t src_elmts)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t elmt_size  = src_desc->elmt_size;

  if(dst_continuous && src_continuous)
    _XMP_local_continuous_copy((char *)dst+dst_offset, (char *)src_desc->real_addr+src_offset, 
			       dst_elmts, src_elmts, elmt_size);
  else
    _local_NON_continuous_copy((char *)dst+dst_offset, (char *)src_desc->real_addr+src_offset,
			       dst_dims, src_dims, dst_info, src_info, dst_elmts, src_elmts, elmt_size);
}
