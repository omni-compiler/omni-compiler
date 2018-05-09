#include <string.h>
#include <stdlib.h>
#include "xmp_internal.h"
#include "xmp_math_function.h"

/*************************************************************************/
/* DESCRIPTION : Calculate chunk size                                    */
/* ARGUMENT    : [IN] copy_chunk_dim : Maximum dimensions of which array */
/*                                     has all elements                  */
/*               [IN] *array         : Information of array              */
/* RETURN     : Chunk size for copy                                      */
/* EXAMPLE    :                                                          */
/*   int a[10][20];                                                      */
/*   a[:][:]     (copy_chunk_dim = 0) -> 10 * 20 * sizeof(int)           */
/*   a[0][:]     (copy_chunk_dim = 1) -> 1  * 20 * sizeof(int)           */
/*   a[0:2][:]   (copy_chunk_dim = 1) -> 2  * 20 * sizeof(int)           */
/*   a[0:2:2][:] (copy_chunk_dim = 1) -> 1  * 20 * sizeof(int)           */
/*   a[0][0:3]   (copy_chunk_dim = 2) -> 3 * sizeof(int)                 */
/*   a[:][0]     (copy_chunk_dim = 2) -> 1 * sizeof(int)                 */
/*   a[0][:10:2] (copy_chunk_dim = 2) -> 1 * sizeof(int)                 */
/*************************************************************************/
size_t _XMP_calc_copy_chunk(const int copy_chunk_dim, const _XMP_array_section_t* array)
{
  if(copy_chunk_dim == 0)   // All elements are copied
    return array[0].length * array[0].distance;

  if(array[copy_chunk_dim-1].stride == 1)
    return array[copy_chunk_dim-1].length * array[copy_chunk_dim-1].distance;
  else
    return array[copy_chunk_dim-1].distance;
}

/******************************************************************/
/* DESCRIPTION : Set stride for local copy                        */
/* ARGUMENT    : [OUT] *stride    : Stride for local copy         */
/*               [IN] *array_info : Information of array          */
/*               [IN] dims        : Number of dimensions of array */
/*               [IN] chunk_size  : Chunk size for copy           */
/*               [IN] copy_elmts  : Num of elements for copy      */
/******************************************************************/
static void _XMP_set_stride(size_t* stride, const _XMP_array_section_t* array_info, const int dims, 
			    const size_t chunk_size, const size_t copy_elmts)
{
  // Temporally variables to reduce offset calculation
  size_t stride_offset[dims], tmp[dims], chunk_len, num = 0;
  
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  // array_info[dims-1].distance is an element size
  // chunk_size >= array_info[dims-1].distance
  switch (dims){
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(size_t i=0;i<array_info[0].length;i+=chunk_len){
      stride[num++] = stride_offset[0] * i;
    }
    break;
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
	stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 3:
    if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    stride[num++] = tmp[0] + tmp[1] + tmp[2];
	  }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
	  stride[num++] = tmp[0] + tmp[1];
	}
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 4:
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
	    for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	      tmp[3] = stride_offset[3] * l;
	      stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	    }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    stride[num++] = tmp[0] + tmp[1] + tmp[2];
	  }
	}
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 5:
    if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
	      for(size_t m=0;m<array_info[4].length;m+=chunk_len){
		tmp[4] = stride_offset[4] * m;
		stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	      }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 6:
    if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
		for(size_t n=0;n<array_info[5].length;n+=chunk_len){
		  tmp[5] = stride_offset[5] * n;
		  stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
		}
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
              }
            }
          }
        }
      }
    }
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 7:
    if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
      chunk_len = chunk_size / array_info[6].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(size_t n=0;n<array_info[5].length;n++){
                  tmp[5] = stride_offset[5] * n;
		  for(size_t p=0;p<array_info[6].length;p+=chunk_len){
		    tmp[6] = stride_offset[6] * p;
		    stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
		  }
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(size_t n=0;n<array_info[5].length;n+=chunk_len){
                  tmp[5] = stride_offset[5] * n;
                  stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
              }
            }
          }
        }
      }
    }
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  }
}

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
/* DESCRIPTION : Execute copy operation in only local node for NON-contiguous array */
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
static void _local_NON_contiguous_copy(char *dst, const char *src, const int dst_dims, const int src_dims,
				       const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
				       const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  if(dst_elmts == src_elmts){
    size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
    size_t copy_elmts = dst_elmts/(copy_chunk/elmt_size);
    size_t dst_stride[copy_elmts], src_stride[copy_elmts];

    // Set stride
    _XMP_set_stride(dst_stride, dst_info, dst_dims, copy_chunk, copy_elmts);

    if(_is_the_same_shape(dst_dims, src_dims, dst_info, src_info)){
      // The _is_the_same_shape() is used to reduce cost of the second _XMP_set_stride()
      for(size_t i=0;i<copy_elmts;i++)
	src_stride[i] = dst_stride[i];
    }
    else
      _XMP_set_stride(src_stride, src_info, src_dims, copy_chunk, copy_elmts);

    // Execute local memory copy
    if(_XMP_check_overlapping(dst+dst_stride[0], dst+dst_stride[copy_elmts-1]+copy_chunk,
			      src+src_stride[0], src+src_stride[copy_elmts-1]+copy_chunk)){

      char *tmp = malloc(copy_elmts * copy_chunk);

      size_t offset = 0;
      for(size_t i=0;i<copy_elmts;i++){
	memcpy(tmp+offset, src+src_stride[i], copy_chunk);
	offset += copy_chunk;
      }

      offset = 0;
      for(size_t i=0;i<copy_elmts;i++){
	memmove(dst+dst_stride[i], tmp+offset, copy_chunk);
	offset += copy_chunk;
      }

      free(tmp);
    }
    else
      for(size_t i=0;i<copy_elmts;i++)
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
/*               [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE) */
/*               [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)      */
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
void _XMP_local_put(_XMP_coarray_t *dst_desc, const void *src, const int dst_contiguous, const int src_contiguous, 
		    const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		    const _XMP_array_section_t *src_info, const size_t dst_elmts, const size_t src_elmts)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t elmt_size  = dst_desc->elmt_size;

  if(dst_contiguous && src_contiguous)
    _XMP_local_contiguous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src+src_offset,
			       dst_elmts, src_elmts, elmt_size);
  else
    _local_NON_contiguous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src+src_offset,
			       dst_dims, src_dims, dst_info, src_info, dst_elmts, src_elmts, elmt_size);
}

/****************************************************************************************/
/* DESCRIPTION : Execute get operation in only local node                               */
/* ARGUMENT    : [OUT] *dst          : Pointer of destination array                     */
/*               [IN] *src_desc      : Descriptor of source coarray                     */
/*               [IN] dst_contiguous : Is destination region contiguous ? (TRUE/FALSE)  */
/*               [IN] src_contiguous : Is source region contiguous ? (TRUE/FALSE)       */
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
void _XMP_local_get(void *dst, const _XMP_coarray_t *src_desc, const int dst_contiguous, const int src_contiguous, 
		    const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
		    const _XMP_array_section_t *src_info, const size_t dst_elmts, const size_t src_elmts)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  size_t src_offset = _XMP_get_offset(src_info, src_dims);
  size_t elmt_size  = src_desc->elmt_size;

  if(dst_contiguous && src_contiguous)
    _XMP_local_contiguous_copy((char *)dst+dst_offset, (char *)src_desc->real_addr+src_offset, 
			       dst_elmts, src_elmts, elmt_size);
  else
    _local_NON_contiguous_copy((char *)dst+dst_offset, (char *)src_desc->real_addr+src_offset,
			       dst_dims, src_dims, dst_info, src_info, dst_elmts, src_elmts, elmt_size);
}
