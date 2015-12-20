#include <string.h>
#include "xmp_internal.h"

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
void _XMP_set_stride(size_t* stride, const _XMP_array_section_t* array_info, const int dims, 
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
