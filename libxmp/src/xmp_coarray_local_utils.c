#include <string.h>
#include "xmp_internal.h"

/*****************************************************************/
/* DESCRIPTION : Calculate chunk size                            */
/* ARGUMENT    : [IN] dim    : Maximum dimensions of which array */
/*                             has all elements                  */
/*               [IN] *array : Information of array              */
/* RETURN     : Chunk size for copy                              */
/* EXAMPLE    :                                                  */
/*   int a[10][20];                                              */
/*   a[:][:] (dim = 0)     -> 800                                */
/*   a[0][:] (dim = 1)     -> 80                                 */
/*   a[0:2][:] (dim = 1)   -> 160                                */
/*   a[0:2:2][:] (dim = 1) -> 80                                 */
/*   a[0][0:3] (dim = 2)   -> 12                                 */
/*   a[:][0] (dim = 2)     -> 4                                  */
/*   a[0][::2] (dim = 2)   -> 4                                  */
/*****************************************************************/
size_t _XMP_calc_copy_chunk(const unsigned int dim, const _XMP_array_section_t* array)
{
  if(dim == 0)   // All elements are copied
    return array[0].length * array[0].distance;

  if(array[dim-1].stride == 1)
    return array[dim-1].length * array[dim-1].distance;
  else
    return array[dim-1].distance;
}

/*****************************************************************/
/* DESCRIPTION : Set stride for local copy                       */
/* ARGUMENT    : [OUT] *stride   : Stride for local copy         */
/*               [IN] *array     : Pointer of array              */
/*               [IN] dims       : Number of dimensions of array */
/*               [IN] chunk_size : Chunk size for copy           */
/*               [IN] copy_elmts : Num of elements for copy      */
/*****************************************************************/
void _XMP_set_stride(size_t* stride, const _XMP_array_section_t* array, const int dims, 
		     const unsigned int chunk_size, const unsigned int copy_elmts)
{
  // Temporally variables to reduce offset calculation
  size_t stride_offset[dims], tmp[dims];
  for(int i=0;i<dims;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  // array[dims-1].distance is an element size
  // chunk_size >= array[dims-1].distance
  switch (dims){
    int chunk_len;
  case 1:
    chunk_len = chunk_size / array[0].distance;
    for(int i=0,num=0;i<array[0].length;i+=chunk_len){
      stride[num++] = stride_offset[0] * i;
    }
    break;
  case 2:
    if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                               // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
	stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 3:
    if(array[1].distance > chunk_size){ // array[1].distance > chunk_size >= array[2].distance
      chunk_len = chunk_size / array[2].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    stride[num++] = tmp[0] + tmp[1] + tmp[2];
	  }
        }
      }
    }
    else if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
	  stride[num++] = tmp[0] + tmp[1];
	}
      }
    }
    else{                                   // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 4:
    if(array[2].distance > chunk_size){ // array[2].distance > chunk_size >= array[3].distance
      chunk_len = chunk_size / array[3].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
	    for(int l=0;l<array[3].length;l+=chunk_len){
	      tmp[3] = stride_offset[3] * l;
	      stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	    }
          }
        }
      }
    }
    else if(array[1].distance > chunk_size){ // array[1].distance > chunk_size >= array[2].distance
      chunk_len = chunk_size / array[2].distance;
      for(int i=0,num=0;i<array[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    stride[num++] = tmp[0] + tmp[1] + tmp[2];
	  }
	}
      }
    }
    else if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 5:
    if(array[3].distance > chunk_size){ // array[3].distance > chunk_size >= array[4].distance
      chunk_len = chunk_size / array[4].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
	      for(int m=0;m<array[4].length;m+=chunk_len){
		tmp[4] = stride_offset[4] * m;
		stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	      }
            }
          }
        }
      }
    }
    else if(array[2].distance > chunk_size){ // array[2].distance > chunk_size >= array[3].distance
      chunk_len = chunk_size / array[3].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array[1].distance > chunk_size){ // array[1].distance > chunk_size >= array[2].distance
      chunk_len = chunk_size / array[2].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 6:
    if(array[4].distance > chunk_size){ // array[4].distance > chunk_size >= array[5].distance
      chunk_len = chunk_size / array[5].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array[4].length;m++){
                tmp[4] = stride_offset[4] * m;
		for(int n=0;n<array[5].length;n+=chunk_len){
		  tmp[5] = stride_offset[5] * n;
		  stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
		}
              }
            }
          }
        }
      }
    }
    else if(array[3].distance > chunk_size){ // array[3].distance > chunk_size >= array[4].distance
      chunk_len = chunk_size / array[4].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
              }
            }
          }
        }
      }
    }
    if(array[2].distance > chunk_size){ // array[2].distance > chunk_size >= array[3].distance
      chunk_len = chunk_size / array[3].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array[1].distance > chunk_size){ // array[1].distance > chunk_size >= array[2].distance
      chunk_len = chunk_size / array[2].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 7:
    if(array[5].distance > chunk_size){ // array[5].distance > chunk_size >= array[6].distance
      chunk_len = chunk_size / array[6].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(int n=0;n<array[5].length;n++){
                  tmp[5] = stride_offset[5] * n;
		  for(int p=0;p<array[6].length;p+=chunk_len){
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
    else if(array[4].distance > chunk_size){ // array[4].distance > chunk_size >= array[5].distance
      chunk_len = chunk_size / array[5].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(int n=0;n<array[5].length;n+=chunk_len){
                  tmp[5] = stride_offset[5] * n;
                  stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
                }
              }
            }
          }
        }
      }
    }
    else if(array[3].distance > chunk_size){ // array[3].distance > chunk_size >= array[4].distance
      chunk_len = chunk_size / array[4].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
              }
            }
          }
        }
      }
    }
    if(array[2].distance > chunk_size){ // array[2].distance > chunk_size >= array[3].distance
      chunk_len = chunk_size / array[3].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array[1].distance > chunk_size){ // array[1].distance > chunk_size >= array[2].distance
      chunk_len = chunk_size / array[2].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array[0].distance > chunk_size){ // array[0].distance > chunk_size >= array[1].distance
      chunk_len = chunk_size / array[1].distance;
      for(int i=0,num=0;i<array[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array[0].distance
      chunk_len = chunk_size / array[0].distance;
      for(int i=0,num=0;i<array[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  }
}
