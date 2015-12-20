#include <stdlib.h>
#include <inttypes.h>
#include "xmp_internal.h"

/******************************************************************/
/* DESCRIPTION : Set addresses                                    */
/* ARGUMENT    : [OUT] *addrs     : Addresses                     */
/*               [IN] *base_addr  : Base address                  */
/*               [IN] *array_info : Information of array          */
/*               [IN] dims        : Number of dimensions of array */
/*               [IN] chunk_size  : Chunk size for copy           */
/*               [IN] copy_elmts  : Num of elements for copy      */
/******************************************************************/
void _XMP_set_coarray_addresses_with_chunk(uint64_t* addrs, const uint64_t base_addr, const _XMP_array_section_t* array_info, 
					   const int dims, const size_t chunk_size, const size_t copy_elmts)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  // array_info[dims-1].distance is an element size
  // chunk_size >= array_info[dims-1].distance
  switch (dims){
    int chunk_len;
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
      addrs[num++] = stride_offset[0] * i + base_addr;
    }
    break;
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 3:
    if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 4:
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 5:
    if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 6:
    if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
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
                  addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + base_addr;
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  case 7:
    if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
      chunk_len = chunk_size / array_info[6].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
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
                    addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + base_addr;
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
      for(size_t i=0,num=0;i<array_info[0].length;i++){
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
                  addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + base_addr;
                }
              }
            }
          }
        }
      }
    }
    else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(size_t m=0;m<array_info[4].length;m+=chunk_len){
                tmp[4] = stride_offset[4] * m;
                addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + base_addr;
              }
            }
          }
        }
      }
    }
    else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(size_t k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(size_t l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              addrs[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + base_addr;
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	    tmp[2] = stride_offset[2] * k;
	    addrs[num++] = tmp[0] + tmp[1] + tmp[2] + base_addr;
	  }
	}
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          addrs[num++] = tmp[0] + tmp[1] + base_addr;
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        addrs[num++] = stride_offset[0] * i + base_addr;
      }
    }
    break;
  }
}

void _XMP_set_coarray_addresses(const uint64_t addr, const _XMP_array_section_t *array, const int dims, 
				const size_t elmts, uint64_t* addrs)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array[i].stride * array[i].distance;
 
  switch (dims){
  case 1:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      addrs[num++] = addr + tmp[0];
    }
    break;
  case 2:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	addrs[num++] = addr + tmp[0] + tmp[1];
      }
    }
    break;
  case 3:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(size_t k=0;k<array[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2];
	}
      }
    }
    break;
  case 4:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
	  for(size_t l=0;l<array[3].length;l++){
	    tmp[3] = stride_offset[3] * l;
	    addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3];
	  }
	}
      }
    }
    break;
  case 5:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
	    for(size_t m=0;m<array[4].length;m++){
	      tmp[4] = stride_offset[4] * m;
	      addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
	    }
          }
        }
      }
    }
    break;
  case 6:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(size_t m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
	      for(size_t n=0;n<array[5].length;n++){
		tmp[5] = stride_offset[5] * n;
		addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
	      }
            }
          }
        }
      }
    }
    break;
  case 7:
    for(size_t i=0, num=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(size_t m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
              for(size_t n=0;n<array[5].length;n++){
                tmp[5] = stride_offset[5] * n;
		for(size_t p=0;p<array[6].length;p++){
		  tmp[6] = stride_offset[6] * p;
		  addrs[num++] = addr + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
		}
	      }
            }
          }
        }
      }
    }
    break;
  }
}

/***************************************************************************/
/* DESCRIPTION : Check the dimension of an array has all element ?         */
/* ARGUMENT    : [IN] *array_info : Information of array                   */
/*               [IN] dim         : Dimension                              */
/* RETURN      : If the dimension of an array has all element, return TRUE */
/***************************************************************************/
static int _is_all_element(const _XMP_array_section_t *array_info, int dim){
  if(array_info[dim].elmts == array_info[dim].length)
    return _XMP_N_INT_TRUE;
  else
    return _XMP_N_INT_FALSE;
}


/**************************************************************************************/
/* DESCRIPTION : Check round of array[dim]                                            */
/* ARGUMENT    : [IN] *array      : Information of array                              */
/*               [IN] dim         : Dimension                                         */
/* RETURN      : If a round of array[dim], return TRUE                                */
/* NOTE        : The following 3 lines are equal to this function                     */
/*    int last_elmt = array[dim].start + (array[dim].length - 1) * array[dim].stride; */
/*    int next_elmt = last_elmt + array[dim].stride - array[dim].elmts;               */
/*    return next_elmt == array[dim].start;                                           */
/**************************************************************************************/
static int _check_round(const _XMP_array_section_t *array, const int dim)
{
  return array[dim].length * array[dim].stride - array[dim].elmts == 0;
}

/**
   If 1dim array has a constant stride, return TRUE (Always TRUE)
*/
static int _is_constant_stride_1dim()
{
  return _XMP_N_INT_TRUE;
}

/********************************************************************/
/* DESCRIPTION : Is 2dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 2dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_2dim(const _XMP_array_section_t *array_info)
{
  if(array_info[0].stride == 1 && _check_round(array_info, 1)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[1].stride == 1){
    return _XMP_N_INT_TRUE;
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 3dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 3dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_3dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 4dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 4dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_4dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 5dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 5dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_5dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 6dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 6dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_6dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
     _is_all_element(array_info, 5)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
       _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1 &&
            _is_all_element(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4) &&
            array_info[5].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _check_round(array_info, 5)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/********************************************************************/
/* DESCRIPTION : Is 7dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 7dim array has a constant stride, return TRUE   */
/********************************************************************/
static int _is_constant_stride_7dim(const _XMP_array_section_t *array_info)
{
  if(array_info[1].stride == 1 && _is_all_element(array_info, 2) &&
     _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
     _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[0].stride == 1){
    if(_check_round(array_info, 1) && array_info[2].stride == 1 &&
       _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
       _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _check_round(array_info, 2) &&
            array_info[3].stride == 1 && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _check_round(array_info, 3) && array_info[4].stride == 1 &&
            _is_all_element(array_info, 5) && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _check_round(array_info, 4) &&
            array_info[5].stride == 1 && _is_all_element(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _check_round(array_info, 5) && array_info[6].stride == 1){
      return _XMP_N_INT_TRUE;
    }
    else if(_is_all_element(array_info, 1) && _is_all_element(array_info, 2) &&
            _is_all_element(array_info, 3) && _is_all_element(array_info, 4) &&
            _is_all_element(array_info, 5) && _check_round(array_info, 6)){
      return _XMP_N_INT_TRUE;
    }
  }

  return _XMP_N_INT_FALSE;
}

/**********************************************************************************/
/* DESCRIPTION : Check shape of two arrays, the same is except for start          */
/* ARGUMENT    : [IN] *array1_info : Information of array1                        */
/*               [IN] *array2_info : Information of array2                        */
/*               [IN] array1_dims  : Number of dimensions of array1               */
/*               [IN] array2_dims  : Number of dimensions of array2               */
/* RETURN:     : If two arrays have the same stride except for start, return TRUE */
/**********************************************************************************/
static int _is_the_same_shape_except_for_start(const _XMP_array_section_t *array1_info,
                                               const _XMP_array_section_t *array2_info,
                                               const int array1_dims, const int array2_dims)
{
  if(array1_dims != array2_dims) return _XMP_N_INT_FALSE;

  for(int i=0;i<array1_dims;i++)
    if(array1_info[i].length != array2_info[i].length ||
       array1_info[i].elmts  != array2_info[i].elmts ||
       array1_info[i].stride != array2_info[i].stride)
      return _XMP_N_INT_FALSE;

  return _XMP_N_INT_TRUE;
}

/********************************************************************/
/* DESCRIPTION : Check two arrays have the same stride              */
/* ARGUMENT    : [IN] *array1_info : Information of array1          */
/*               [IN] *array2_info : Information of array2          */
/*               [IN] array1_dims  : Number of dimensions of array1 */
/*               [IN] array2_dims  : Number of dimensions of array2 */
/* RETURN:     : If two arrays have the same stride, return TRUE    */
/* NOTE        : This function does not support the following very  */
/*               rare case.                                         */
/*               int a[10][10]; -> a[0:2][0:5:2];                   */
/*               An array has continuity jumped over the dimension  */
/********************************************************************/
int _XMP_is_the_same_constant_stride(const _XMP_array_section_t *array1_info,
				     const _XMP_array_section_t *array2_info,
				     const int array1_dims, const int array2_dims)
{
  if(! _is_the_same_shape_except_for_start(array1_info, array2_info,
                                           array1_dims, array2_dims))
    return _XMP_N_INT_FALSE;

  switch (array1_dims){
  case 1:
    return _is_constant_stride_1dim();
  case 2:
    return _is_constant_stride_2dim(array1_info);
  case 3:
    return _is_constant_stride_3dim(array1_info);
  case 4:
    return _is_constant_stride_4dim(array1_info);
  case 5:
    return _is_constant_stride_5dim(array1_info);
  case 6:
    return _is_constant_stride_6dim(array1_info);
  case 7:
    return _is_constant_stride_7dim(array1_info);
  default:
    _XMP_fatal("Coarray Error ! Dimension is too big.\n");
    return _XMP_N_INT_FALSE; // dummy
  }
}

/***************************************************************/
/* DESCRIPTION : Caluculate stride size of array               */
/* ARGUMENT    : [IN] *array_info : Information of array       */
/*               [IN] dims        : Demension of array         */
/*               [IN] chunk_size  : Size of chunk              */
/* RETURN:     : Stride size                                   */
/***************************************************************/
size_t _XMP_calc_stride(const _XMP_array_section_t *array_info, const int dims,
			const size_t chunk_size)
{
  uint64_t stride_offset[dims], tmp[dims];
  size_t stride[2];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (dims){
    size_t chunk_len;
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
      stride[num++] = stride_offset[0] * chunk_len * i;
      if(num == 2) goto end;
    }
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(size_t j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
          if(num == 2) goto end;
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
        if(num == 2) goto end;
      }
    }
  case 3:
   if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 4:
   if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 5:
   if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
	       if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 6:
   if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
     chunk_len = chunk_size / array_info[5].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
		 if(num == 2) goto end;
	       }
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
               if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
 case 7:
   if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
     chunk_len = chunk_size / array_info[6].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
		   if(num == 2) goto end;
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
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
		 if(num == 2) goto end;
	       }
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
     chunk_len = chunk_size / array_info[4].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
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
               if(num == 2) goto end;
	     }
	   }
	 }
       }
     }
   }
   else if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
     chunk_len = chunk_size / array_info[3].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k++){
	   tmp[2] = stride_offset[2] * k;
	   for(size_t l=0;l<array_info[3].length;l+=chunk_len){
	     tmp[3] = stride_offset[3] * l;
	     stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
	     if(num == 2) goto end;
	   }
	 }
       }
     }
   }
   else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
     chunk_len = chunk_size / array_info[2].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j++){
	 tmp[1] = stride_offset[1] * j;
	 for(size_t k=0;k<array_info[2].length;k+=chunk_len){
	   tmp[2] = stride_offset[2] * k;
	   stride[num++] = tmp[0] + tmp[1] + tmp[2];
	   if(num == 2) goto end;
	 }
       }
     }
   }
   else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
     chunk_len = chunk_size / array_info[1].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i++){
       tmp[0] = stride_offset[0] * i;
       for(size_t j=0;j<array_info[1].length;j+=chunk_len){
	 tmp[1] = stride_offset[1] * j;
	 stride[num++] = tmp[0] + tmp[1];
	 if(num == 2) goto end;
       }
     }
   }
   else{                                   // chunk_size >= array_info[0].distance
     chunk_len = chunk_size / array_info[0].distance;
     for(size_t i=0,num=0;i<array_info[0].length;i+=chunk_len){
       stride[num++] = stride_offset[0] * i;
       if(num == 2) goto end;
     }
   }
   break;
  }

 end:
  return stride[1] - stride[0];
}
