#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "common.h"

size_t _XMP_calc_copy_chunk(const unsigned int copy_chunk_dim, const _XMP_array_section_t* array)
{
  if(copy_chunk_dim == 0)   // All elements are copied
    return array[0].length * array[0].distance;

  if(array[copy_chunk_dim-1].stride == 1)
    return array[copy_chunk_dim-1].length * array[copy_chunk_dim-1].distance;
  else
    return array[copy_chunk_dim-1].distance;
}

unsigned int _XMP_get_dim_of_allelmts(const int dims,
                                      const _XMP_array_section_t* array_info)
{
  unsigned int val = dims;

  for(int i=dims-1;i>=0;i--){
    if(array_info[i].start == 0 && array_info[i].length == array_info[i].elmts)
      val--;
    else
      return val;
  }

  return val;
}

int _heavy_check_stride(_XMP_array_section_t* array_info, int dims)
{
  int elmts = 1;
  for(int i=0;i<dims;i++)
    elmts *= array_info[i].length;
  
  int tmp[dims], stride_offset[dims];

  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  unsigned int chunk_dim  = _XMP_get_dim_of_allelmts(dims, array_info);
  unsigned int chunk_size = _XMP_calc_copy_chunk(chunk_dim, array_info);
  size_t elmt_size        = array_info[dims-1].distance;
  size_t copy_elmts       = elmts/(chunk_size/elmt_size);
  int stride[copy_elmts];

  switch (dims){
    int chunk_len;
  case 1:
    chunk_len = chunk_size / array_info[0].distance;
    for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
      stride[num++] = stride_offset[0] * i;
    }
    break;
  case 2:
    if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                               // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 3:
    if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 4:
    if(array_info[2].distance > chunk_size){ // array_info[2].distance > chunk_size >= array_info[3].distance
      chunk_len = chunk_size / array_info[3].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 5:
    if(array_info[3].distance > chunk_size){ // array_info[3].distance > chunk_size >= array_info[4].distance
      chunk_len = chunk_size / array_info[4].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 6:
    if(array_info[4].distance > chunk_size){ // array_info[4].distance > chunk_size >= array_info[5].distance
      chunk_len = chunk_size / array_info[5].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(int n=0;n<array_info[5].length;n+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  case 7:
    if(array_info[5].distance > chunk_size){ // array_info[5].distance > chunk_size >= array_info[6].distance
      chunk_len = chunk_size / array_info[6].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(int n=0;n<array_info[5].length;n++){
                  tmp[5] = stride_offset[5] * n;
                  for(int p=0;p<array_info[6].length;p+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m++){
                tmp[4] = stride_offset[4] * m;
                for(int n=0;n<array_info[5].length;n+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l++){
              tmp[3] = stride_offset[3] * l;
              for(int m=0;m<array_info[4].length;m+=chunk_len){
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
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k++){
            tmp[2] = stride_offset[2] * k;
            for(int l=0;l<array_info[3].length;l+=chunk_len){
              tmp[3] = stride_offset[3] * l;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
          }
        }
      }
    }
    else if(array_info[1].distance > chunk_size){ // array_info[1].distance > chunk_size >= array_info[2].distance
      chunk_len = chunk_size / array_info[2].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j++){
          tmp[1] = stride_offset[1] * j;
          for(int k=0;k<array_info[2].length;k+=chunk_len){
            tmp[2] = stride_offset[2] * k;
            stride[num++] = tmp[0] + tmp[1] + tmp[2];
          }
        }
      }
    }
    else if(array_info[0].distance > chunk_size){ // array_info[0].distance > chunk_size >= array_info[1].distance
      chunk_len = chunk_size / array_info[1].distance;
      for(int i=0,num=0;i<array_info[0].length;i++){
        tmp[0] = stride_offset[0] * i;
        for(int j=0;j<array_info[1].length;j+=chunk_len){
          tmp[1] = stride_offset[1] * j;
          stride[num++] = tmp[0] + tmp[1];
        }
      }
    }
    else{                                   // chunk_size >= array_info[0].distance
      chunk_len = chunk_size / array_info[0].distance;
      for(int i=0,num=0;i<array_info[0].length;i+=chunk_len){
        stride[num++] = stride_offset[0] * i;
      }
    }
    break;
  }

  for(int i=1;i<copy_elmts;i++)
    if(stride[1]-stride[0] != stride[i]-stride[i-1])
      return false;

  return true;
}

static int _is_all_element(const _XMP_array_section_t *array_info, int dim){
  if(array_info[dim].elmts == array_info[dim].length)
    return _XMP_N_INT_TRUE;
  else
    return _XMP_N_INT_FALSE;
}

int _XMP_is_constant_stride_1dim()
{
  return _XMP_N_INT_TRUE;
}

static int _check_round(const _XMP_array_section_t *array_info, const int dim)
{
  int last_elmt = array_info[dim].start + (array_info[dim].length - 1) * array_info[dim].stride;
  int next_elmt = last_elmt + array_info[dim].stride - array_info[dim].elmts;

  return next_elmt == array_info[dim].start;
}

int _XMP_is_constant_stride_2dim(const _XMP_array_section_t *array_info)
{
  if(array_info[0].stride == 1 && _check_round(array_info, 1)){
    return _XMP_N_INT_TRUE;
  }
  else if(array_info[1].stride == 1){
    return _XMP_N_INT_TRUE;
  }
  
  return _XMP_N_INT_FALSE;
}

int _XMP_is_constant_stride_3dim(const _XMP_array_section_t *array_info)
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

int _XMP_is_constant_stride_4dim(const _XMP_array_section_t *array_info)
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

int _XMP_is_constant_stride_5dim(const _XMP_array_section_t *array_info)
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

int _XMP_is_constant_stride_6dim(const _XMP_array_section_t *array_info)
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


int _XMP_is_constant_stride_7dim(const _XMP_array_section_t *array_info)
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

int _is_the_same_constant_stride(const _XMP_array_section_t *array,
				 const int dims)
{
  switch (dims){
  case 1:
    return _XMP_is_constant_stride_1dim();
  case 2:
    return _XMP_is_constant_stride_2dim(array);
  case 3:
    return _XMP_is_constant_stride_3dim(array);
  case 4:
    return _XMP_is_constant_stride_4dim(array);
  case 5:
    return _XMP_is_constant_stride_5dim(array);
  case 6:
    return _XMP_is_constant_stride_6dim(array);
  case 7:
    return _XMP_is_constant_stride_7dim(array);
  default:
    exit(1);

    return _XMP_N_INT_FALSE; // dummy
  }
}

