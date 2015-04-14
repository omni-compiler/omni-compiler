#include <string.h>
#include "xmp_internal.h"

void _XMP_stride_memcpy_1dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0, tmp;
  size_t stride_offset = array[0].stride * array[0].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp = stride_offset * i;
      memcpy(buf1 + buf1_offset, buf2 + tmp, element_size);
      buf1_offset += element_size;
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp = stride_offset * i;
      memcpy(buf1 + tmp, buf2 + buf1_offset, element_size);
      buf1_offset += element_size;
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp = stride_offset * i;
      memcpy(buf1 + tmp, buf2, element_size);
    }
  }
}

void _XMP_stride_memcpy_2dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[2], stride_offset[2];

  for(int i=0;i<2;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1], element_size);
        buf1_offset += element_size;
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        memcpy(buf1 + tmp[0] + tmp[1], buf2 + buf1_offset, element_size);
        buf1_offset += element_size;
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        memcpy(buf1 + tmp[0] + tmp[1], buf2, element_size);
      }
    }
  }
}

void _XMP_stride_memcpy_3dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[3], stride_offset[3];

  for(int i=0;i<3;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2], element_size);
          buf1_offset += element_size;
        }
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2 + buf1_offset, element_size);
          buf1_offset += element_size;
        }
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2, element_size);
        }
      }
    }
  }
}

void _XMP_stride_memcpy_4dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[4], stride_offset[4];

  for(int i=0;i<4;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3], element_size);
            buf1_offset += element_size;
          }
        }
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
                   buf2 + buf1_offset, element_size);
            buf1_offset += element_size;
          }
        }
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
                   buf2, element_size);
          }
        }
      }
    }
  }
}

void _XMP_stride_memcpy_5dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[5], stride_offset[5];

  for(int i=0;i<5;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4], element_size);
              buf1_offset += element_size;
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
                     buf2 + buf1_offset, element_size);
              buf1_offset += element_size;
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
                     buf2, element_size);
            }
          }
        }
      }
    }
  }
}

void _XMP_stride_memcpy_6dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[6], stride_offset[6];

  for(int i=0;i<6;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5],
                       element_size);
                buf1_offset += element_size;
              }
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5],
                       buf2 + buf1_offset, element_size);
                buf1_offset += element_size;
              }
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5],
                       buf2, element_size);
              }
            }
          }
        }
      }
    }
  }
}

void _XMP_stride_memcpy_7dim(char *buf1, const char *buf2, const _XMP_array_section_t *array, size_t element_size, int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[7], stride_offset[7];

  for(int i=0;i<7;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  if(flag == _XMP_PACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(int q=0;q<array[6].length;q++){
                  tmp[6] = stride_offset[6] * q;
                  memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6],
                         element_size);
                  buf1_offset += element_size;
                }
              }
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_UNPACK){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(int q=0;q<array[6].length;q++){
                  tmp[6] = stride_offset[6] * q;
                  memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6],
                         buf2 + buf1_offset, element_size);
                  buf1_offset += element_size;
                }
              }
            }
          }
        }
      }
    }
  }
  else if(flag == _XMP_MPUT){
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(int q=0;q<array[6].length;q++){
                  tmp[6] = stride_offset[6] * q;
                  memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6],
                         buf2, element_size);
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
   Continuous PUT operation in local node
*/
static void _local_continuous_put(const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info, 
				  const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst, 
				  const void *src, size_t dst_point, size_t src_point,
				  const size_t transfer_coarray_elmts, const size_t transfer_array_elmts,
				  const size_t elmt_size)
{
  if(transfer_coarray_elmts == transfer_array_elmts){
    memcpy(dst->real_addr + dst_point, (char *)src + src_point, transfer_coarray_elmts * elmt_size);
  }
  else if(transfer_array_elmts == 1){
    for(int i=0;i<transfer_coarray_elmts;i++)
      memcpy(dst->real_addr + dst_point + elmt_size*i, (char *)src + src_point, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

static void _local_NON_continuous_mput(const int dst_dims, char *dst, const char *src, _XMP_array_section_t *dst_info,
				       const size_t elmt_size)
{
  switch (dst_dims){
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 6:
    _XMP_stride_memcpy_6dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  case 7:
    _XMP_stride_memcpy_7dim(dst, src, dst_info, elmt_size, _XMP_MPUT);
    break;;
  }
}

static void _set_stride(size_t* stride, const _XMP_array_section_t* array, const int dims)
{
  uint64_t stride_offset[dims], tmp[dims];

  // Temporally variables to reduce calculation for offset
  for(int i=0;i<dims;i++)
    stride_offset[i] = array[i].stride * array[i].distance;

  int num = 0;
  switch (dims){
  case 1:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      stride[num++] = tmp[0];
    }
    break;;
  case 2:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        stride[num++] = tmp[0] + tmp[1];
      }
    }
    break;;
  case 3:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          stride[num++] = tmp[0] + tmp[1] + tmp[2];
        }
      }
    }
    break;;
  case 4:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
          }
        }
      }
    }
    break;;
  case 5:
    for(int i=0;i<array[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int l=0;l<array[3].length;l++){
            tmp[3] = stride_offset[3] * l;
            for(int m=0;m<array[4].length;m++){
              tmp[4] = stride_offset[4] * m;
              stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
            }
          }
        }
      }
    }
    break;;
  case 6:
    for(int i=0;i<array[0].length;i++){
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
                stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5];
              }
            }
          }
        }
      }
    }
    break;;
  case 7:
    for(int i=0;i<array[0].length;i++){
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
                for(int p=0;p<array[6].length;p++){
                  tmp[6] = stride_offset[6] * p;
                  stride[num++] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6];
                }
              }
            }
          }
        }
      }
    }
    break;;
  }
}

/**
   NON-continuous PUT operation in local node
*/
static void _local_NON_continuous_put(const int dst_dims, const int src_dims, _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
				      char *dst, char *src, const size_t transfer_elmts, const size_t elmt_size)
{
  // Fix me: If the lengths and strides in src and dst are the same, this operation will be faster.

  size_t dst_stride[transfer_elmts], src_stride[transfer_elmts];
  _set_stride(dst_stride, dst_info, dst_dims);
  _set_stride(src_stride, src_info, src_dims);

  for(int i=0;i<transfer_elmts;i++)
    memcpy(dst+dst_stride[i], src+src_stride[i], elmt_size);
}

/**
   PUT operation in local node
*/
void _XMP_local_put(const int dst_continuous, const int src_continuous, const int dst_dims, const int src_dims, 
		    _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
		    _XMP_coarray_t *dst, void *src, 
		    const size_t transfer_coarray_elmts, const size_t transfer_array_elmts)
{
  uint64_t dst_point = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t elmt_size   = dst->elmt_size;

  if(dst_continuous && src_continuous){
    _local_continuous_put(dst_dims, src_dims, dst_info, src_info, dst, src, dst_point, src_point,
			  transfer_coarray_elmts, transfer_array_elmts, elmt_size);
  }
  else{
    if(transfer_coarray_elmts == transfer_array_elmts){
      _local_NON_continuous_put(dst_dims, src_dims, dst_info, src_info, (char *)dst->real_addr+dst_point, 
				(char *)src+src_point, transfer_coarray_elmts, elmt_size);
    }
    else if(transfer_array_elmts == 1){
      _local_NON_continuous_mput(dst_dims, (char *)dst->real_addr+dst_point, (char *)src+src_point, dst_info, elmt_size);
    }
    else{
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    }
  }
}
