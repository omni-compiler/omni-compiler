#include <string.h>
#include "xmp_internal.h"

static int _is_all_elmt(const _XMP_array_section_t* array_info, const int dim)
{
  if(array_info[dim].start == 0 && array_info[dim].length == array_info[dim].elmts){
    return _XMP_N_INT_TRUE;
  }
  else{
    return _XMP_N_INT_FALSE;
  }
}

// How depth is memory continuity ?
// when depth is 0, all dimension is not continuous.
// eg. a[:][2:2:1]    -> depth is 1. The last diemnsion is continuous.
//     a[:][2:2:2]    -> depth is 0.
//     a[:][:]        -> depth is 2. But, this function is called when array is not continuous.
//                       So depth does not become 2.
//     b[:][:][1:2:2]   -> depth is 0.
//     b[:][:][1]       -> depth is 1.
//     b[:][2:2:2][1]   -> depth is 1.
//     b[:][2:2:2][:]   -> depth is 1.
//     b[2:2:2][:][:]   -> depth is 2.
//     b[2:2][2:2][2:2] -> depth is 1.
//     c[1:2][1:2][1:2][1:2] -> depth is 1.
//     c[1:2:2][:][:][:]     -> depth is 3.
//     c[1:2:2][::2][:][:]   -> depth is 2.
int _XMP_get_depth(const int dims, const _XMP_array_section_t* array_info)  // 7 >= dims >= 2
{
  if(dims == 2){
    if(array_info[1].stride == 1)
      return 1;
    else
      return 0;
  }

  int all_elmt_flag[_XMP_N_MAX_DIM];
  for(int i=1;i<dims;i++)
    all_elmt_flag[i] = _is_all_elmt(array_info, i);

  switch (dims){
  case 3:
    if(all_elmt_flag[1] && all_elmt_flag[2]){
      return 2;
    }
    else if(array_info[2].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  case 4:
    if(all_elmt_flag[1] && all_elmt_flag[2] && all_elmt_flag[3]){
      return 3;
    }
    else if(all_elmt_flag[2] && all_elmt_flag[3]){
      return 2;
    }
    else if(array_info[3].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
    break;;

  case 5:
    if(all_elmt_flag[1] && all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4]){
      return 4;
    }
    else if(all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4]){
      return 3;
    }
    else if(all_elmt_flag[3] && all_elmt_flag[4]){
      return 2;
    }
    else if(array_info[4].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
    break;;
  case 6:
    if(all_elmt_flag[1] && all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4] &&
       all_elmt_flag[5]){
      return 5;
    }
    else if(all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4] && all_elmt_flag[5]){
      return 4;
    }
    else if(all_elmt_flag[3] && all_elmt_flag[4] && all_elmt_flag[5]){
      return 3;
    }
    else if(all_elmt_flag[4] && all_elmt_flag[5]){
      return 2;
    }
    else if(array_info[5].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
    break;;

  case 7:
    if(all_elmt_flag[1] && all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4] &&
       all_elmt_flag[5] && all_elmt_flag[6]){
      return 6;
    }
    else if(all_elmt_flag[2] && all_elmt_flag[3] && all_elmt_flag[4] &&
            all_elmt_flag[5] && all_elmt_flag[6]){
      return 5;
    }
    else if(all_elmt_flag[3] && all_elmt_flag[4] && all_elmt_flag[5] && all_elmt_flag[6]){
      return 4;
    }
    else if(all_elmt_flag[4] && all_elmt_flag[5] && all_elmt_flag[6]){
      return 3;
    }
    else if(all_elmt_flag[5] && all_elmt_flag[6]){
      return 2;
    }
    else if(array_info[6].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
    break;;
  default:
    _XMP_fatal("Dimensions of Coarray is too big.");
    return -1;
    break;;
  }
}

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


