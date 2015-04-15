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


