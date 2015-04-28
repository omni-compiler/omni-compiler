#include <string.h>
#include "xmp_internal.h"

/******************************************************************/
/* DESCRIPTION : Search maximum dimension which has all elements  */
/* ARGUMENT    : [IN] dims       : Number of dimensions of array  */
/*             : [IN] *array_info : Information of array          */
/* RETURN      : Maximum dimension                                */
/* EXAMPLE     : int a[10], b[10][20], c[10][20][30];             */
/*               a[:], a[0:10]        -> 0                        */
/*               a[0], a[1:9], a[::2] -> 1                        */
/*               b[:][:]              -> 0                        */
/*               b[1][:], b[2:2:2][:] -> 1                        */
/*               b[:][2:2], b[1][1]   -> 2                        */
/*               c[:][:][:]                 -> 0                  */
/*               c[2][:][:], c[2:2:2][:][:] -> 1                  */
/*               c[2][2:2][:]               -> 2                  */
/*               c[:][:][::2]               -> 3                  */
/******************************************************************/
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

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 1-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_1dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info, 
			     size_t element_size, const int flag)
{
  size_t buf1_offset = 0, tmp;
  size_t stride_offset = array_info[0].stride * array_info[0].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[0].stride == 1){
      element_size *= array_info[0].length;
      memcpy(buf1, buf2, element_size);
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp = stride_offset * i;
	memcpy(buf1 + buf1_offset, buf2 + tmp, element_size);
	buf1_offset += element_size;
      }
    }
    break;;
  case _XMP_UNPACK:
    if(array_info[0].stride == 1){
      element_size *= array_info[0].length;
      memcpy(buf1, buf2, element_size);
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp = stride_offset * i;
	memcpy(buf1 + tmp, buf2 + buf1_offset, element_size);
	buf1_offset += element_size;
      }
    }
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp = stride_offset * i;
      memcpy(buf1 + tmp, buf2, element_size);
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 2-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_2dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[2], stride_offset[2];

  for(int i=0;i<2;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[1].stride == 1){
      element_size *= array_info[1].length;
      for(int i=0;i<array_info[0].length;i++){
	memcpy(buf1 + buf1_offset, buf2 + stride_offset[0] * i, element_size);
	buf1_offset += element_size;
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1], element_size);
	  buf1_offset += element_size;
	}
      }
    }
    break;;
  case _XMP_UNPACK:
    if(array_info[1].stride == 1){
      element_size *= array_info[1].length;
      for(int i=0;i<array_info[0].length;i++){
	memcpy(buf1 + stride_offset[0] * i, buf2 + buf1_offset, element_size);
	buf1_offset += element_size;
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + tmp[0] + tmp[1], buf2 + buf1_offset, element_size);
	  buf1_offset += element_size;
	}
      }
    }
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        memcpy(buf1 + tmp[0] + tmp[1], buf2, element_size);
      }
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 3-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_3dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[3], stride_offset[3];

  for(int i=0;i<3;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[2].stride == 1){
      element_size *= array_info[2].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1], element_size);
	  buf1_offset += element_size;
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2], element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    break;;
  case _XMP_UNPACK:
    if(array_info[2].stride == 1){
      element_size *= array_info[2].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + tmp[0] + tmp[1], buf2 + buf1_offset, element_size);
	  buf1_offset += element_size;
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2 + buf1_offset, element_size);
	    buf1_offset += element_size;
	  }
        }
      }
    }
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2, element_size);
        }
      }
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 4-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_4dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[4], stride_offset[4];

  for(int i=0;i<4;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[3].stride == 1){
      element_size *= array_info[3].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2], element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3], element_size);
	      buf1_offset += element_size;
	    }
	  }
        }
      }
    }
    break;;
  case _XMP_UNPACK:
    if(array_info[3].stride == 1){
      element_size *= array_info[3].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2 + buf1_offset, element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
		     buf2 + buf1_offset, element_size);
	      buf1_offset += element_size;
	    }
          }
        }
      }
    }
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
                   buf2, element_size);
          }
        }
      }
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 5-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_5dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[5], stride_offset[5];

  for(int i=0;i<5;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[4].stride == 1){
      element_size *= array_info[4].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
		     element_size);
	      buf1_offset += element_size;
	    }
	  }
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
		       element_size);
		buf1_offset += element_size;
	      }
            }
          }
        }
      }
    }
    break;;
  case _XMP_UNPACK:
    if(array_info[4].stride == 1){
      element_size *= array_info[4].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
		     buf2 + buf1_offset, element_size);
	      buf1_offset += element_size;
	    }
	  }
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
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
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
                     buf2, element_size);
            }
          }
        }
      }
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 6-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_6dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[6], stride_offset[6];

  for(int i=0;i<6;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[5].stride == 1){
      element_size *= array_info[5].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
		       element_size);
		buf1_offset += element_size;
	      }
	    }
	  }
	}
      }
    }
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
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
    break;;
  case _XMP_UNPACK:
    if(array_info[5].stride == 1){
      element_size *= array_info[5].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
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
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
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
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array_info[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5],
                       buf2, element_size);
              }
            }
          }
        }
      }
    }
    break;;
  }
}

/********************************************************************/
/* DESCRIPTION : Memory copy with stride for 7-dimensional array    */
/* ARGUMENT    : [OUT] *buf1       : Pointer of destination         */
/*             : [IN] *buf2        : Pointer of source              */
/*             : [IN] *array_info  : Information of array           */
/*             : [IN] element_size : Elements size                  */
/*             : [IN] flag         : Kind of copy                   */
/********************************************************************/
void _XMP_stride_memcpy_7dim(char *buf1, const char *buf2, const _XMP_array_section_t *array_info,
                             size_t element_size, const int flag)
{
  size_t buf1_offset = 0;
  size_t tmp[7], stride_offset[7];

  for(int i=0;i<7;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[6].stride == 1){
      element_size *= array_info[6].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
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
    else{ 
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
		  tmp[5] = stride_offset[5] * p;
		  for(int q=0;q<array_info[6].length;q++){
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
    break;;
  case _XMP_UNPACK:
    if(array_info[6].stride == 1){
      element_size *= array_info[6].length;
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
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
    else{
      for(int i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(int j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(int k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(int m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(int n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(int p=0;p<array_info[5].length;p++){
		  tmp[5] = stride_offset[5] * p;
		  for(int q=0;q<array_info[6].length;q++){
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
    break;;
  case _XMP_SCALAR_MCOPY:
    for(int i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<array_info[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(int q=0;q<array_info[6].length;q++){
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
    break;;
  }
}
