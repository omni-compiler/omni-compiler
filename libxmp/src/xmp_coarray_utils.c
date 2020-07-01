#include <string.h>
#include "xmp_internal.h"
#include "xmp_math_function.h"

/***********************************************************/
/* DESCRIPTION : Check the size is less than SIZE_MAX      */
/* ARGUMENT    : [IN] s : size                             */
/***********************************************************/
void _XMP_check_less_than_SIZE_MAX(const long s)
{
  if(s > SIZE_MAX){
    fprintf(stderr, "Coarray size is %ld. Coarray size must be < %zu\n", s, SIZE_MAX);
    _XMP_fatal_nomsg();
  }
}

/***********************************************************/
/* DESCRIPTION : Caclulate offset                          */
/* ARGUMENT    : [IN] *array_info : Information of array   */
/*               [IN] dims        : Number of dimensions   */
/***********************************************************/
size_t _XMP_get_offset(const _XMP_array_section_t *array_info, const int dims)
{
  size_t offset = 0;
  for(int i=0;i<dims;i++)
    offset += array_info[i].start * array_info[i].distance;

  return offset;
}

/****************************************************************************/
/* DESCRIPTION : Calculate maximum chunk for copy                           */
/* ARGUMENT    : [IN] dst_dims  : Number of dimensions of destination array */
/*               [IN] src_dims  : Number of dimensions of source array      */
/*               [IN] *dst_info : Information of destination array          */
/*               [IN] *src_info : Information of source array               */
/* RETURN     : Maximum chunk for copy                                      */
/* NOTE       : This function is used to reduce callings of memcpy()        */
/* EXAMPLE    : int a[10]:[*], b[10]:[*], c[5][2];                          */
/*              a[0:10:2]:[1] = b[0:10:2] -> 4                              */
/*              c[1:2][:]:[*] = b[0:4]    -> 16                             */
/****************************************************************************/
size_t _XMP_calc_max_copy_chunk(const int dst_dims, const int src_dims,
				const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info)
{
  int dst_copy_chunk_dim = _XMP_get_dim_of_allelmts(dst_dims, dst_info);
  int src_copy_chunk_dim = _XMP_get_dim_of_allelmts(src_dims, src_info);
  size_t dst_copy_chunk  = _XMP_calc_copy_chunk(dst_copy_chunk_dim, dst_info);
  size_t src_copy_chunk  = _XMP_calc_copy_chunk(src_copy_chunk_dim, src_info);

  return _XMP_M_MIN(dst_copy_chunk, src_copy_chunk);
}


/**********************************************************************/
/* DESCRIPTION : Check of dst and src overlap                         */
/* ARGUMENT    : [IN] *dst_start : Start pointer of destination array */
/*               [IN] *dst_end   : End pointer of destination array   */
/*               [IN] *src_start : Start pointer of source array      */
/*               [IN] *src_end   : End pointer of source array        */
/* NOTE       : When a[0:5]:[1] = a[1:5], return true.                */
/**********************************************************************/
_Bool _XMP_check_overlapping(const char *dst_start, const char *dst_end,
			     const char *src_start, const char *src_end)
{
  return (dst_start <= src_start && src_start < dst_end) ||
         (src_start <= dst_start && dst_start < src_end);
}

/********************************************************************************/
/* DESCRIPTION : Execute copy operation in only local node for contiguous array */
/* ARGUMENT    : [OUT] *dst     : Pointer of destination array                  */
/*               [IN] *src      : Pointer of source array                       */
/*               [IN] dst_elmts : Number of elements of destination array       */
/*               [IN] src_elmts : Number of elements of source array            */
/*               [IN] elmt_size : Element size                                  */
/* NOTE       : This function is called by both put and get functions           */
/********************************************************************************/
void _XMP_local_contiguous_copy(char *dst, const char *src, const size_t dst_elmts,
				const size_t src_elmts, const size_t elmt_size)
{
  if(dst_elmts == src_elmts){ /* a[0:100]:[1] = b[1:100]; or a[0:100] = b[1:100]:[1];*/
    size_t offset = dst_elmts * elmt_size;
    if(_XMP_check_overlapping(dst, dst+offset, src, src+offset)){
      memmove(dst, src, offset);
    }
    else
      memcpy(dst, src, offset);
  }
  else if(src_elmts == 1){    /* a[0:100]:[1] = b[1]; or a[0:100] = b[1]:[1]; */
    size_t offset = 0;
    for(size_t i=0;i<dst_elmts;i++){
      if(dst+offset != src)
	memcpy(dst+offset, src, elmt_size);

      offset += elmt_size;
    }
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/******************************************************************/
/* DESCRIPTION : Search maximum dimension which has all elements  */
/* ARGUMENT    : [IN] dims       : Number of dimensions of array  */
/*             : [IN] *array_info : Information of array          */
/* RETURN      : Maximum dimension                                */
/* EXAMPLE     : int a[10], b[10][20], c[10][20][30];             */
/*               a[:]                  -> 0                       */
/*               a[0], a[1:9], a[:3:2] -> 1                       */
/*               b[:][:]               -> 0                       */
/*               b[1][:], b[2:2:2][:]  -> 1                       */
/*               b[:][2:2], b[1][1]    -> 2                       */
/*               c[:][:][:]                 -> 0                  */
/*               c[2][:][:], c[2:2:2][:][:] -> 1                  */
/*               c[2][2:2][:]               -> 2                  */
/*               c[:][:][:3:2]              -> 3                  */
/******************************************************************/
int _XMP_get_dim_of_allelmts(const int dims, const _XMP_array_section_t* array_info)
{
  int val = dims;

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
  size_t buf1_offset = 0;
  size_t tmp, stride_offset = array_info[0].stride * array_info[0].distance;

  switch (flag){
  case _XMP_PACK:
    if(array_info[0].stride == 1){
      memcpy(buf1, buf2, element_size*array_info[0].length);
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp = stride_offset * i;
	memcpy(buf1 + buf1_offset, buf2 + tmp, element_size);
	buf1_offset += element_size;
      }
    }
    break;
  case _XMP_UNPACK:
    if(array_info[0].stride == 1){
      memcpy(buf1, buf2, element_size*array_info[0].length);
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp = stride_offset * i;
	memcpy(buf1 + tmp, buf2 + buf1_offset, element_size);
	buf1_offset += element_size;
      }
    }
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp = stride_offset * i;
      if(buf1 + tmp != buf2)
	memcpy(buf1 + tmp, buf2, element_size);
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	memcpy(buf1 + buf1_offset, buf2 + stride_offset[0] * i, element_size);
	buf1_offset += element_size;
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1], element_size);
	  buf1_offset += element_size;
	}
      }
    }
    break;
  case _XMP_UNPACK:
    if(array_info[1].stride == 1){
      element_size *= array_info[1].length;
      for(size_t i=0;i<array_info[0].length;i++){
	memcpy(buf1 + stride_offset[0] * i, buf2 + buf1_offset, element_size);
	buf1_offset += element_size;
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + tmp[0] + tmp[1], buf2 + buf1_offset, element_size);
	  buf1_offset += element_size;
	}
      }
    }
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
	if(buf1 + tmp[0] + tmp[1] != buf2)
	  memcpy(buf1 + tmp[0] + tmp[1], buf2, element_size);
      }
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1], element_size);
	  buf1_offset += element_size;
	}
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2], element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    break;
  case _XMP_UNPACK:
    if(array_info[2].stride == 1){
      element_size *= array_info[2].length;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  memcpy(buf1 + tmp[0] + tmp[1], buf2 + buf1_offset, element_size);
	  buf1_offset += element_size;
	}
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2 + buf1_offset, element_size);
	    buf1_offset += element_size;
	  }
        }
      }
    }
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
	  if(buf1 + tmp[0] + tmp[1] + tmp[2] != buf2)
          memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2, element_size);
        }
      }
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2], element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + buf1_offset, buf2 + tmp[0] + tmp[1] + tmp[2] + tmp[3], element_size);
	      buf1_offset += element_size;
	    }
	  }
        }
      }
    }
    break;
  case _XMP_UNPACK:
    if(array_info[3].stride == 1){
      element_size *= array_info[3].length;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    memcpy(buf1 + tmp[0] + tmp[1] + tmp[2], buf2 + buf1_offset, element_size);
	    buf1_offset += element_size;
	  }
	}
      }
    }
    else{
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
		     buf2 + buf1_offset, element_size);
	      buf1_offset += element_size;
	    }
          }
        }
      }
    }
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    if(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] != buf2)
	      memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3],
		     buf2, element_size);
          }
        }
      }
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
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
    break;
  case _XMP_UNPACK:
    if(array_info[4].stride == 1){
      element_size *= array_info[4].length;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
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
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(size_t n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      if(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] != buf2)
		memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4],
		       buf2, element_size);
            }
          }
        }
      }
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
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
    break;
  case _XMP_UNPACK:
    if(array_info[5].stride == 1){
      element_size *= array_info[5].length;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
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
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(size_t n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(size_t p=0;p<array_info[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		if(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] != buf2)
                memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5],
                       buf2, element_size);
              }
            }
          }
        }
      }
    }
    break;
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
		  tmp[5] = stride_offset[5] * p;
		  for(size_t q=0;q<array_info[6].length;q++){
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
    break;
  case _XMP_UNPACK:
    if(array_info[6].stride == 1){
      element_size *= array_info[6].length;
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
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
      for(size_t i=0;i<array_info[0].length;i++){
	tmp[0] = stride_offset[0] * i;
	for(size_t j=0;j<array_info[1].length;j++){
	  tmp[1] = stride_offset[1] * j;
	  for(size_t k=0;k<array_info[2].length;k++){
	    tmp[2] = stride_offset[2] * k;
	    for(size_t m=0;m<array_info[3].length;m++){
	      tmp[3] = stride_offset[3] * m;
	      for(size_t n=0;n<array_info[4].length;n++){
		tmp[4] = stride_offset[4] * n;
		for(size_t p=0;p<array_info[5].length;p++){
		  tmp[5] = stride_offset[5] * p;
		  for(size_t q=0;q<array_info[6].length;q++){
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
    break;
  case _XMP_SCALAR_MCOPY:
    for(size_t i=0;i<array_info[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(size_t j=0;j<array_info[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(size_t k=0;k<array_info[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(size_t m=0;m<array_info[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(size_t n=0;n<array_info[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(size_t p=0;p<array_info[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(size_t q=0;q<array_info[6].length;q++){
                  tmp[6] = stride_offset[6] * q;
		  if(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] != buf2)
		    memcpy(buf1 + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6],
			   buf2, element_size);
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
int _is_all_element(const _XMP_array_section_t *array_info, int dim){
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
int _check_round(const _XMP_array_section_t *array, const int dim)
{
  return array[dim].length * array[dim].stride - array[dim].elmts == 0;
}

/**
   If 1dim array has a constant stride, return TRUE (Always TRUE)
*/
int _is_constant_stride_1dim()
{
  return _XMP_N_INT_TRUE;
}

/********************************************************************/
/* DESCRIPTION : Is 2dim array has a constant stride ?              */
/* ARGUMENT    : [IN] *array_info : Information of array            */
/* RETURN:     : If 2dim array has a constant stride, return TRUE   */
/********************************************************************/
int _is_constant_stride_2dim(const _XMP_array_section_t *array_info)
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
int _is_constant_stride_3dim(const _XMP_array_section_t *array_info)
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
int _is_constant_stride_4dim(const _XMP_array_section_t *array_info)
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
int _is_constant_stride_5dim(const _XMP_array_section_t *array_info)
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
int _is_constant_stride_6dim(const _XMP_array_section_t *array_info)
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
int _is_constant_stride_7dim(const _XMP_array_section_t *array_info)
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
int _is_the_same_shape_except_for_start(const _XMP_array_section_t *array1_info,
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
// size_t _XMP_calc_stride(const _XMP_array_section_t *array_info, const int dims,
long _XMP_calc_stride(const _XMP_array_section_t *array_info, const int dims,
			const size_t chunk_size)
{
  // uint64_t stride_offset[dims], tmp[dims];
  // size_t stride[2];
  long stride_offset[dims], tmp[dims];
  long stride[2];

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
