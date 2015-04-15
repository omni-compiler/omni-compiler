#include <string.h>
#include "xmp_internal.h"

/**
   Continuous PUT/GET operation in local node
*/
static void _local_continuous_copy(char *dst, const void *src, const size_t dst_elmts, const size_t src_elmts, 
				   const size_t elmt_size)
{
  if(dst_elmts == src_elmts){
    memcpy(dst, src, dst_elmts * elmt_size);
  }
  else if(src_elmts == 1){ /* a[0:100]:[1] = b[2]; */
    for(int i=0;i<dst_elmts;i++)
      memcpy(dst+elmt_size*i, src, elmt_size);
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

/**
   NON-continuous multiple put/get for scalar variable
   e.g. a[0:100:2]:[1] = b;   a[0:100:2] = b:[2];
 */
static void _local_NON_continuous_scalar_mcopy(const int dst_dims, char *dst, const char *src, 
					       _XMP_array_section_t *dst_info, const size_t elmt_size)
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

/**
   Set stride size of array[dim] to stride[]
 */
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
   NON-continuous PUT/GET operation in local node
*/
static void _local_NON_continuous_copy(const int dst_dims, const int src_dims, _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
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
   PUT operation in local node. "dst" is a coarray
*/
void _XMP_local_put(const int dst_continuous, const int src_continuous, const int dst_dims, const int src_dims, 
		    _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info,
		    _XMP_coarray_t *dst, void *src, const size_t dst_elmts, const size_t src_elmts)
{
  uint64_t dst_point = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t elmt_size   = dst->elmt_size;

  if(dst_continuous && src_continuous){
    _local_continuous_copy((char *)dst->real_addr+dst_point, (char *)src+src_point,
			   dst_elmts, src_elmts, elmt_size);
  }
  else{
    if(dst_elmts == src_elmts){
      _local_NON_continuous_copy(dst_dims, src_dims, dst_info, src_info, (char *)dst->real_addr+dst_point, 
				 (char *)src+src_point, dst_elmts, elmt_size);
    }
    else if(src_elmts == 1){ /* a[0:100:2]:[1] = b[2]; */
      _local_NON_continuous_scalar_mcopy(dst_dims, (char *)dst->real_addr+dst_point, 
					 (char *)src+src_point, dst_info, elmt_size);
    }
    else{
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    }
  }
}


/**
   GET operation in local node. "src" is a coarray
*/
void _XMP_local_get(const int src_continuous, const int dst_continuous, const int src_dims, const int dst_dims,
                    _XMP_array_section_t *src_info, _XMP_array_section_t *dst_info,
                    _XMP_coarray_t *src, void *dst, const size_t src_elmts, const size_t dst_elmts)
{
  uint64_t dst_point = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_point = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t elmt_size   = src->elmt_size;

  if(dst_continuous && src_continuous){
    _local_continuous_copy((char *)dst+dst_point, (char *)src->real_addr+src_point, dst_elmts, src_elmts, 
			   elmt_size);
  }
  else{
    if(src_elmts == dst_elmts){
      _local_NON_continuous_copy(dst_dims, src_dims, dst_info, src_info, (char *)dst+dst_point,
				 (char *)src->real_addr+src_point, dst_elmts, elmt_size);
    }
    else if(src_elmts == 1){ /* a[0:100:2] = b[2]:[1]; */
      _local_NON_continuous_scalar_mcopy(dst_dims, (char *)dst+dst_point,
					 (char *)src->real_addr+src_point, dst_info, elmt_size);
    }
    else{
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    }
  }
}
