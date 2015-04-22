#include "xmp_internal.h"
#define _XMP_UNROLLING (4)

static size_t _calc_start_offset(const _XMP_array_section_t* arrray_info, int dim)
{
  size_t start_offset = 0;

  for(int i=0;i<dim;i++)
    start_offset += arrray_info[i].start * arrray_info[i].distance;

  return start_offset;
}

static void _pack_7_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[6].distance;
  size_t start_offset = _calc_start_offset(src, 7);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_7dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_6dim(archive_ptr, src_ptr + start_offset, src, src[6].distance * src[6].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_5dim(archive_ptr, src_ptr + start_offset, src, src[5].distance * src[5].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_4dim(archive_ptr, src_ptr + start_offset, src, src[4].distance * src[4].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_3dim(archive_ptr, src_ptr + start_offset, src, src[3].distance * src[3].length, _XMP_PACK);
    break;;
  case 5:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, src[2].distance * src[2].length, _XMP_PACK);
    break;;
  case 6:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_6_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[5].distance;
  size_t start_offset = _calc_start_offset(src, 6);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_6dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_5dim(archive_ptr, src_ptr + start_offset, src, src[5].distance * src[5].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_4dim(archive_ptr, src_ptr + start_offset, src, src[4].distance * src[4].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_3dim(archive_ptr, src_ptr + start_offset, src, src[3].distance * src[3].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, src[2].distance * src[2].length, _XMP_PACK);
    break;;
  case 5:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_5_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[4].distance;
  size_t start_offset = _calc_start_offset(src, 5);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_5dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_4dim(archive_ptr, src_ptr + start_offset, src, src[4].distance * src[4].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_3dim(archive_ptr, src_ptr + start_offset, src, src[3].distance * src[3].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, src[2].distance * src[2].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_4_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[3].distance;
  size_t start_offset = _calc_start_offset(src, 4);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_4dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_3dim(archive_ptr, src_ptr + start_offset, src, src[3].distance * src[3].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, src[2].distance * src[2].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_3_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = src[2].distance;
  size_t start_offset = _calc_start_offset(src, 3);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_3dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, src[2].distance * src[2].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_2_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr, 
			      const int continuous_dim) // continuous_dim is 0 or 1
{
  size_t element_size = src[1].distance;
  size_t start_offset = _calc_start_offset(src, 2);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_2dim(archive_ptr, src_ptr + start_offset, src, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_1dim(archive_ptr, src_ptr + start_offset, src, src[1].distance * src[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_1_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr)
{
  // for(i=0;i<src[0].length;i++){
  //   src_offset = start_offset + (stride_offset * i);
  //   memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
  //   archive_offset += element_size;
  // }
  size_t element_size = src[0].distance;
  int repeat = src[0].length / _XMP_UNROLLING;
  int left   = src[0].length % _XMP_UNROLLING;
  size_t stride_offset = src[0].stride * element_size;
  size_t archive_offset = 0, src_offset;
  int i = 0;
  size_t start_offset = _calc_start_offset(src, 1);

  if(repeat == 0){
    for(i=0;i<left;i++){
      src_offset = start_offset + (stride_offset * i);
      archive_offset = i * element_size;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      src_offset = start_offset + (stride_offset * (i+2));
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;
    case 2 :
      src_offset = start_offset + (stride_offset * (i+1));
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;
    case 1 :
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
    }
  }
}

void _XMP_pack_coarray(char* archive_ptr, const char* src_ptr, const int src_dims, const _XMP_array_section_t* src)
{
  if(src_dims == 1){ 
    _pack_1_dim_array(src, archive_ptr, src_ptr);
    return;
  }

  // How depth is memory continuity ?
  int continuous_dim = _XMP_get_depth(src_dims, src);

  switch (src_dims){
  case 2:
    _pack_2_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  case 3:
    _pack_3_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  case 4:
    _pack_4_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  case 5:
    _pack_5_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  case 6:
    _pack_6_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  case 7:
    _pack_7_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    break;;
  default:
    _XMP_fatal("Dimension of coarray is too big");
    break;;
  }
}

static void _unpack_7_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[6].distance;
  size_t start_offset = _calc_start_offset(dst, 7);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_7dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_6dim(dst_ptr + start_offset, src_ptr, dst, dst[6].distance * dst[6].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_5dim(dst_ptr + start_offset, src_ptr, dst, dst[5].distance * dst[5].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_4dim(dst_ptr + start_offset, src_ptr, dst, dst[4].distance * dst[4].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, dst[3].distance * dst[3].length, _XMP_UNPACK);
    break;;
  case 5:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance * dst[2].length, _XMP_UNPACK);
    break;;
  case 6:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _unpack_6_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[5].distance;
  size_t start_offset = _calc_start_offset(dst, 6);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_6dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_5dim(dst_ptr + start_offset, src_ptr, dst, dst[5].distance * dst[5].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_4dim(dst_ptr + start_offset, src_ptr, dst, dst[4].distance * dst[4].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, dst[3].distance * dst[3].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance * dst[2].length, _XMP_UNPACK);
    break;;
  case 5:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _unpack_5_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[4].distance;
  size_t start_offset = _calc_start_offset(dst, 5);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_5dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_4dim(dst_ptr + start_offset, src_ptr, dst, dst[4].distance * dst[4].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, dst[3].distance * dst[3].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance * dst[2].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_4_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, 
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[3].distance;
  size_t start_offset = _calc_start_offset(dst, 4);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_4dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, dst[3].distance * dst[3].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance * dst[2].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_3_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[2].distance;
  size_t start_offset = _calc_start_offset(dst, 3);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance * dst[2].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_2_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  // continuous_dim is 0 or 1
  size_t element_size = dst[1].distance;
  size_t start_offset = _calc_start_offset(dst, 2);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_1dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance * dst[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}


static void _unpack_1_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, char* dst_ptr)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //    src_offset += element_size;
  //  }
  size_t element_size = dst[0].distance;
  int repeat = dst[0].length / _XMP_UNROLLING;
  int left   = dst[0].length % _XMP_UNROLLING;
  size_t stride_offset = dst[0].stride * element_size;
  size_t dst_offset, src_offset = 0;
  size_t start_offset = _calc_start_offset(dst, 1);
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = start_offset + (i * stride_offset);
      src_offset = i * element_size;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      dst_offset = start_offset + (stride_offset * (i+2));
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;
    case 2 :
      dst_offset = start_offset + (stride_offset * (i+1));
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;
    case 1:
      dst_offset = start_offset + (stride_offset * i);
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
    }
  }
}

static void _unpack_1_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr, char* dst_ptr)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //  }
  size_t element_size = dst[0].distance;
  int repeat = dst[0].length / _XMP_UNROLLING;
  int left   = dst[0].length % _XMP_UNROLLING;
  size_t start_offset  = dst[0].start  * element_size;
  size_t stride_offset = dst[0].stride * element_size;
  size_t dst_offset;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = start_offset + (i * stride_offset);
       memcpy(dst_ptr + dst_offset, src_ptr, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
 
      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      dst_offset = start_offset + (stride_offset * (i+2));
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
    case 2 :
      dst_offset = start_offset + (stride_offset * (i+1));
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
    case 1:
      dst_offset = start_offset + (stride_offset * i);
      memcpy(dst_ptr + dst_offset, src_ptr, element_size);
    }
  }
}

void _XMP_unpack_coarray(char *dst_ptr, const int dst_dims, const char* src_ptr, 
			 const _XMP_array_section_t* dst, const int flag)
{
  // flag == 0; src_offset is changed.
  // flag == 1; src_offset is fixed for scalar_mput or scalar_mget.

  if(dst_dims == 1){
    if(flag == 0)
      _unpack_1_dim_array(dst, src_ptr, dst_ptr);
    else
      _unpack_1_dim_array_fixed_src(dst, src_ptr, dst_ptr);
    
    return;
  }

  if(flag == 0){
    int continuous_dim = _XMP_get_depth(dst_dims, dst);
    switch (dst_dims){
    case 2:
      _unpack_2_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    case 3:
      _unpack_3_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    case 4:
      _unpack_4_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    case 5:
      _unpack_5_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    case 6:
      _unpack_6_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    case 7:
      _unpack_7_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
      break;;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;;
    }
  }
  else{
    size_t start_offset = _calc_start_offset(dst, dst_dims);
    switch (dst_dims){
    case 2:
      _XMP_stride_memcpy_2dim(dst_ptr + start_offset, src_ptr, dst, dst[1].distance, _XMP_MPUT);
      break;;
    case 3:
      _XMP_stride_memcpy_3dim(dst_ptr + start_offset, src_ptr, dst, dst[2].distance, _XMP_MPUT);
      break;;
    case 4:
      _XMP_stride_memcpy_4dim(dst_ptr + start_offset, src_ptr, dst, dst[3].distance, _XMP_MPUT);
      break;;
    case 5:
      _XMP_stride_memcpy_5dim(dst_ptr + start_offset, src_ptr, dst, dst[4].distance, _XMP_MPUT);
      break;;
    case 6:
      _XMP_stride_memcpy_6dim(dst_ptr + start_offset, src_ptr, dst, dst[5].distance, _XMP_MPUT);
      break;;
    case 7:
      _XMP_stride_memcpy_7dim(dst_ptr + start_offset, src_ptr, dst, dst[6].distance, _XMP_MPUT);
      break;;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;;
    }
  }
}
