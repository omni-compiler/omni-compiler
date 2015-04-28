#include "xmp_internal.h"
#define _XMP_UNROLLING (4)

static void _pack_7_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src,
			      const int continuous_dim)
{
  size_t element_size = src_info[6].distance;
  size_t start_offset = _XMP_get_offset(src_info, 7);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_7dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_6dim(archive, src + start_offset, src_info, src_info[6].distance * src_info[6].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_5dim(archive, src + start_offset, src_info, src_info[5].distance * src_info[5].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_4dim(archive, src + start_offset, src_info, src_info[4].distance * src_info[4].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_3dim(archive, src + start_offset, src_info, src_info[3].distance * src_info[3].length, _XMP_PACK);
    break;;
  case 5:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, src_info[2].distance * src_info[2].length, _XMP_PACK);
    break;;
  case 6:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_6_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src,
			      const int continuous_dim)
{
  size_t element_size = src_info[5].distance;
  size_t start_offset = _XMP_get_offset(src_info, 6);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_6dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_5dim(archive, src + start_offset, src_info, src_info[5].distance * src_info[5].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_4dim(archive, src + start_offset, src_info, src_info[4].distance * src_info[4].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_3dim(archive, src + start_offset, src_info, src_info[3].distance * src_info[3].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, src_info[2].distance * src_info[2].length, _XMP_PACK);
    break;;
  case 5:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_5_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src,
			      const int continuous_dim)
{
  size_t element_size = src_info[4].distance;
  size_t start_offset = _XMP_get_offset(src_info, 5);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_5dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_4dim(archive, src + start_offset, src_info, src_info[4].distance * src_info[4].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_3dim(archive, src + start_offset, src_info, src_info[3].distance * src_info[3].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, src_info[2].distance * src_info[2].length, _XMP_PACK);
    break;;
  case 4:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_4_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src,
			      const int continuous_dim)
{
  size_t element_size = src_info[3].distance;
  size_t start_offset = _XMP_get_offset(src_info, 4);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_4dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_3dim(archive, src + start_offset, src_info, src_info[3].distance * src_info[3].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, src_info[2].distance * src_info[2].length, _XMP_PACK);
    break;;
  case 3:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_3_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src,
			      const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = src_info[2].distance;
  size_t start_offset = _XMP_get_offset(src_info, 3);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_3dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, src_info[2].distance * src_info[2].length, _XMP_PACK);
    break;;
  case 2:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_2_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src, 
			      const int continuous_dim) // continuous_dim is 0 or 1
{
  size_t element_size = src_info[1].distance;
  size_t start_offset = _XMP_get_offset(src_info, 2);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_2dim(archive, src + start_offset, src_info, element_size, _XMP_PACK);
    break;;
  case 1:
    _XMP_stride_memcpy_1dim(archive, src + start_offset, src_info, src_info[1].distance * src_info[1].length, _XMP_PACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _pack_1_dim_array(const _XMP_array_section_t* src_info, char* archive, const char* src)
{
  // for(i=0;i<src_info[0].length;i++){
  //   src_offset = start_offset + (stride_offset * i);
  //   memcpy(archive + archive_offset, src + src_offset, element_size);
  //   archive_offset += element_size;
  // }
  size_t element_size = src_info[0].distance;
  int repeat = src_info[0].length / _XMP_UNROLLING;
  int left   = src_info[0].length % _XMP_UNROLLING;
  size_t stride_offset = src_info[0].stride * element_size;
  size_t archive_offset = 0, src_offset;
  int i = 0;
  size_t start_offset = _XMP_get_offset(src_info, 1);

  if(repeat == 0){
    for(i=0;i<left;i++){
      src_offset = start_offset + (stride_offset * i);
      archive_offset = i * element_size;
      memcpy(archive + archive_offset, src + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;

      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      src_offset = start_offset + (stride_offset * (i+2));
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;
    case 2 :
      src_offset = start_offset + (stride_offset * (i+1));
      memcpy(archive + archive_offset, src + src_offset, element_size);
      archive_offset += element_size;
    case 1 :
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive + archive_offset, src + src_offset, element_size);
    }
  }
}

void _XMP_pack_coarray(char* archive, const char* src, const int src_dims, const _XMP_array_section_t* src_info)
{
  if(src_dims == 1){ 
    _pack_1_dim_array(src_info, archive, src);
    return;
  }

  // How depth is memory continuity ?
  int continuous_dim = src_dims - _XMP_get_dim_of_allelmts(src_dims, src_info);

  switch (src_dims){
  case 2:
    _pack_2_dim_array(src_info, archive, src, continuous_dim);
    break;;
  case 3:
    _pack_3_dim_array(src_info, archive, src, continuous_dim);
    break;;
  case 4:
    _pack_4_dim_array(src_info, archive, src, continuous_dim);
    break;;
  case 5:
    _pack_5_dim_array(src_info, archive, src, continuous_dim);
    break;;
  case 6:
    _pack_6_dim_array(src_info, archive, src, continuous_dim);
    break;;
  case 7:
    _pack_7_dim_array(src_info, archive, src, continuous_dim);
    break;;
  default:
    _XMP_fatal("Dimension of coarray is too big");
    break;;
  }
}

static void _unpack_7_dim_array(const _XMP_array_section_t* dst_info, const char* src,
				char* dst, const int continuous_dim)
{
  size_t element_size = dst_info[6].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 7);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_7dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_6dim(dst + start_offset, src, dst_info, dst_info[6].distance * dst_info[6].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_5dim(dst + start_offset, src, dst_info, dst_info[5].distance * dst_info[5].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_4dim(dst + start_offset, src, dst_info, dst_info[4].distance * dst_info[4].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, dst_info[3].distance * dst_info[3].length, _XMP_UNPACK);
    break;;
  case 5:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[2].distance * dst_info[2].length, _XMP_UNPACK);
    break;;
  case 6:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _unpack_6_dim_array(const _XMP_array_section_t* dst_info, const char* src,
				char* dst, const int continuous_dim)
{
  size_t element_size = dst_info[5].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 6);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_6dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_5dim(dst + start_offset, src, dst_info, dst_info[5].distance * dst_info[5].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_4dim(dst + start_offset, src, dst_info, dst_info[4].distance * dst_info[4].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, dst_info[3].distance * dst_info[3].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[2].distance * dst_info[2].length, _XMP_UNPACK);
    break;;
  case 5:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;;
  }
}

static void _unpack_5_dim_array(const _XMP_array_section_t* dst_info, const char* src,
				char* dst, const int continuous_dim)
{
  size_t element_size = dst_info[4].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 5);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_5dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_4dim(dst + start_offset, src, dst_info, dst_info[4].distance * dst_info[4].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, dst_info[3].distance * dst_info[3].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[2].distance * dst_info[2].length, _XMP_UNPACK);
    break;;
  case 4:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_4_dim_array(const _XMP_array_section_t* dst_info, const char* src, 
				char* dst, const int continuous_dim)
{
  size_t element_size = dst_info[3].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 4);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_4dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, dst_info[3].distance * dst_info[3].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[2].distance * dst_info[2].length, _XMP_UNPACK);
    break;;
  case 3:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_3_dim_array(const _XMP_array_section_t* dst_info, const char* src,
                                char* dst, const int continuous_dim)
{
  size_t element_size = dst_info[2].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 3);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[2].distance * dst_info[2].length, _XMP_UNPACK);
    break;;
  case 2:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

static void _unpack_2_dim_array(const _XMP_array_section_t* dst_info, const char* src,
				char* dst, const int continuous_dim)
{
  // continuous_dim is 0 or 1
  size_t element_size = dst_info[1].distance;
  size_t start_offset = _XMP_get_offset(dst_info, 2);

  switch (continuous_dim){
  case 0:
    _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, element_size, _XMP_UNPACK);
    break;;
  case 1:
    _XMP_stride_memcpy_1dim(dst + start_offset, src, dst_info, dst_info[1].distance * dst_info[1].length, _XMP_UNPACK);
    break;;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}


static void _unpack_1_dim_array(const _XMP_array_section_t* dst_info, const char* src, char* dst)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //    src_offset += element_size;
  //  }
  size_t element_size = dst_info[0].distance;
  int repeat = dst_info[0].length / _XMP_UNROLLING;
  int left   = dst_info[0].length % _XMP_UNROLLING;
  size_t stride_offset = dst_info[0].stride * element_size;
  size_t dst_offset, src_offset = 0;
  size_t start_offset = _XMP_get_offset(dst_info, 1);
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = start_offset + (i * stride_offset);
      src_offset = i * element_size;
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;

      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      dst_offset = start_offset + (stride_offset * (i+2));
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;
    case 2 :
      dst_offset = start_offset + (stride_offset * (i+1));
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;
    case 1:
      dst_offset = start_offset + (stride_offset * i);
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
}

static void _unpack_1_dim_array_fixed_src(const _XMP_array_section_t* dst_info, const char* src, char* dst)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //  }
  size_t element_size = dst_info[0].distance;
  int repeat = dst_info[0].length / _XMP_UNROLLING;
  int left   = dst_info[0].length % _XMP_UNROLLING;
  size_t start_offset  = dst_info[0].start  * element_size;
  size_t stride_offset = dst_info[0].stride * element_size;
  size_t dst_offset;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst + dst_offset, src, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst + dst_offset, src, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src, element_size);
 
      dst_offset += stride_offset;
      memcpy(dst + dst_offset, src, element_size);
 
      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      dst_offset = start_offset + (stride_offset * (i+2));
      memcpy(dst + dst_offset, src, element_size);
    case 2 :
      dst_offset = start_offset + (stride_offset * (i+1));
      memcpy(dst + dst_offset, src, element_size);
    case 1:
      dst_offset = start_offset + (stride_offset * i);
      memcpy(dst + dst_offset, src, element_size);
    }
  }
}

void _XMP_unpack_coarray(char *dst, const int dst_dims, const char* src, 
			 const _XMP_array_section_t* dst_info, const int flag)
{
  if(dst_dims == 1){
    if(flag == _XMP_UNPACK)
      _unpack_1_dim_array(dst_info, src, dst);
    else if(flag == _XMP_SCALAR_MCOPY)
      _unpack_1_dim_array_fixed_src(dst_info, src, dst);
    else
      _XMP_fatal("Unexpected error !");
    return;
  }

  if(flag == _XMP_UNPACK){
    int continuous_dim = dst_dims - _XMP_get_dim_of_allelmts(dst_dims, dst_info);
    switch (dst_dims){
    case 2:
      _unpack_2_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    case 3:
      _unpack_3_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    case 4:
      _unpack_4_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    case 5:
      _unpack_5_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    case 6:
      _unpack_6_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    case 7:
      _unpack_7_dim_array(dst_info, src, dst, continuous_dim);
      break;;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;;
    }
  }
  else if(flag == _XMP_SCALAR_MCOPY){
    size_t start_offset = _XMP_get_offset(dst_info, dst_dims);
    switch (dst_dims){
    case 2:
      _XMP_stride_memcpy_2dim(dst + start_offset, src, dst_info, dst_info[1].distance, _XMP_SCALAR_MCOPY);
      break;;
    case 3:
      _XMP_stride_memcpy_3dim(dst + start_offset, src, dst_info, dst_info[2].distance, _XMP_SCALAR_MCOPY);
      break;;
    case 4:
      _XMP_stride_memcpy_4dim(dst + start_offset, src, dst_info, dst_info[3].distance, _XMP_SCALAR_MCOPY);
      break;;
    case 5:
      _XMP_stride_memcpy_5dim(dst + start_offset, src, dst_info, dst_info[4].distance, _XMP_SCALAR_MCOPY);
      break;;
    case 6:
      _XMP_stride_memcpy_6dim(dst + start_offset, src, dst_info, dst_info[5].distance, _XMP_SCALAR_MCOPY);
      break;;
    case 7:
      _XMP_stride_memcpy_7dim(dst + start_offset, src, dst_info, dst_info[6].distance, _XMP_SCALAR_MCOPY);
      break;;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;;
    }
  }
  else{
    _XMP_fatal("Unexpected error !");
  }
}
