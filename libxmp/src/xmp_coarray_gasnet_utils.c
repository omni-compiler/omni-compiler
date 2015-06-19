#include "xmp_internal.h"
#define _XMP_UNROLLING (4)

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 7-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_7dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, src_info, src_info[2].distance, _XMP_PACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, src_info, src_info[3].distance, _XMP_PACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, src_info, src_info[4].distance, _XMP_PACK);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst, src, src_info, src_info[5].distance, _XMP_PACK);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst, src, src_info, src_info[6].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 6-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_6dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, src_info, src_info[2].distance, _XMP_PACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, src_info, src_info[3].distance, _XMP_PACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, src_info, src_info[4].distance, _XMP_PACK);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst, src, src_info, src_info[5].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 5-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_5dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, src_info, src_info[2].distance, _XMP_PACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, src_info, src_info[3].distance, _XMP_PACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, src_info, src_info[4].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 4-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_4dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, src_info, src_info[2].distance, _XMP_PACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, src_info, src_info[3].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 3-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_3dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, src_info, src_info[2].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 2-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_2dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src, 
			     const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, src_info, src_info[0].distance, _XMP_PACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, src_info, src_info[1].distance, _XMP_PACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute pack operation for 1-dimensional array          */
/* ARGUMENT    : [IN] *src_info       : Information of source array      */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] *src            : Pointer of source array          */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _pack_1dim_array(const _XMP_array_section_t* src_info, char* dst, const char* src,
			     const int dim_of_allelmts)
{
  if(dim_of_allelmts == 0){
    memcpy(dst, src, src_info[0].distance * src_info[0].elmts);
    return;
  }
  
  // for(i=0;i<src_info[0].length;i++){
  //   src_offset = stride_offset * i;
  //   memcpy(dst + archive_offset, src + src_offset, element_size);
  //   archive_offset += element_size;
  // }
  size_t element_size = src_info[0].distance;
  int repeat = src_info[0].length / _XMP_UNROLLING;
  int left   = src_info[0].length % _XMP_UNROLLING;
  size_t stride_offset = src_info[0].stride * element_size;
  size_t dst_offset = 0, src_offset;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      src_offset = stride_offset * i;
      dst_offset = i * element_size;
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      src_offset = stride_offset * i;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;

      src_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;

      src_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;

      src_offset += stride_offset;
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;

      i += _XMP_UNROLLING;
    }

    switch (left) {
    case 3 :
      src_offset = stride_offset * (i+2);
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;
    case 2 :
      src_offset = stride_offset * (i+1);
      memcpy(dst + dst_offset, src + src_offset, element_size);
      dst_offset += element_size;
    case 1 :
      src_offset = stride_offset * i;
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
}

/***********************************************************************/
/* DESCRIPTION : Execute pack operation                                */
/* ARGUMENT    : [OUT] *dst     : Pointer of destination array         */
/*               [IN] *src      : Pointer of source array              */
/*               [IN] src_dims  : Number of dimensions of source array */
/*               [IN] *src_info : Information of source array          */
/***********************************************************************/
void _XMP_pack_coarray(char* dst, const char* src, const int src_dims, const _XMP_array_section_t* src_info)
{
  size_t src_offset   = _XMP_get_offset(src_info, src_dims);
  int dim_of_allelmts = _XMP_get_dim_of_allelmts(src_dims, src_info);
  
  switch (src_dims){
  case 1:
    _pack_1dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 2:
    _pack_2dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 3:
    _pack_3dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 4:
    _pack_4dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 5:
    _pack_5dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 6:
    _pack_6dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  case 7:
    _pack_7dim_array(src_info, dst, src + src_offset, dim_of_allelmts);
    break;
  default:
    _XMP_fatal("Dimension of coarray is too big");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 7-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_7dim_array(const _XMP_array_section_t* dst_info, const char* src,
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, dst_info[2].distance, _XMP_UNPACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, dst_info, dst_info[3].distance, _XMP_UNPACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, dst_info, dst_info[4].distance, _XMP_UNPACK);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst, src, dst_info, dst_info[5].distance, _XMP_UNPACK);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst, src, dst_info, dst_info[6].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 6-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_6dim_array(const _XMP_array_section_t* dst_info, const char* src,
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, dst_info[2].distance, _XMP_UNPACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, dst_info, dst_info[3].distance, _XMP_UNPACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, dst_info, dst_info[4].distance, _XMP_UNPACK);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst, src, dst_info, dst_info[5].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
    break;
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 5-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_5dim_array(const _XMP_array_section_t* dst_info, const char* src,
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, dst_info[2].distance, _XMP_UNPACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, dst_info, dst_info[3].distance, _XMP_UNPACK);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst, src, dst_info, dst_info[4].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 4-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_4dim_array(const _XMP_array_section_t* dst_info, const char* src, 
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, dst_info[2].distance, _XMP_UNPACK);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst, src, dst_info, dst_info[3].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 3-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_3dim_array(const _XMP_array_section_t* dst_info, const char* src,
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst, src, dst_info, dst_info[2].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 2-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_2dim_array(const _XMP_array_section_t* dst_info, const char* src,
			       char* dst, const int dim_of_allelmts)
{
  switch (dim_of_allelmts){
  case 0:
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    break;
  case 1:
    _XMP_stride_memcpy_1dim(dst, src, dst_info, dst_info[0].distance, _XMP_UNPACK);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst, src, dst_info, dst_info[1].distance, _XMP_UNPACK);
    break;
  default:
    _XMP_fatal("Dimension of coarray error");
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 1-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/*               [IN] dim_of_allelmts : Dimension which has all elements */
/*************************************************************************/
static void _unpack_1dim_array(const _XMP_array_section_t* dst_info, const char* src, char* dst,
			       const int dim_of_allelmts)
{
  if(dim_of_allelmts == 0){
    memcpy(dst, src, dst_info[0].distance * dst_info[0].elmts);
    return;
  }
  
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //    src_offset += element_size;
  //  }
  size_t element_size = dst_info[0].distance;
  int repeat = dst_info[0].length / _XMP_UNROLLING;
  int left   = dst_info[0].length % _XMP_UNROLLING;
  size_t stride_offset = dst_info[0].stride * element_size;
  size_t dst_offset, src_offset = 0;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = i * stride_offset;
      src_offset = i * element_size;
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = i * stride_offset;
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
      dst_offset = stride_offset * (i+2);
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;
    case 2 :
      dst_offset = stride_offset * (i+1);
      memcpy(dst + dst_offset, src + src_offset, element_size);
      src_offset += element_size;
    case 1:
      dst_offset = stride_offset * i;
      memcpy(dst + dst_offset, src + src_offset, element_size);
    }
  }
}

/*************************************************************************/
/* DESCRIPTION : Execute unpack operation for 1-dimensional array        */
/* ARGUMENT    : [IN] *dst_info       : Information of destination array */
/*               [IN] *src            : Pointer of source array          */
/*               [OUT] *dst           : Pointer of destination array     */
/* NOTE       : Only one element is copied to multiple regions           */
/*************************************************************************/
static void _unpack_1dim_array_fixed_src(const _XMP_array_section_t* dst_info,
					 const char* src, char* dst)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //  }
  size_t element_size = dst_info[0].distance;
  int repeat = dst_info[0].length / _XMP_UNROLLING;
  int left   = dst_info[0].length % _XMP_UNROLLING;
  size_t stride_offset = dst_info[0].stride * element_size;
  size_t dst_offset;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = i * stride_offset;
      memcpy(dst + dst_offset, src, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = i * stride_offset;
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
      dst_offset = stride_offset * (i+2);
      memcpy(dst + dst_offset, src, element_size);
    case 2 :
      dst_offset = stride_offset * (i+1);
      memcpy(dst + dst_offset, src, element_size);
    case 1:
      dst_offset = stride_offset * i;
      memcpy(dst + dst_offset, src, element_size);
    }
  }
}

/****************************************************************************/
/* DESCRIPTION : Execute pack operation                                     */
/* ARGUMENT    : [OUT] *dst     : Pointer of destination array              */
/*               [IN] dst_dims  : Number of dimensions of destination array */
/*               [IN] *src      : Pointer of source array                   */
/*               [IN] *dst_info : Information of destination array          */
/*               [IN] flag      : Kind of unpack                            */
/****************************************************************************/
void _XMP_unpack_coarray(char *dst, const int dst_dims, const char* src, 
			 const _XMP_array_section_t* dst_info, const int flag)
{
  size_t dst_offset = _XMP_get_offset(dst_info, dst_dims);
  if(flag == _XMP_UNPACK){
    int dim_of_allelmts = _XMP_get_dim_of_allelmts(dst_dims, dst_info);
    switch (dst_dims){
    case 1:
      _unpack_1dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 2:
      _unpack_2dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 3:
      _unpack_3dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 4:
      _unpack_4dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 5:
      _unpack_5dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 6:
      _unpack_6dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    case 7:
      _unpack_7dim_array(dst_info, src, dst + dst_offset, dim_of_allelmts);
      break;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;
    }
  }
  else if(flag == _XMP_SCALAR_MCOPY){
    switch (dst_dims){
    case 1:
      _unpack_1dim_array_fixed_src(dst_info, src, dst + dst_offset);
      // Perhaps _unpack_1dim_array_fixed_src() is faster than _XMP_stride_memcpy_1dim()
      break;
    case 2:
      _XMP_stride_memcpy_2dim(dst + dst_offset, src, dst_info, dst_info[1].distance, _XMP_SCALAR_MCOPY);
      break;
    case 3:
      _XMP_stride_memcpy_3dim(dst + dst_offset, src, dst_info, dst_info[2].distance, _XMP_SCALAR_MCOPY);
      break;
    case 4:
      _XMP_stride_memcpy_4dim(dst + dst_offset, src, dst_info, dst_info[3].distance, _XMP_SCALAR_MCOPY);
      break;
    case 5:
      _XMP_stride_memcpy_5dim(dst + dst_offset, src, dst_info, dst_info[4].distance, _XMP_SCALAR_MCOPY);
      break;
    case 6:
      _XMP_stride_memcpy_6dim(dst + dst_offset, src, dst_info, dst_info[5].distance, _XMP_SCALAR_MCOPY);
      break;
    case 7:
      _XMP_stride_memcpy_7dim(dst + dst_offset, src, dst_info, dst_info[6].distance, _XMP_SCALAR_MCOPY);
      break;
    default:
      _XMP_fatal("Dimension of coarray is too big.");
      break;
    }
  }
  else{
    _XMP_fatal("Unexpected error !");
  }
}
