#include "xmp_internal.h"
#define UNROLLING (4)

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
// ushiro no jigen kara kazoete "depth" banme made rennzokuka ?
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
static int _get_depth(const int dims, const _XMP_array_section_t* array_info)  // 7 >= dims >= 2
{
  if(dims == 2){
    if(array_info[1].stride == 1)
      return 1;
    else
      return 0;
  }
  else if(dims == 3){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2)){
      return 2;
    }
    else if(array_info[2].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 4){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3)){
      return 2;
    }
    else if(array_info[3].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 5){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
      return 2;
    }
    else if(array_info[4].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 6){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
       _is_all_elmt(array_info, 5)){
      return 5;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) &&
            _is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
            _is_all_elmt(array_info, 5)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5)){
      return 2;
    }
    else if(array_info[5].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 7){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
       _is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
      return 6;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) &&
            _is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5) &&
            _is_all_elmt(array_info, 6)){
      return 5;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
            _is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5) &&
            _is_all_elmt(array_info, 6)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
      return 2;
    }
    else if(array_info[6].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else{
    _XMP_fatal("Dimensions of Coarray is too big.");
    return -1;
  }
}

static void _pack_7_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[6].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[7];
  size_t stride_offset[7], length;

  for(int i=0;i<7;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 6){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 5){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 4){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 3){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
            archive_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[5].distance * src[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
              archive_offset += length;
            }
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[6].distance * src[6].length;
    for(int i=0;i<6;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<src[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		archive_offset += length;
	      }
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<7;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<src[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		for(int q=0;q<src[6].length;q++){
		  tmp[6] = stride_offset[6] * q;
		  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]);
		  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		  archive_offset += length;
		}
              }
            }
          }
        }
      }
    }
  }
}

static void _pack_6_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[5].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[6];
  size_t stride_offset[6], length;

  for(int i=0;i<6;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 5){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 4){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 3){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
            archive_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[5].distance * src[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	      archive_offset += length;
	    }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<6;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<src[5].length;p++){
		tmp[5] = stride_offset[5] * p;
		src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		archive_offset += length;
	      }
            }
          }
        }
      }
    }
  }
}

static void _pack_5_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[4].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[5];
  size_t stride_offset[5], length;

  for(int i=0;i<5;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 4){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 3){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	    archive_offset += length;
	  }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<src[4].length;n++){
	      tmp[4] = stride_offset[4] * n;
	      src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	      archive_offset += length;
	    }
          }
        }
      }
    }
  }
}

static void _pack_4_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[3].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[4];
  size_t stride_offset[4], length;

  for(int i=0;i<4;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 3){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 2){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	src_offset = start_offset + (tmp[0] + tmp[1]);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	  archive_offset += length;
	}
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  for(int m=0;m<src[3].length;m++){
	    tmp[3] = stride_offset[3] * m;
	    src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	    archive_offset += length;
	  }
	}
      }
    }
  }
}

static void _pack_3_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = src[2].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[3];
  size_t stride_offset[3], length;

  for(int i=0;i<3;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 2){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 1){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	src_offset = start_offset + (tmp[0] + tmp[1]);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	archive_offset += length;
      }
    }
  }
  else{ // continuous_dim == 0
    length = element_size;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	  archive_offset += length;
	}
      }
    }
  }
}

static void _pack_2_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr, 
			      const int continuous_dim) // continuous_dim is 0 or 1
{
  size_t element_size = src[1].distance;
  size_t start_offset = 0;
  size_t archive_offset = 0, src_offset;

  for(int i=0;i<2;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 1){
    int length = element_size * src[1].length;
    size_t stride_offset = (src[0].stride * src[1].elmts) * element_size;
    for(int i=0;i<src[0].length;i++){
      src_offset = start_offset + stride_offset * i;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else{ // continuous_dim == 0
    size_t stride_offset[2];
    stride_offset[0] = src[0].stride * src[1].elmts * element_size;
    stride_offset[1] = src[1].stride * element_size;
    for(int i=0;i<src[0].length;i++){
      size_t tmp = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	src_offset = start_offset + (tmp + stride_offset[1] * j);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
	archive_offset += element_size;
      }
    }
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
  int repeat = src[0].length / UNROLLING;
  int left   = src[0].length % UNROLLING;
  size_t start_offset  = src[0].start  * element_size;
  size_t stride_offset = src[0].stride * element_size;
  size_t archive_offset = 0, src_offset;
  int i = 0;

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

      i += UNROLLING;
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
  int continuous_dim = _get_depth(src_dims, src);

  if(src_dims == 2){
    _pack_2_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 3){
    _pack_3_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 4){
    _pack_4_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 5){
    _pack_5_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 6){
    _pack_6_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 7){
    _pack_7_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else{
    _XMP_fatal("Dimension of coarray is too big");
    return;
  }
}

static void _unpack_7_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[6].distance;
  size_t start_offset = 0, dst_offset;
  int tmp[7];
  size_t stride_offset[7], length;

  for(int i=0;i<7;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 6){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else if(continuous_dim == 5){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr, length);
      }
    }
  }
  else if(continuous_dim == 4){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr, length);
        }
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr, length);
          }
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(dst_ptr + dst_offset, src_ptr, length);
            }
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[6].distance * dst[6].length;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
                memcpy(dst_ptr + dst_offset, src_ptr, length);
              }
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<7;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                for(int q=0;q<dst[6].length;q++){
                  tmp[6] = stride_offset[6] * q;
                  dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]);
                  memcpy(dst_ptr + dst_offset, src_ptr, length);
                }
              }
            }
          }
        }
      }
    }
  }
}


static void _unpack_7_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[6].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[7];
  size_t stride_offset[7], length;

  for(int i=0;i<7;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 6){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 5){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 4){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
              src_offset += length;
            }
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[6].distance * dst[6].length;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		src_offset += length;
	      }
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<7;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		for(int q=0;q<dst[6].length;q++){
		  tmp[6] = stride_offset[6] * q;
		  dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]);
		  memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		  src_offset += length;
		}
              }
            }
          }
        }
      }
    }
  }
}

static void _unpack_6_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[5].distance;
  size_t start_offset = 0, dst_offset;
  int tmp[6];
  size_t stride_offset[6], length;

  for(int i=0;i<6;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 5){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else if(continuous_dim == 4){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr, length);
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr, length);
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr, length);
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(dst_ptr + dst_offset, src_ptr, length);
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
                dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
                memcpy(dst_ptr + dst_offset, src_ptr, length);
              }
            }
          }
        }
      }
    }
  }
}

static void _unpack_6_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[5].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[6];
  size_t stride_offset[6], length;

  for(int i=0;i<6;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 5){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 4){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	      src_offset += length;
	    }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<dst[5].length;p++){
		tmp[5] = stride_offset[5] * p;
		dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		src_offset += length;
	      }
            }
          }
        }
      }
    }
  }
}

static void _unpack_5_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[4].distance;
  size_t start_offset = 0, dst_offset;
  int tmp[5];
  size_t stride_offset[5], length;

  for(int i=0;i<5;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 4){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else if(continuous_dim == 3){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr, length);
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr, length);
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr, length);
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(dst_ptr + dst_offset, src_ptr, length);
            }
          }
        }
      }
    }
  }
}

static void _unpack_5_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[4].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[5];
  size_t stride_offset[5], length;

  for(int i=0;i<5;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 4){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 3){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	    src_offset += length;
	  }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<dst[4].length;n++){
	      tmp[4] = stride_offset[4] * n;
	      dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	      src_offset += length;
	    }
          }
        }
      }
    }
  }
}

static void _unpack_4_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[3].distance;
  size_t start_offset = 0, dst_offset;
  int tmp[4];
  size_t stride_offset[4], length;

  for(int i=0;i<4;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 3){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else if(continuous_dim == 2){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr, length);
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr, length);
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr, length);
          }
        }
      }
    }
  }
}

static void _unpack_4_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, 
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[3].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[4];
  size_t stride_offset[4], length;

  for(int i=0;i<4;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 3){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 2){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
}

static void _unpack_3_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[2].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[3];
  size_t stride_offset[3], length;

  for(int i=0;i<3;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 2){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 1){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	src_offset += length;
      }
    }
  }
  else{ // continuous_dim == 0
    length = element_size;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
}

static void _unpack_3_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[2].distance;
  size_t start_offset = 0, dst_offset;
  int tmp[3];
  size_t stride_offset[3], length;

  for(int i=0;i<3;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 2){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else if(continuous_dim == 1){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr, length);
      }
    }
  }
  else{ // continuous_dim == 0
    length = element_size;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr, length);
        }
      }
    }
  }
}

static void _unpack_2_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  // continuous_dim is 0 or 1
  size_t element_size = dst[1].distance;
  size_t start_offset  = (dst[0].start * dst[1].elmts + dst[1].start) * element_size;
  size_t dst_offset, src_offset = 0;
  int i;

  if(continuous_dim == 1){
    int length = element_size * dst[1].length;
    size_t stride_offset = (dst[0].stride * dst[1].elmts) * element_size;
    for(i=0;i<dst[0].length;i++){
      dst_offset = start_offset + stride_offset * i;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else{ // continuous_dim == 0
    int j;
    size_t stride_offset[2];
    stride_offset[0] = dst[0].stride * dst[1].elmts * element_size;
    stride_offset[1] = dst[1].stride * element_size;
    for(i=0;i<dst[0].length;i++){
      size_t tmp = stride_offset[0] * i;
      for(j=0;j<dst[1].length;j++){
        dst_offset = start_offset + (tmp + stride_offset[1] * j);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
        src_offset += element_size;
      }
    }
  }
}

static void _unpack_2_dim_array_fixed_src(const _XMP_array_section_t* dst, const char* src_ptr,
					  char* dst_ptr, const int continuous_dim)
{
  // continuous_dim is 0 or 1
  size_t element_size = dst[1].distance;
  size_t start_offset  = (dst[0].start * dst[1].elmts + dst[1].start) * element_size;
  size_t dst_offset;
  int i;

  if(continuous_dim == 1){
    int length = element_size * dst[1].length;
    size_t stride_offset = (dst[0].stride * dst[1].elmts) * element_size;
    for(i=0;i<dst[0].length;i++){
      dst_offset = start_offset + stride_offset * i;
      memcpy(dst_ptr + dst_offset, src_ptr, length);
    }
  }
  else{ // continuous_dim == 0
    int j;
    size_t stride_offset[2];
    stride_offset[0] = dst[0].stride * dst[1].elmts * element_size;
    stride_offset[1] = dst[1].stride * element_size;
    for(i=0;i<dst[0].length;i++){
      size_t tmp = stride_offset[0] * i;
      for(j=0;j<dst[1].length;j++){
        dst_offset = start_offset + (tmp + stride_offset[1] * j);
        memcpy(dst_ptr + dst_offset, src_ptr, element_size);
      }
    }
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
  int repeat = dst[0].length / UNROLLING;
  int left   = dst[0].length % UNROLLING;
  size_t start_offset  = dst[0].start  * element_size;
  size_t stride_offset = dst[0].stride * element_size;
  size_t dst_offset, src_offset = 0;
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

      i += UNROLLING;
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
  int repeat = dst[0].length / UNROLLING;
  int left   = dst[0].length % UNROLLING;
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
 
      i += UNROLLING;
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
  // flag == 1; src_offset is fixed for put_scalar_bcast.

  if(dst_dims == 1){
    if(flag == 0)
      _unpack_1_dim_array(dst, src_ptr, dst_ptr);
    else
      _unpack_1_dim_array_fixed_src(dst, src_ptr, dst_ptr);
    
    return;
  }

  int continuous_dim = _get_depth(dst_dims, dst);

  if(dst_dims == 2){
    if(flag == 0)
      _unpack_2_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_2_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else if(dst_dims == 3){
    if(flag == 0)
      _unpack_3_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_3_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else if(dst_dims == 4){
    if(flag == 0)
      _unpack_4_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_4_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else if(dst_dims == 5){
    if(flag == 0)
      _unpack_5_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_5_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else if(dst_dims == 6){
    if(flag == 0)
      _unpack_6_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_6_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else if(dst_dims == 7){
    if(flag == 0)
      _unpack_7_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    else
      _unpack_7_dim_array_fixed_src(dst, src_ptr, dst_ptr, continuous_dim);
  }
  else{
    _XMP_fatal("Dimension of coarray is too big.");
  }
}
