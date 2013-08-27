#include <stdio.h>
#include <string.h>
#include <omp.h>

int _xmp_omp_num_procs = 1;

void _XMPF_pack_vector(char * restrict dst, char * restrict src,
		       int count, int blocklength, int stride){

  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for
    for (int i = 0; i < count; i++){
      memcpy(dst + i * blocklength, src + i * stride, blocklength);
    }
  }
  else {
    for (int i = 0; i < count; i++){
      memcpy(dst + i * blocklength, src + i * stride, blocklength);
    }
  }

}


void _XMPF_unpack_vector(char * restrict dst, char * restrict src,
			 int count, int blocklength, int stride){

  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for
    for (int i = 0; i < count; i++){
      memcpy(dst + i * stride, src + i * blocklength, blocklength);
    }
  }
  else {
    for (int i = 0; i < count; i++){
      memcpy(dst + i * stride, src + i * blocklength, blocklength);
    }
  }

}

void _XMPF_unpack_transpose_vector(char * restrict dst, char * restrict src,
                                   int icount, int jcount, int wordlength, 
                                   int src_stride, int dst_stride){

  if (_xmp_omp_num_procs > 1 && icount > 8 * _xmp_omp_num_procs){
#pragma omp parallel for
    for (int i = 0; i < icount; i++){
      for (int j = 0; j < jcount; j++){
        memcpy(dst + j * dst_stride + i * wordlength, 
               src + i * src_stride + j * wordlength, wordlength);
      }
    }
  }
  else {
    for (int i = 0; i < icount; i++){
      for (int j = 0; j < jcount; i++){
        memcpy(dst + j * dst_stride + i * wordlength, 
               src + i * src_stride + j * wordlength, wordlength);
      }
    }
  }

}

#include "xmpf_internal.h"

int _xmp_reflect_pack_flag = 0;

void _XMPF_check_reflect_type(void)
{
  char *reflect_type = getenv("XMP_REFLECT_TYPE");
  _xmp_omp_num_procs = omp_get_num_procs();

  if (reflect_type){

    if (strcmp(reflect_type, "REFLECT_NOPACK") == 0){
      _xmp_reflect_pack_flag = 0;
      //xmpf_dbg_printf("REFLECT_NOPACK\n");
      return;
    }
    else if (strcmp(reflect_type, "REFLECT_PACK") == 0){
      _xmp_reflect_pack_flag = 1;
      //xmpf_dbg_printf("REFLECT_PACK\n");
      return;
    }

  }

  // not specified or a wrong value
  if (_xmp_omp_num_procs > 1){
    _xmp_reflect_pack_flag = 1;
    //xmpf_dbg_printf("not specified and REFLECT_PACK\n");
  }
  else {
    _xmp_reflect_pack_flag = 0;
    //xmpf_dbg_printf("not specified and REFLECT_NOPACK\n");
  }

}
