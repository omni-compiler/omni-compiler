#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <complex.h>

int _xmp_omp_num_procs = 1;

void _XMPF_pack_vector(char * restrict dst, char * restrict src,
		       int count, int blocklength, int stride){
  long i;
  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for
    for (i = 0; i < count; i++){
      memcpy(dst + i * blocklength, src + i * stride, blocklength);
    }
  }
  else {
    for (i = 0; i < count; i++){
      memcpy(dst + i * blocklength, src + i * stride, blocklength);
    }
  }

}


void _XMPF_unpack_vector(char * restrict dst, char * restrict src,
			 int count, int blocklength, int stride){
  long i;
  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for
    for (i = 0; i < count; i++){
      memcpy(dst + i * stride, src + i * blocklength, blocklength);
    }
  }
  else {
    for (i = 0; i < count; i++){
      memcpy(dst + i * stride, src + i * blocklength, blocklength);
    }
  }

}

void _XMPF_unpack_transpose_vector(char * restrict dst, char * restrict src,
                                   int icount, int jcount, int type_size,
                                   int src_stride, int dst_stride){

  long i,j;
  if (_xmp_omp_num_procs > 1 && icount > 8 * _xmp_omp_num_procs){
    if (type_size == 16){
      long ii,jj,nblk=16;
      double _Complex *dst0 = (double _Complex *)dst;
      double _Complex *src0 = (double _Complex *)src;
      for (jj = 0; jj < jcount; jj+=nblk){
#pragma omp parallel for
        for (ii = 0; ii < icount; ii+=nblk){
          long jmin=((jj+nblk) < jcount)? (jj+nblk):jcount;
          for (j = jj; j < jmin; j++){
            long imin=((ii+nblk) < icount)? (ii+nblk):icount;
            for (i = ii; i < imin; i++){
              dst0[j * dst_stride + i]=src0[i * src_stride + j];
            }
          }
        }
      }
    }
    else {
      for (j = 0; j < jcount; j++){
        for (i = 0; i < icount; i++){
          memcpy(dst + j * dst_stride * type_size + i * type_size,
                 src + i * src_stride * type_size + j * type_size, type_size);
        }
      }
    }
  }
  else {
    for (j = 0; j < jcount; j++){
      for (i = 0; i < icount; i++){
        memcpy(dst + j * dst_stride * type_size + i * type_size,
               src + i * src_stride * type_size + j * type_size, type_size);
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
