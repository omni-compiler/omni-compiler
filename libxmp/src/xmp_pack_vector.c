#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <complex.h>

int _xmp_omp_num_procs = 1;

void _XMP_pack_vector(char * restrict dst, char * restrict src,
		      int count, int blocklength, long stride){
  long i;
  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for private(i)
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

void _XMP_pack_vector2(char * restrict dst, char * restrict src,
                       int count, int blocklength,
                       int nnodes, int type_size, int src_block_dim){
  long j,k;
  if (src_block_dim == 1){
#pragma omp parallel for private(j,k)
    for (j = 0; j < count; j++){
      for (k = 0; k < nnodes; k++){
        memcpy(dst + ((k * count +j ) * blocklength ) * type_size,
               src + ((k + j * nnodes) * blocklength ) * type_size,
               blocklength * type_size);
      }
    }
  }
}

void _XMP_unpack_vector(char * restrict dst, char * restrict src,
			int count, int blocklength, long stride){
  long i;
  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){
#pragma omp parallel for private(i)
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
                                   int dst_stride, int src_stride,
                                   int type_size, int dst_block_dim){
  long i,j;
  if (dst_block_dim == 1){
    if (type_size == 16){
      long ii,jj,imin,jmin,nblk=16;
      double _Complex *dst0 = (double _Complex *)dst;
      double _Complex *src0 = (double _Complex *)src;
      for (jj = 0; jj < src_stride; jj+=nblk){
        jmin=((jj+nblk) < src_stride)? (jj+nblk):src_stride;
#pragma omp parallel for private(i,j,ii,imin)
        for (ii = 0; ii < dst_stride; ii+=nblk){
          imin=((ii+nblk) < dst_stride)? (ii+nblk):dst_stride;
          for (j = jj; j < jmin; j++){
            for (i = ii; i < imin; i++){
              dst0[j * dst_stride + i] = src0[i * src_stride + j];
            }
          }
        }
      }
    }
    else {
      for (j = 0; j < src_stride; j++){
        for (i = 0; i < dst_stride; i++){
          memcpy(dst + (j * dst_stride + i) * type_size,
                 src + (i * src_stride + j) * type_size, type_size);
        }
      }
    }
  }
}

#include "xmp_internal.h"

#define _XMP_SUM_VECTOR(_type) \
  for (long i = 0; i < count; i++){ \
    for (long j = 0; j < blocklength; j++){ \
      ((_type *)dst)[i * stride + j] += ((_type *)src)[i * blocklength + j]; \
    } \
  }

void _XMP_sum_vector(int type, char * restrict dst, char * restrict src,
		     int count, int blocklength, long stride){

  if (_xmp_omp_num_procs > 1 && count > 8 * _xmp_omp_num_procs){

    switch (type){

    case _XMP_N_TYPE_SHORT:
#pragma omp parallel for
      _XMP_SUM_VECTOR(short);
      break;

    case _XMP_N_TYPE_UNSIGNED_SHORT:
#pragma omp parallel for
      _XMP_SUM_VECTOR(unsigned short);
      break;

    case _XMP_N_TYPE_INT:
#pragma omp parallel for
      _XMP_SUM_VECTOR(int);
      break;

    case _XMP_N_TYPE_UNSIGNED_INT:
#pragma omp parallel for
      _XMP_SUM_VECTOR(unsigned int);
      break;

    case _XMP_N_TYPE_LONG:
#pragma omp parallel for
      _XMP_SUM_VECTOR(long);
      break;

    case _XMP_N_TYPE_UNSIGNED_LONG:
#pragma omp parallel for
      _XMP_SUM_VECTOR(unsigned long);
      break;

    case _XMP_N_TYPE_LONGLONG:
#pragma omp parallel for
      _XMP_SUM_VECTOR(long long);
      break;

    case _XMP_N_TYPE_UNSIGNED_LONGLONG:
#pragma omp parallel for
      _XMP_SUM_VECTOR(unsigned long long);
      break;

    case _XMP_N_TYPE_FLOAT:
#pragma omp parallel for
      _XMP_SUM_VECTOR(float);
      break;

    case _XMP_N_TYPE_DOUBLE:
#pragma omp parallel for
      _XMP_SUM_VECTOR(double);
      break;
      
    case _XMP_N_TYPE_LONG_DOUBLE:
#pragma omp parallel for
      _XMP_SUM_VECTOR(long double);
      break;
	
#ifdef __STD_IEC_559_COMPLEX__

    case _XMP_N_TYPE_FLOAT_IMAGINARY:
#pragma omp parallel for
      _XMP_SUM_VECTOR(float imaginary);
      break;

    case _XMP_N_TYPE_FLOAT_COMPLEX:
#pragma omp parallel for
      _XMP_SUM_VECTOR(float complex);
      break;

    case _XMP_N_TYPE_DOUBLE_IMAGINARY:
#pragma omp parallel for
      _XMP_SUM_VECTOR(double imaginary);
      break;

    case _XMP_N_TYPE_DOUBLE_COMPLEX:
#pragma omp parallel for
      _XMP_SUM_VECTOR(double complex);
      break;

    case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
#pragma omp parallel for
      _XMP_SUM_VECTOR(long double imaginary);
      break;

    case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
#pragma omp parallel for
      _XMP_SUM_VECTOR(long double complex);
      break;

#endif

    case _XMP_N_TYPE_BOOL:
    case _XMP_N_TYPE_CHAR:
    case _XMP_N_TYPE_UNSIGNED_CHAR:
    case _XMP_N_TYPE_NONBASIC:
    default:
      _XMP_fatal("_XMP_sum_vector: array arguments must be of a numerical type");
      break;
    }
    
  }
  else {

    switch (type){

    case _XMP_N_TYPE_SHORT:
      _XMP_SUM_VECTOR(short);
      break;

    case _XMP_N_TYPE_UNSIGNED_SHORT:
      _XMP_SUM_VECTOR(unsigned short);
      break;

    case _XMP_N_TYPE_INT:
      _XMP_SUM_VECTOR(int);
      break;

    case _XMP_N_TYPE_UNSIGNED_INT:
      _XMP_SUM_VECTOR(unsigned int);
      break;

    case _XMP_N_TYPE_LONG:
      _XMP_SUM_VECTOR(long);
      break;

    case _XMP_N_TYPE_UNSIGNED_LONG:
      _XMP_SUM_VECTOR(unsigned long);
      break;

    case _XMP_N_TYPE_LONGLONG:
      _XMP_SUM_VECTOR(long long);
      break;

    case _XMP_N_TYPE_UNSIGNED_LONGLONG:
      _XMP_SUM_VECTOR(unsigned long long);
      break;

    case _XMP_N_TYPE_FLOAT:
      _XMP_SUM_VECTOR(float);
      break;

    case _XMP_N_TYPE_DOUBLE:
      _XMP_SUM_VECTOR(double);
      break;
      
    case _XMP_N_TYPE_LONG_DOUBLE:
      _XMP_SUM_VECTOR(long double);
      break;
	
#ifdef __STD_IEC_559_COMPLEX__

    case _XMP_N_TYPE_FLOAT_IMAGINARY:
      _XMP_SUM_VECTOR(float imaginary);
      break;

    case _XMP_N_TYPE_FLOAT_COMPLEX:
      _XMP_SUM_VECTOR(float complex);
      break;

    case _XMP_N_TYPE_DOUBLE_IMAGINARY:
      _XMP_SUM_VECTOR(double imaginary);
      break;

    case _XMP_N_TYPE_DOUBLE_COMPLEX:
      _XMP_SUM_VECTOR(double complex);
      break;

    case _XMP_N_TYPE_LONG_DOUBLE_IMAGINARY:
      _XMP_SUM_VECTOR(long double imaginary);
      break;

    case _XMP_N_TYPE_LONG_DOUBLE_COMPLEX:
      _XMP_SUM_VECTOR(long double complex);
      break;

#endif

    case _XMP_N_TYPE_BOOL:
    case _XMP_N_TYPE_CHAR:
    case _XMP_N_TYPE_UNSIGNED_CHAR:
    case _XMP_N_TYPE_NONBASIC:
    default:
      _XMP_fatal("_XMP_sum_vector: array arguments must be of a numerical type");
      break;
    }

  }

}

#include "xmp_internal.h"

int _xmp_reflect_pack_flag = 0;

void _XMP_check_reflect_type(void)
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
