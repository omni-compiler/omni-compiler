#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xmp_internal.h"

static int _coarray_dims, _image_dims, *_image_num, _array_dims;
static int _transfer_coarray_elmts, _transfer_array_elmts;
static _XMP_array_section_t *_coarray, *_array;

void _XMP_coarray_rdma_coarray_set_1(const int start1, const int length1, const int stride1)
{
  _transfer_coarray_elmts = length1;
  _coarray_dims           = 1;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);
}

void _XMP_coarray_rdma_coarray_set_2(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2)
{
  _transfer_coarray_elmts = length1 * length2;
  _coarray_dims           = 2;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);
}

void _XMP_coarray_rdma_coarray_set_3(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2,
                                     const int start3, const int length3, const int stride3)
{
  _transfer_coarray_elmts = length1 * length2 * length3;
  _coarray_dims           = 3;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);

  _coarray[2].start       = start3;
  _coarray[2].length      = length3;
  _coarray[2].stride      = ((length3 == 1)? 1 : stride3);
}

void _XMP_coarray_rdma_coarray_set_4(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2,
                                     const int start3, const int length3, const int stride3, 
				     const int start4, const int length4, const int stride4)
{
  _transfer_coarray_elmts = length1 * length2 * length3 * length4;
  _coarray_dims           = 4;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);

  _coarray[2].start       = start3;
  _coarray[2].length      = length3;
  _coarray[2].stride      = ((length3 == 1)? 1 : stride3);

  _coarray[3].start       = start4;
  _coarray[3].length      = length4;
  _coarray[3].stride      = ((length4 == 1)? 1 : stride4);
}

void _XMP_coarray_rdma_coarray_set_5(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2,
                                     const int start3, const int length3, const int stride3, 
				     const int start4, const int length4, const int stride4,
                                     const int start5, const int length5, const int stride5)
{
  _transfer_coarray_elmts = length1 * length2 * length3 * length4 * length5;
  _coarray_dims           = 5;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);

  _coarray[2].start       = start3;
  _coarray[2].length      = length3;
  _coarray[2].stride      = ((length3 == 1)? 1 : stride3);

  _coarray[3].start       = start4;
  _coarray[3].length      = length4;
  _coarray[3].stride      = ((length4 == 1)? 1 : stride4);

  _coarray[4].start       = start5;
  _coarray[4].length      = length5;
  _coarray[4].stride      = ((length5 == 1)? 1 : stride5);
}

void _XMP_coarray_rdma_coarray_set_6(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2,
                                     const int start3, const int length3, const int stride3, 
				     const int start4, const int length4, const int stride4,
                                     const int start5, const int length5, const int stride5, 
				     const int start6, const int length6, const int stride6)
{
  _transfer_coarray_elmts = length1 * length2 * length3 * length4 * length5 * length6;
  _coarray_dims           = 6;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);

  _coarray[2].start       = start3;
  _coarray[2].length      = length3;
  _coarray[2].stride      = ((length3 == 1)? 1 : stride3);

  _coarray[3].start       = start4;
  _coarray[3].length      = length4;
  _coarray[3].stride      = ((length4 == 1)? 1 : stride4);

  _coarray[4].start       = start5;
  _coarray[4].length      = length5;
  _coarray[4].stride      = ((length5 == 1)? 1 : stride5);

  _coarray[5].start       = start6;
  _coarray[5].length      = length6;
  _coarray[5].stride      = ((length6 == 1)? 1 : stride6);
}

void _XMP_coarray_rdma_coarray_set_7(const int start1, const int length1, const int stride1, 
				     const int start2, const int length2, const int stride2,
				     const int start3, const int length3, const int stride3, 
				     const int start4, const int length4, const int stride4,
				     const int start5, const int length5, const int stride5, 
				     const int start6, const int length6, const int stride6,
				     const int start7, const int length7, const int stride7)
{
  _transfer_coarray_elmts = length1 * length2 * length3 * length4 * length5 * length6 * length7;
  _coarray_dims           = 7;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);

  _coarray[1].start       = start2;
  _coarray[1].length      = length2;
  _coarray[1].stride      = ((length2 == 1)? 1 : stride2);

  _coarray[2].start       = start3;
  _coarray[2].length      = length3;
  _coarray[2].stride      = ((length3 == 1)? 1 : stride3);

  _coarray[3].start       = start4;
  _coarray[3].length      = length4;
  _coarray[3].stride      = ((length4 == 1)? 1 : stride4);

  _coarray[4].start       = start5;
  _coarray[4].length      = length5;
  _coarray[4].stride      = ((length5 == 1)? 1 : stride5);

  _coarray[5].start       = start6;
  _coarray[5].length      = length6;
  _coarray[5].stride      = ((length6 == 1)? 1 : stride6);

  _coarray[6].start       = start7;
  _coarray[6].length      = length7;
  _coarray[6].stride      = ((length7 == 1)? 1 : stride7);
}

//void _XMP_coarray_rdma_array_set(const int dim, const int start, const int length, const int stride, const int elmts, const int distance)
//{
//  _array[dim].start    = start;
//  _array[dim].length   = length;
//  _transfer_array_elmts *= length;
//  _array[dim].stride = ((length == 1)? 1 : stride);
//  _array[dim].elmts    = elmts;
//  _array[dim].distance = distance;
//}

void _XMP_coarray_rdma_array_set_1(const int start1, const int length1, const int stride1, const int elmts1, const int distance1)
{
  _transfer_array_elmts = length1;
  _array_dims           = 1;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;
}

void _XMP_coarray_rdma_array_set_2(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
                                   const int start2, const int length2, const int stride2, const int elmts2, const int distance2)
{
  _transfer_array_elmts = length1 * length2;
  _array_dims           = 2;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;
}

void _XMP_coarray_rdma_array_set_3(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
                                   const int start2, const int length2, const int stride2, const int elmts2, const int distance2,
                                   const int start3, const int length3, const int stride3, const int elmts3, const int distance3)
{
  _transfer_array_elmts = length1 * length2 * length3;
  _array_dims           = 3;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;

  _array[2].start       = start3;
  _array[2].length      = length3;
  _array[2].stride      = ((length3 == 1)? 1 : stride3);
  _array[2].elmts       = elmts3;
  _array[2].distance    = distance3;
}

void _XMP_coarray_rdma_array_set_4(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
                                   const int start2, const int length2, const int stride2, const int elmts2, const int distance2,
                                   const int start3, const int length3, const int stride3, const int elmts3, const int distance3,
                                   const int start4, const int length4, const int stride4, const int elmts4, const int distance4)
{
  _transfer_array_elmts = length1 * length2 * length3 * length4;
  _array_dims           = 4;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;

  _array[2].start       = start3;
  _array[2].length      = length3;
  _array[2].stride      = ((length3 == 1)? 1 : stride3);
  _array[2].elmts       = elmts3;
  _array[2].distance    = distance3;

  _array[3].start       = start4;
  _array[3].length      = length4;
  _array[3].stride      = ((length4 == 1)? 1 : stride4);
  _array[3].elmts       = elmts4;
  _array[3].distance    = distance4;
}

void _XMP_coarray_rdma_array_set_5(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
                                   const int start2, const int length2, const int stride2, const int elmts2, const int distance2,
                                   const int start3, const int length3, const int stride3, const int elmts3, const int distance3,
                                   const int start4, const int length4, const int stride4, const int elmts4, const int distance4,
                                   const int start5, const int length5, const int stride5, const int elmts5, const int distance5)
{
  _transfer_array_elmts = length1 * length2 * length3 * length4 * length5;
  _array_dims           = 5;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;

  _array[2].start       = start3;
  _array[2].length      = length3;
  _array[2].stride      = ((length3 == 1)? 1 : stride3);
  _array[2].elmts       = elmts3;
  _array[2].distance    = distance3;

  _array[3].start       = start4;
  _array[3].length      = length4;
  _array[3].stride      = ((length4 == 1)? 1 : stride4);
  _array[3].elmts       = elmts4;
  _array[3].distance    = distance4;

  _array[4].start       = start5;
  _array[4].length      = length5;
  _array[4].stride      = ((length5 == 1)? 1 : stride5);
  _array[4].elmts       = elmts5;
  _array[4].distance    = distance5;
}

void _XMP_coarray_rdma_array_set_6(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
				   const int start2, const int length2, const int stride2, const int elmts2, const int distance2,
                                   const int start3, const int length3, const int stride3, const int elmts3, const int distance3,
                                   const int start4, const int length4, const int stride4, const int elmts4, const int distance4,
                                   const int start5, const int length5, const int stride5, const int elmts5, const int distance5,
                                   const int start6, const int length6, const int stride6, const int elmts6, const int distance6)
{
  _transfer_array_elmts = length1 * length2 * length3 * length4 * length5 * length6;
  _array_dims           = 6;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;

  _array[2].start       = start3;
  _array[2].length      = length3;
  _array[2].stride      = ((length3 == 1)? 1 : stride3);
  _array[2].elmts       = elmts3;
  _array[2].distance    = distance3;

  _array[3].start       = start4;
  _array[3].length      = length4;
  _array[3].stride      = ((length4 == 1)? 1 : stride4);
  _array[3].elmts       = elmts4;
  _array[3].distance    = distance4;

  _array[4].start       = start5;
  _array[4].length      = length5;
  _array[4].stride      = ((length5 == 1)? 1 : stride5);
  _array[4].elmts       = elmts5;
  _array[4].distance    = distance5;

  _array[5].start       = start6;
  _array[5].length      = length6;
  _array[5].stride      = ((length6 == 1)? 1 : stride6);
  _array[5].elmts       = elmts6;
  _array[5].distance    = distance6;
}

void _XMP_coarray_rdma_array_set_7(const int start1, const int length1, const int stride1, const int elmts1, const int distance1,
				   const int start2, const int length2, const int stride2, const int elmts2, const int distance2,
				   const int start3, const int length3, const int stride3, const int elmts3, const int distance3,
				   const int start4, const int length4, const int stride4, const int elmts4, const int distance4,
				   const int start5, const int length5, const int stride5, const int elmts5, const int distance5,
				   const int start6, const int length6, const int stride6, const int elmts6, const int distance6,
				   const int start7, const int length7, const int stride7, const int elmts7, const int distance7)
{
  _transfer_array_elmts = length1 * length2 * length3 * length4 * length5 * length6 * length7;
  _array_dims           = 7;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  _array[0].start       = start1;
  _array[0].length      = length1;
  _array[0].stride      = ((length1 == 1)? 1 : stride1);
  _array[0].elmts       = elmts1;
  _array[0].distance    = distance1;

  _array[1].start       = start2;
  _array[1].length      = length2;
  _array[1].stride      = ((length2 == 1)? 1 : stride2);
  _array[1].elmts       = elmts2;
  _array[1].distance    = distance2;

  _array[2].start       = start3;
  _array[2].length      = length3;
  _array[2].stride      = ((length3 == 1)? 1 : stride3);
  _array[2].elmts       = elmts3;
  _array[2].distance    = distance3;

  _array[3].start       = start4;
  _array[3].length      = length4;
  _array[3].stride      = ((length4 == 1)? 1 : stride4);
  _array[3].elmts       = elmts4;
  _array[3].distance    = distance4;

  _array[4].start       = start5;
  _array[4].length      = length5;
  _array[4].stride      = ((length5 == 1)? 1 : stride5);
  _array[4].elmts       = elmts5;
  _array[4].distance    = distance5;

  _array[5].start       = start6;
  _array[5].length      = length6;
  _array[5].stride      = ((length6 == 1)? 1 : stride6);
  _array[5].elmts       = elmts6;
  _array[5].distance    = distance6;

  _array[6].start       = start7;
  _array[6].length      = length7;
  _array[6].stride      = ((length7 == 1)? 1 : stride7);
  _array[6].elmts       = elmts7;
  _array[6].distance    = distance7;
}

//void _XMP_coarray_rdma_node_set(const int dim, const int image_num)
//{
//  _image_num[dim]  = image_num;
//}

void _XMP_coarray_rdma_node_set_1(const int n1)
{
  _image_dims   = 1;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
}

void _XMP_coarray_rdma_node_set_2(const int n1, const int n2)
{
  _image_dims   = 2;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
}

void _XMP_coarray_rdma_node_set_3(const int n1, const int n2, const int n3)
{
  _image_dims   = 3;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
}

void _XMP_coarray_rdma_node_set_4(const int n1, const int n2, const int n3, const int n4)
{
  _image_dims   = 4;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
}

void _XMP_coarray_rdma_node_set_5(const int n1, const int n2, const int n3, const int n4,
                                  const int n5)
{
  _image_dims   = 5;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
  _image_num[4] = n5;
}

void _XMP_coarray_rdma_node_set_6(const int n1, const int n2, const int n3, const int n4,
                                  const int n5, const int n6)
{
  _image_dims   = 6;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
  _image_num[4] = n5;
  _image_num[5] = n6;
}

void _XMP_coarray_rdma_node_set_7(const int n1, const int n2, const int n3, const int n4, 
				  const int n5, const int n6, const int n7)
{
  _image_dims   = 7;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
  _image_num[4] = n5;
  _image_num[5] = n6;
  _image_num[6] = n7;
}

// If array a is continuous, retrun _XMP_N_INT_TRUE.
// If array a is non-continuous (e.g. stride access), return _XMP_N_INT_FALSE.
static int check_continuous(const _XMP_array_section_t *a, const int dims)
{
  // Only 1 elements is transferred.
  // ex) a[2]
  // ex) b
  if(_transfer_coarray_elmts == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension length is transferred.
  // ex) a[1][2][2:3]
  if(_transfer_coarray_elmts == (a+dims-1)->length)
    if((a+dims-1)->stride == 1)
      return _XMP_N_INT_TRUE;

  // Is the dimension not continuous ?
  for(int i=0;i<dims;i++)
    if((a+i)->stride != 1)
      return _XMP_N_INT_FALSE;

  // (.., i-2, i-1)-th dimension's length is "1" &&
  // i-th dimension's stride is "1" && 
  // (i+1, i+2, ..)-th dimensions are ":".
  // ex) a[1][3][1:2][:]   // i = 2
  // ex) a[2][:][:][:]     // i = 0
  // ex) a[:][:][:][:]     // i = -1
  // Note that: the last dimension must be continuous ((a+dims-1)->stride != 1)
  int i;
  for(i=dims-1;i>=0;i--)
    if((a+i)->length != (a+i)->elmts)
      break;

  if(i == -1 || i == 0){
    return _XMP_N_INT_TRUE;     // Note that (a+i)->stride != 1
  }
  else{  // i != 0
    if(a->length != 1){         // a[:][1:2][:]  i == 1
      return _XMP_N_INT_FALSE;
    }
    else{                       // a[1][2][:][:] ?  i == 2
      for(int j=0;j<i;j++)
	if((a+j)->length != 1)
	  return _XMP_N_INT_FALSE;
    
      return _XMP_N_INT_TRUE;
    }
  }
}

void _XMP_coarray_rdma_do(const int rdma_code, void *remote_coarray, void *local_array, void *local_coarray)
/* If a local array is a coarray, local_coarray != NULL. */
{
  if(_transfer_coarray_elmts == 0) return;

  if(_transfer_coarray_elmts != _transfer_array_elmts)
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;

  int target_image = 0;
  for(int i=0;i<_image_dims;i++)
    target_image += ((_XMP_coarray_t*)remote_coarray)->distance_of_image_elmts[i] * (_image_num[i] - 1);

  for(int i=0;i<_coarray_dims;i++){
    _coarray[i].elmts    = ((_XMP_coarray_t*)remote_coarray)->coarray_elmts[i];
    _coarray[i].distance = ((_XMP_coarray_t*)remote_coarray)->distance_of_coarray_elmts[i];
  }

  int remote_coarray_is_continuous, local_array_is_continuous;
  remote_coarray_is_continuous = check_continuous(_coarray, _coarray_dims);
  local_array_is_continuous    = check_continuous(_array, _array_dims); 

#ifdef _XMP_COARRAY_FJRDMA
  if(remote_coarray_is_continuous == _XMP_N_INT_FALSE || local_array_is_continuous == _XMP_N_INT_FALSE)
    _XMP_fatal("Sorry! Not continuous array is not supported.");
#endif

  if(_XMP_N_COARRAY_PUT == rdma_code){
    if(target_image == _XMP_world_rank && remote_coarray_is_continuous && local_array_is_continuous){ // Fix me:
      if(local_array == NULL)
	memcpy(((_XMP_coarray_t*)remote_coarray)->real_addr, ((_XMP_coarray_t*)local_coarray)->real_addr, _transfer_coarray_elmts*((_XMP_coarray_t*)remote_coarray)->elmt_size);
      else{
	uint64_t dst_point = (uint64_t)get_offset(_coarray, _coarray_dims);
	uint64_t src_point = (uint64_t)get_offset(_array, _array_dims);
	memcpy(((_XMP_coarray_t*)remote_coarray)->real_addr + dst_point, 
	       (char *)local_array + src_point, _transfer_coarray_elmts*((_XMP_coarray_t*)remote_coarray)->elmt_size);
      }
    }
    else{
#ifdef _XMP_COARRAY_GASNET
      _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_image,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts);
#elif _XMP_COARRAY_FJRDMA
      _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_image, 
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts);
#endif
    }
  }
  else if(_XMP_N_COARRAY_GET == rdma_code){
    if(target_image == _XMP_world_rank && remote_coarray_is_continuous && local_array_is_continuous){ // Fix me:
      if(local_array == NULL)
	memcpy(((_XMP_coarray_t*)local_coarray)->real_addr, ((_XMP_coarray_t*)remote_coarray)->real_addr, _transfer_coarray_elmts*((_XMP_coarray_t*)remote_coarray)->elmt_size);
      else{
	uint64_t src_point = (uint64_t)get_offset(_coarray, _coarray_dims);
        uint64_t dst_point = (uint64_t)get_offset(_array, _array_dims);
	memcpy((char *)local_array + dst_point, 
	       ((_XMP_coarray_t*)remote_coarray)->real_addr + src_point, _transfer_coarray_elmts*((_XMP_coarray_t*)remote_coarray)->elmt_size);
      }
    }
    else{
#ifdef _XMP_COARRAY_GASNET
      _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_image,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts);
#elif _XMP_COARRAY_FJRDMA
      _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_image, 
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts);
#endif
    }
  }
  else{
    _XMP_fatal("Unexpected Operation !!");
  }

  free(_coarray);
  free(_array);
  free(_image_num);
}

void _XMP_coarray_rdma_do_f(const int *rdma_code, void *remote_coarray, void *local_array, void *local_coarray)
/* If a local array is a coarray, local_coarray != NULL. */
{
  _XMP_coarray_rdma_do(*rdma_code, remote_coarray, local_array, local_coarray);
}

void _XMP_coarray_sync_all()
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

void _XMP_coarray_sync_memory()
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

void xmp_sync_memory(const int* status)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

void xmp_sync_all(const int* status)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

void xmp_sync_image(int image, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images()");
}

void xmp_sync_image_f(int *image, int* status)
{
  xmp_sync_image(*image, status);
}

void xmp_sync_images(int num, int* image_set, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_images()");
}

void xmp_sync_images_f(int *num, int* image_set, int* status)
{
  xmp_sync_images(*num, image_set, status);
}

void xmp_sync_images_all(int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_all()");
}

size_t get_offset(const _XMP_array_section_t *array, const int dims)
{
  size_t offset = 0;
  for(int i=0;i<dims;i++)
    offset += (array+i)->start * (array+i)->distance;

  return offset;
}

void _XMP_coarray_shortcut_put(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src, 
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;

  if(rank == _XMP_world_rank){
    memcpy(dst->real_addr + dst_offset, src->real_addr + src_offset, transfer_size);
  }
  else{
#ifdef _XMP_COARRAY_GASNET
    gasnet_put_nbi_bulk(rank, dst->addr[rank]+dst_offset,
			src->addr[_XMP_world_rank]+src_offset, transfer_size);
#elif _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_shortcut_put(rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst, src, transfer_size);
#endif
  }
}

void _XMP_coarray_shortcut_get(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src,
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;
  
  if(rank == _XMP_world_rank){
    memcpy(dst->real_addr + dst_offset, src->real_addr + src_offset, transfer_size);
  }
  else{
#ifdef _XMP_COARRAY_GASNET
    gasnet_get_bulk(dst->addr[_XMP_world_rank]+dst_offset, rank, src->addr[rank]+src_offset, transfer_size);
#elif _XMP_COARRAY_FJRDMA
    _XMP_fjrdma_shortcut_get(rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst, src, transfer_size);
#endif
  }
}

void _XMP_coarray_shortcut_put_f(const int *target, const void *dst, const void *src, const size_t *dst_offset, 
				 const size_t *src_offset, const size_t *transfer_size)
{
  _XMP_coarray_shortcut_put(*target, dst, src, *dst_offset, *src_offset, *transfer_size);
}

void _XMP_coarray_shortcut_get_f(const int *target, const void *dst, const void *src, const size_t *dst_offset, 
				 const size_t *src_offset, const size_t *transfer_size)
{
  _XMP_coarray_shortcut_get(*target, dst, src, *dst_offset, *src_offset, *transfer_size);
}
