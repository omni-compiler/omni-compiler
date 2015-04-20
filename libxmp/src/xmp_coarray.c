#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "xmp_internal.h"
#include "xmp_constant.h"

static size_t _elmt_size;
static int _coarray_dims, _image_dims, *_image_elmts;
static int *_coarray_elmts, _total_coarray_elmts;
static int _coarray_dims, _image_dims, *_image_num, _array_dims;
static int _transfer_coarray_elmts, _transfer_array_elmts;
static _XMP_array_section_t *_coarray, *_array;
struct _coarray_queue_t{
  unsigned int     max_size;   /**< Max size of queue */
  unsigned int          num;   /**< How many coarrays are in this queue */
  _XMP_coarray_t **coarrays;   /**< pointer of coarrays */
};
static struct _coarray_queue_t _coarray_queue;
static void _push_coarray_queue(_XMP_coarray_t *c);

/**
   Set coarray information when allocating coarray
 */
void _XMP_coarray_malloc_info_1(const int n1, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 1;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _total_coarray_elmts = n1;
}

void _XMP_coarray_malloc_info_2(const int n1, const int n2, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 2;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _total_coarray_elmts = n1*n2;
}

void _XMP_coarray_malloc_info_3(const int n1, const int n2, const int n3, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 3;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _total_coarray_elmts = n1*n2*n3;
}

void _XMP_coarray_malloc_info_4(const int n1, const int n2, const int n3, const int n4,
				const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 4;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _total_coarray_elmts = n1*n2*n3*n4;
}

void _XMP_coarray_malloc_info_5(const int n1, const int n2, const int n3, const int n4,
				const int n5, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 5;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _coarray_elmts[4]    = n5;
  _total_coarray_elmts = n1*n2*n3*n4*n5;
}

void _XMP_coarray_malloc_info_6(const int n1, const int n2, const int n3, const int n4,
				const int n5, const int n6, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 6;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _coarray_elmts[4]    = n5;
  _coarray_elmts[5]    = n6;
  _total_coarray_elmts = n1*n2*n3*n4*n5*n6;
}

void _XMP_coarray_malloc_info_7(const int n1, const int n2, const int n3, const int n4, 
				const int n5, const int n6, const int n7, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 7;
  _coarray_elmts       = malloc(sizeof(int) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _coarray_elmts[4]    = n5;
  _coarray_elmts[5]    = n6;
  _coarray_elmts[6]    = n7;
  _total_coarray_elmts = n1*n2*n3*n4*n5*n6*n7;
}

/**
    Set image information when allocating coarray
 */
void _XMP_coarray_malloc_image_info_1()
{
  _image_dims     = 1;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = 1;
}

void _XMP_coarray_malloc_image_info_2(const int i1)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");
    
  _image_dims     = 2;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = total_node_size / total_image_size;
}

void _XMP_coarray_malloc_image_info_3(const int i1, const int i2)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1*i2;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_dims     = 3;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = total_node_size / total_image_size;
}

void _XMP_coarray_malloc_image_info_4(const int i1, const int i2, const int i3)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1*i2*i3;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_dims     = 4;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = total_node_size / total_image_size;
}

void _XMP_coarray_malloc_image_info_5(const int i1, const int i2, const int i3, const int i4)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1*i2*i3*i4;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_dims     = 5;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = total_node_size / total_image_size;
}

void _XMP_coarray_malloc_image_info_6(const int i1, const int i2, const int i3, const int i4,
                                      const int i5)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1*i2*i3*i4*i5;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_dims    = 6;
  _image_elmts   = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = i5;
  _image_elmts[5] = total_node_size / total_image_size;
}

void _XMP_coarray_malloc_image_info_7(const int i1, const int i2, const int i3, const int i4,
                                      const int i5, const int i6)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = i1*i2*i3*i4*i5*i6;

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_dims    = 7;
  _image_elmts   = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = i5;
  _image_elmts[5] = i6;
  _image_elmts[6] = total_node_size / total_image_size;
}

/**
   Create coarray object and allocate coarray.
 */
void _XMP_coarray_malloc_do(void **coarray, void *addr)
{
  int *distance_of_coarray_elmts = _XMP_alloc(sizeof(int) * _coarray_dims);

  for(int i=0;i<_coarray_dims-1;i++){
    int distance = 1;
    for(int j=i+1;j<_coarray_dims;j++){
      distance *= _coarray_elmts[j];
    }
    distance_of_coarray_elmts[i] = distance * _elmt_size;
  }
  distance_of_coarray_elmts[_coarray_dims-1] = _elmt_size;

  int *distance_of_image_elmts = _XMP_alloc(sizeof(int) * _image_dims);
  for(int i=_image_dims-1;i>=1;i--){
    int distance = 1;  
    for(int j=0;j<i;j++){
      distance *= _image_elmts[j];
    }
    distance_of_image_elmts[i] = distance;
  }
  distance_of_image_elmts[0] = 1;

  _XMP_coarray_t* c = _XMP_alloc(sizeof(_XMP_coarray_t));
  c->elmt_size      = _elmt_size;
  c->coarray_dims   = _coarray_dims;
  c->coarray_elmts  = _coarray_elmts;
  c->image_dims     = _image_dims;
  c->distance_of_coarray_elmts = distance_of_coarray_elmts;
  c->distance_of_image_elmts   = distance_of_image_elmts;
  *coarray                     = c;

#ifdef _XMP_GASNET
  _XMP_gasnet_malloc_do(*coarray, addr, (size_t)_total_coarray_elmts*_elmt_size);
#elif _XMP_FJRDMA
  _XMP_fjrdma_malloc_do(*coarray, addr, (size_t)_total_coarray_elmts*_elmt_size);
#endif
  
  free(_image_elmts);  // Note: Do not free() _coarray_elmts.

  _push_coarray_queue(c);
}

void _XMP_coarray_malloc_do_f(void **coarray, void *addr)
{
  _XMP_coarray_malloc_do(coarray, addr);
}

/**
   Set transfer coarray information when executing RDMA
 */
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

/**
   Set transfer array information when executing RDMA
 */
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

/**
   Set image information when executing RDMA
 */
void _XMP_coarray_rdma_image_set_1(const int n1)
{
  _image_dims   = 1;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
}

void _XMP_coarray_rdma_image_set_2(const int n1, const int n2)
{
  _image_dims   = 2;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
}

void _XMP_coarray_rdma_image_set_3(const int n1, const int n2, const int n3)
{
  _image_dims   = 3;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
}

void _XMP_coarray_rdma_image_set_4(const int n1, const int n2, const int n3, const int n4)
{
  _image_dims   = 4;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
}

void _XMP_coarray_rdma_image_set_5(const int n1, const int n2, const int n3, const int n4,
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

void _XMP_coarray_rdma_image_set_6(const int n1, const int n2, const int n3, const int n4,
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

void _XMP_coarray_rdma_image_set_7(const int n1, const int n2, const int n3, const int n4, 
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
static int _check_continuous(const _XMP_array_section_t *a, const int dims, const int transfer_elmts)
{
  // Only 1 elements is transferred.
  // ex) a[2]
  // ex) b
  if(transfer_elmts == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension is transferred.
  // ex) a[1][2][2:3]
  if(transfer_elmts == (a+dims-1)->length)
    if((a+dims-1)->stride == 1)
      return _XMP_N_INT_TRUE;

  // Does non-continuous dimension exist ?
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

/**
   Execute RDMA
 */
void _XMP_coarray_rdma_do(const int rdma_code, void *remote_coarray, void *local_array, void *local_coarray)
/* If a local array is a coarray, local_coarray != NULL. */
{
  if(_transfer_coarray_elmts == 0 || _transfer_array_elmts == 0) return;

  if(rdma_code == _XMP_N_COARRAY_GET){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_coarray_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;  // e.g. a[0:3] = b[0:2]:[3] is NG, but a[0:3] = b[0:1]:[3] is OK
  }
  else if(rdma_code == _XMP_N_COARRAY_PUT){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_array_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");  // e.g. a[0:3]:[3] = b[0:2] is NG, but a[0:3]:[3] = b[0:1] is OK.
  }

  int target_rank = 0;
  for(int i=0;i<_image_dims;i++)
    target_rank += ((_XMP_coarray_t*)remote_coarray)->distance_of_image_elmts[i] * (_image_num[i] - 1);

  for(int i=0;i<_coarray_dims;i++){
    _coarray[i].elmts    = ((_XMP_coarray_t*)remote_coarray)->coarray_elmts[i];
    _coarray[i].distance = ((_XMP_coarray_t*)remote_coarray)->distance_of_coarray_elmts[i];
  }

  int remote_coarray_is_continuous = _check_continuous(_coarray, _coarray_dims, _transfer_coarray_elmts);
  int local_array_is_continuous    = _check_continuous(_array,   _array_dims,   _transfer_array_elmts); 

  if(rdma_code == _XMP_N_COARRAY_PUT){
    if(target_rank == _XMP_world_rank){
      _XMP_local_put(remote_coarray_is_continuous, local_array_is_continuous, _coarray_dims, _array_dims,
		     _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts, _transfer_array_elmts);
#endif
    }
  }
  else if(rdma_code == _XMP_N_COARRAY_GET){
    if(target_rank == _XMP_world_rank){
      _XMP_local_get(remote_coarray_is_continuous, local_array_is_continuous, _coarray_dims, _array_dims,
                     _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, local_coarray, _transfer_coarray_elmts, _transfer_array_elmts);
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

/**
   Execute sync_all()
 */
void _XMP_coarray_sync_all()
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

/**
   Execute sync_memory()
*/
void _XMP_coarray_sync_memory()
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

/**
   Execute sync_all()
*/
void xmp_sync_memory(const int* status)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_memory();
#endif
}

/**
   Execute sync_memory()
*/
void xmp_sync_all(const int* status)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_all();
#endif
}

/**
   Execute sync_image()
*/
void xmp_sync_image(int image, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images()");
}

void xmp_sync_image_f(int *image, int* status)
{
  xmp_sync_image(*image, status);
}

/**
   Execute sync_images()
*/
void xmp_sync_images(int num, int* image_set, int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_images()");
}

void xmp_sync_images_f(int *num, int* image_set, int* status)
{
  xmp_sync_images(*num, image_set, status);
}

/**
   Execute sync_images_all()
*/
void xmp_sync_images_all(int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_all()");
}


/**
   Get offset
*/
size_t _XMP_get_offset(const _XMP_array_section_t *array, const int dims)
{
  size_t offset = 0;
  for(int i=0;i<dims;i++)
    offset += (array+i)->start * (array+i)->distance;

  return offset;
}

/**
   Execute put operation without preprocessing
*/
void _XMP_coarray_shortcut_put(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src, 
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;

  if(rank == _XMP_world_rank){
    memcpy(dst->real_addr + dst_offset, src->real_addr + src_offset, transfer_size);
  }
  else{
#ifdef _XMP_GASNET
    gasnet_put_nbi_bulk(rank, dst->addr[rank]+dst_offset,
			src->addr[_XMP_world_rank]+src_offset, transfer_size);
#elif _XMP_FJRDMA
    _XMP_fjrdma_shortcut_put(rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst, src, transfer_size);
#endif
  }
}

/**
   Execute get operation without preprocessing
*/
void _XMP_coarray_shortcut_get(const int target, const _XMP_coarray_t *dst, const _XMP_coarray_t *src,
			       const size_t dst_offset, const size_t src_offset, const size_t transfer_size)
{
  if(transfer_size == 0) return;
  int rank = target - 1;
  
  if(rank == _XMP_world_rank){
    memcpy(dst->real_addr + dst_offset, src->real_addr + src_offset, transfer_size);
  }
  else{
#ifdef _XMP_GASNET
    gasnet_get_bulk(dst->addr[_XMP_world_rank]+dst_offset, rank, src->addr[rank]+src_offset, transfer_size);
#elif _XMP_FJRDMA
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

/**
   Build queue for coarray
*/
void _XMP_build_coarray_queue()
{
  _coarray_queue.max_size = _XMP_COARRAY_QUEUE_INITIAL_SIZE;
  _coarray_queue.num      = 0;
  _coarray_queue.coarrays = malloc(sizeof(_XMP_coarray_t*) * _coarray_queue.max_size);
}

/**
   Rebuild the queue when the queue is full
*/
static void _rebuild_coarray_queue()
{
  _coarray_queue.max_size *= _XMP_COARRAY_QUEUE_INCREMENT_RAITO;
  _XMP_coarray_t **tmp;
  size_t next_size = _coarray_queue.max_size * sizeof(_XMP_coarray_t*);
  if((tmp = realloc(_coarray_queue.coarrays, next_size)) == NULL)
    _XMP_fatal("cannot allocate memory");
  else
    _coarray_queue.coarrays = tmp;
}

/**
   Push a coarray to the queue
*/
static void _push_coarray_queue(_XMP_coarray_t *c)
{
  if(_coarray_queue.num >= _coarray_queue.max_size)
    _rebuild_coarray_queue();

  _coarray_queue.coarrays[_coarray_queue.num++] = c;
}

/**
   Pop a coarray from the queue
*/
static _XMP_coarray_t* _pop_coarray_queue()
{
  if(_coarray_queue.num == 0) return NULL;

  _coarray_queue.num--;
  return _coarray_queue.coarrays[_coarray_queue.num];
}

/**
   Deallocate memory space and an object of coarray
*/
static void _XMP_coarray_deallocate(_XMP_coarray_t *c)
{
  if(c == NULL) return;

  free(c->addr);
#ifndef _XMP_GASNET
  free(c->real_addr);
#endif
  free(c->coarray_elmts);
  free(c->distance_of_coarray_elmts);
  free(c->distance_of_image_elmts);
  free(c);
}

/**
   Deallocate memory space and an object of the last coarray
*/
void _XMP_coarray_lastly_deallocate()
{
#ifdef _XMP_GASNET
  _XMP_gasnet_coarray_lastly_deallocate();
#elif _XMP_FJRDMA
  _XMP_fjrdma_coarray_lastly_deallocate();
#endif

  _XMP_coarray_t *_last_coarray_ptr = _pop_coarray_queue();
  _XMP_coarray_deallocate(_last_coarray_ptr);
}
