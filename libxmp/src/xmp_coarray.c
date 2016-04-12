#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "xmp_internal.h"
#include "xmp_constant.h"

static size_t _elmt_size;
static int _coarray_dims;
static long *_coarray_elmts, _total_coarray_elmts;

static int _image_dims, *_image_elmts;

static int _array_dims;
static long _transfer_coarray_elmts, _transfer_array_elmts;
static int *_image_num;

static _XMP_array_section_t *_coarray, *_array;
struct _coarray_queue_t{
  size_t           max_size; /**< Max size of queue */
  int                   num; /**< How many coarrays are in this queue */
  _XMP_coarray_t **coarrays; /**< pointer of coarrays */
};
static struct _coarray_queue_t _coarray_queue;
static void _push_coarray_queue(_XMP_coarray_t *c);

/**
   Set 1-dim coarray information 
 */
void _XMP_coarray_malloc_info_1(const long n1, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 1;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _total_coarray_elmts = n1;
}

/**
   Set 2-dim coarray information
*/
void _XMP_coarray_malloc_info_2(const long n1, const long n2, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 2;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _total_coarray_elmts = n1*n2;
}

/**
   Set 3-dim coarray information
*/
void _XMP_coarray_malloc_info_3(const long n1, const long n2, const long n3, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 3;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _total_coarray_elmts = n1*n2*n3;
}

/**
   Set 4-dim coarray information
*/
void _XMP_coarray_malloc_info_4(const long n1, const long n2, const long n3, const long n4,
				const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 4;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _total_coarray_elmts = n1*n2*n3*n4;
}

/**
   Set 5-dim coarray information
*/
void _XMP_coarray_malloc_info_5(const long n1, const long n2, const long n3, const long n4,
				const long n5, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 5;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _coarray_elmts[4]    = n5;
  _total_coarray_elmts = n1*n2*n3*n4*n5;
}

/**
   Set 6-dim coarray information
*/
void _XMP_coarray_malloc_info_6(const long n1, const long n2, const long n3, const long n4,
				const long n5, const long n6, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 6;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _coarray_elmts[0]    = n1;
  _coarray_elmts[1]    = n2;
  _coarray_elmts[2]    = n3;
  _coarray_elmts[3]    = n4;
  _coarray_elmts[4]    = n5;
  _coarray_elmts[5]    = n6;
  _total_coarray_elmts = n1*n2*n3*n4*n5*n6;
}

/**
   Set 7-dim coarray information
*/
void _XMP_coarray_malloc_info_7(const long n1, const long n2, const long n3, const long n4, 
				const long n5, const long n6, const long n7, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = 7;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
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
   Set n-dim coarray information
*/
void _XMP_coarray_malloc_info_n(const long *n, const int ndims, const size_t elmt_size)
{
  _elmt_size           = elmt_size;
  _coarray_dims        = ndims;
  _coarray_elmts       = malloc(sizeof(long) * _coarray_dims);
  _total_coarray_elmts = 1;
  for (int i = 0; i < ndims; i++){
    _coarray_elmts[i]    = n[i];
    _total_coarray_elmts *= n[i];
  }
}

/**
    Set 1-dim image information
 */
void _XMP_coarray_malloc_image_info_1()
{
  _image_dims     = 1;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = 1;
}

/**
   Check total_node_size and total_image_size are valid.
 */
static void _check_coarray_image(const int total_node_size, const int image_size)
{
  if(total_node_size % image_size != 0)
    _XMP_fatal("Wrong coarray image size.");
}

/**
    Set 2-dim image information
*/
void _XMP_coarray_malloc_image_info_2(const int i1)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  _check_coarray_image(total_node_size, i1);
    
  _image_dims     = 2;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = total_node_size / i1;
}

/**
    Set 3-dim image information
*/
void _XMP_coarray_malloc_image_info_3(const int i1, const int i2)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
 
  _check_coarray_image(total_node_size, i1*i2);

  _image_dims     = 3;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = total_node_size / (i1*i2);
}

/**
    Set 4-dim image information
*/
void _XMP_coarray_malloc_image_info_4(const int i1, const int i2, const int i3)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  _check_coarray_image(total_node_size, i1*i2*i3);

  _image_dims     = 4;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = total_node_size / (i1*i2*i3);
}

/**
    Set 5-dim image information
*/
void _XMP_coarray_malloc_image_info_5(const int i1, const int i2, const int i3, const int i4)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  _check_coarray_image(total_node_size, i1*i2*i3*i4);

  _image_dims     = 5;
  _image_elmts    = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = total_node_size / (i1*i2*i3*i4);
}

/**
    Set 6-dim image information
*/
void _XMP_coarray_malloc_image_info_6(const int i1, const int i2, const int i3, const int i4,
                                      const int i5)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  _check_coarray_image(total_node_size, i1*i2*i3*i4*i5);

  _image_dims    = 6;
  _image_elmts   = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = i5;
  _image_elmts[5] = total_node_size / (i1*i2*i3*i4*i5);
}

void _XMP_coarray_malloc_image_info_7(const int i1, const int i2, const int i3, const int i4,
                                      const int i5, const int i6)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  _check_coarray_image(total_node_size, i1*i2*i3*i4*i5*i6);

  _image_dims    = 7;
  _image_elmts   = malloc(sizeof(int) * _image_dims);
  _image_elmts[0] = i1;
  _image_elmts[1] = i2;
  _image_elmts[2] = i3;
  _image_elmts[3] = i4;
  _image_elmts[4] = i5;
  _image_elmts[5] = i6;
  _image_elmts[6] = total_node_size / (i1*i2*i3*i4*i5*i6);
}

void _XMP_coarray_malloc_image_info_n(const int *i, const int ndims)
{
  int total_node_size  = _XMP_get_execution_nodes()->comm_size;

  int t = 1;
  for (int j = 0; j < ndims-1; j++){
    t *= i[j];
  }

  _check_coarray_image(total_node_size, t);

  _image_dims    = ndims;
  _image_elmts   = malloc(sizeof(int) * _image_dims);
  for (int j = 0; j < ndims-1; j++){
    _image_elmts[j] = i[j];
  }
  _image_elmts[ndims-1] = total_node_size / t;
}

/*
   Set infomation to coarray descriptor
*/
void _XMP_coarray_set_info(_XMP_coarray_t* c)
{
  long *distance_of_coarray_elmts = _XMP_alloc(sizeof(long) * _coarray_dims);

  for(int i=0;i<_coarray_dims-1;i++){
    long distance = 1;
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

  c->elmt_size                 = _elmt_size;
  c->coarray_dims              = _coarray_dims;
  c->coarray_elmts             = _coarray_elmts;
  c->image_dims                = _image_dims;
  c->distance_of_coarray_elmts = distance_of_coarray_elmts;
  c->distance_of_image_elmts   = distance_of_image_elmts;

  free(_image_elmts);  // Note: Do not free() _coarray_elmts.
}

/**
   Create coarray object and allocate coarray.
 */
void _XMP_coarray_malloc_do(void **coarray_desc, void *addr)
{
  _XMP_coarray_t* c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_set_info(c);
  *coarray_desc = c;

  long transfer_size = _total_coarray_elmts*_elmt_size;
  _XMP_check_less_than_SIZE_MAX(transfer_size);
  
#ifdef _XMP_GASNET
  _XMP_gasnet_malloc_do(*coarray_desc, addr, (size_t)transfer_size);
#elif _XMP_FJRDMA
  _XMP_fjrdma_malloc_do(*coarray_desc, addr, (size_t)transfer_size);
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_malloc_do(*coarray_desc, addr, (size_t)transfer_size, false);
#endif
  
  _push_coarray_queue(c);
}


/**
   Create coarray object but not allocate coarray.
 */
void _XMP_coarray_regmem_do(void **coarray_desc, void *addr)
{
  _XMP_coarray_t* c = _XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_coarray_set_info(c);
  *coarray_desc = c;

  long transfer_size = _total_coarray_elmts*_elmt_size;
  _XMP_check_less_than_SIZE_MAX(transfer_size);
  
#if _XMP_GASNET
  //not implemented
  _XMP_fatal("_XMP_coarray_regmem_do is not supported over GASNet.\n");
#elif _XMP_FJRDMA
  _XMP_fjrdma_regmem_do(*coarray_desc, addr, (size_t)transfer_size);
#elif _XMP_MPI3_ONESIDED
  //not implemented
  _XMP_fatal("_XMP_coarray_regmem_do is not supported over MPI3.\n");
#endif

  _push_coarray_queue(c);
}


/** 
   Attach memory to coarray
 */
void _XMP_coarray_attach(_XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size)
{
  _XMP_coarray_set_info(coarray_desc);

#ifdef _XMP_GASNET
  //not implemented
  _XMP_fatal("_XMP_gasnet_coarray_attach is not implemented\n");
#elif _XMP_FJRDMA
  //not implemented
  _XMP_fatal("_XMP_fjrdma_coarray_attach is not implemented\n");
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_attach(coarray_desc, addr, coarray_size, false);
#endif

  _push_coarray_queue(coarray_desc);
}

/** 
   Detach memory from coarray
 */
void _XMP_coarray_detach(_XMP_coarray_t *coarray_desc)
{
#ifdef _XMP_GASNET
  //not implemented
  _XMP_fatal("_XMP_gasnet_coarray_detach is not implemented\n");
#elif _XMP_FJRDMA
  //not implemented
  _XMP_fatal("_XMP_fjrdma_coarray_detach is not implemented\n");
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_detach(coarray_desc, false);
#endif
}

/**
   Wrapper function of _XMP_coarray_malloc_do()
*/
void _XMP_coarray_malloc_do_f(void **coarray_desc, void *addr)
{
  _XMP_coarray_malloc_do(coarray_desc, addr);
}

/**
   Set transfer 1-dim coarray information
 */
void _XMP_coarray_rdma_coarray_set_1(const long start1, const long length1, const long stride1)
{
  _transfer_coarray_elmts = length1;
  _coarray_dims           = 1;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  _coarray[0].start       = start1;
  _coarray[0].length      = length1;
  _coarray[0].stride      = ((length1 == 1)? 1 : stride1);
}

/**
   Set transfer 2-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_2(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2)
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

/**
   Set transfer 3-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_3(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2,
                                     const long start3, const long length3, const long stride3)
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

/**
   Set transfer 4-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_4(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2,
                                     const long start3, const long length3, const long stride3, 
				     const long start4, const long length4, const long stride4)
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

/**
   Set transfer 5-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_5(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2,
                                     const long start3, const long length3, const long stride3, 
				     const long start4, const long length4, const long stride4,
                                     const long start5, const long length5, const long stride5)
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

/**
   Set transfer 6-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_6(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2,
                                     const long start3, const long length3, const long stride3, 
				     const long start4, const long length4, const long stride4,
                                     const long start5, const long length5, const long stride5, 
				     const long start6, const long length6, const long stride6)
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

/**
   Set transfer 7-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_7(const long start1, const long length1, const long stride1, 
				     const long start2, const long length2, const long stride2,
				     const long start3, const long length3, const long stride3, 
				     const long start4, const long length4, const long stride4,
				     const long start5, const long length5, const long stride5, 
				     const long start6, const long length6, const long stride6,
				     const long start7, const long length7, const long stride7)
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
   Set transfer n-dim coarray information
*/
void _XMP_coarray_rdma_coarray_set_n(const int n,
				     const long start[], const long length[], const long stride[]) 
{
  _transfer_coarray_elmts = 1;
  _coarray_dims           = n;
  _coarray                = malloc(sizeof(_XMP_array_section_t) * _coarray_dims);

  for (int i = 0; i < n; i++){
    _transfer_coarray_elmts *= length[i];
    _coarray[i].start       = start[i];
    _coarray[i].length      = length[i];
    _coarray[i].stride      = ((length[i] == 1)? 1 : stride[i]);
  }
}

/**
   Set transfer 1-dim array information
 */
void _XMP_coarray_rdma_array_set_1(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1)
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

/**
   Set transfer 2-dim array information
*/
void _XMP_coarray_rdma_array_set_2(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
                                   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2)
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

/**
   Set transfer 3-dim array information
*/
void _XMP_coarray_rdma_array_set_3(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
                                   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2,
                                   const long start3, const long length3, const long stride3,
				   const long elmts3, const long distance3)
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

/**
   Set transfer 4-dim array information
*/
void _XMP_coarray_rdma_array_set_4(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
                                   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2,
                                   const long start3, const long length3, const long stride3,
				   const long elmts3, const long distance3,
                                   const long start4, const long length4, const long stride4,
				   const long elmts4, const long distance4)
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

/**
   Set transfer 5-dim array information
*/
void _XMP_coarray_rdma_array_set_5(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
                                   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2,
                                   const long start3, const long length3, const long stride3,
				   const long elmts3, const long distance3,
                                   const long start4, const long length4, const long stride4,
				   const long elmts4, const long distance4,
                                   const long start5, const long length5, const long stride5,
				   const long elmts5, const long distance5)
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

/**
   Set transfer 6-dim array information
*/
void _XMP_coarray_rdma_array_set_6(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
				   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2,
                                   const long start3, const long length3, const long stride3,
				   const long elmts3, const long distance3,
                                   const long start4, const long length4, const long stride4,
				   const long elmts4, const long distance4,
                                   const long start5, const long length5, const long stride5,
				   const long elmts5, const long distance5,
                                   const long start6, const long length6, const long stride6,
				   const long elmts6, const long distance6)
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

/**
   Set transfer 7-dim array information
*/
void _XMP_coarray_rdma_array_set_7(const long start1, const long length1, const long stride1,
				   const long elmts1, const long distance1,
				   const long start2, const long length2, const long stride2,
				   const long elmts2, const long distance2,
				   const long start3, const long length3, const long stride3,
				   const long elmts3, const long distance3,
				   const long start4, const long length4, const long stride4,
				   const long elmts4, const long distance4,
				   const long start5, const long length5, const long stride5,
				   const long elmts5, const long distance5,
				   const long start6, const long length6, const long stride6,
				   const long elmts6, const long distance6,
				   const long start7, const long length7, const long stride7,
				   const long elmts7, const long distance7)
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
   Set transfer n-dim array information
*/
void _XMP_coarray_rdma_array_set_n(const int n,
				   const long start[], const long length[], const long stride[],
				   const long elmts[], const long distance[])
{
  _transfer_array_elmts = 1;
  _array_dims           = n;
  _array                = malloc(sizeof(_XMP_array_section_t) * _array_dims);

  for (int i = 0; i < n; i++){
    _transfer_array_elmts *= length[i];
    _array[i].start       = start[i];
    _array[i].length      = length[i];
    _array[i].stride      = ((length[i] == 1)? 1 : stride[i]);
    _array[i].elmts       = elmts[i];
    _array[i].distance    = distance[i];
  }
}

/**
   Set 1-dim image information
 */
void _XMP_coarray_rdma_image_set_1(const int n1)
{
  _image_dims   = 1;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
}

/**
   Set 2-dim image information
*/
void _XMP_coarray_rdma_image_set_2(const int n1, const int n2)
{
  _image_dims   = 2;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
}

/**
   Set 3-dim image information
*/
void _XMP_coarray_rdma_image_set_3(const int n1, const int n2, const int n3)
{
  _image_dims   = 3;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
}

/**
   Set 4-dim image information
*/
void _XMP_coarray_rdma_image_set_4(const int n1, const int n2, const int n3, const int n4)
{
  _image_dims   = 4;
  _image_num    = malloc(sizeof(int) * _image_dims);
  _image_num[0] = n1;
  _image_num[1] = n2;
  _image_num[2] = n3;
  _image_num[3] = n4;
}

/**
   Set 5-dim image information
*/
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

/**
   Set 6-dim image information
*/
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

/**
   Set 7-dim image information
*/
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

/**
   Set n-dim image information
*/
void _XMP_coarray_rdma_image_set_n(const int ndims, const int n[])
{
  _image_dims   = ndims;
  _image_num    = malloc(sizeof(int) * _image_dims);

  for (int i = 0; i < ndims; i++){
    _image_num[i] = n[i];
  }
}

/*************************************************************************/
/* DESCRIPTION : Check region is continuous                              */
/* ARGUMENT    : [IN] *array_info : Information of array                 */
/*               [IN] dims        : Number of dimensions of array        */
/*               [IN] elmts       : Number of transfer elements of array */
/* RETURN      : If the region is continuous, return TRUE                */
/* EXAMPLE     : a[:]      -> TRUE                                       */
/*               a[:100:2] -> FALSE                                      */
/*               a[0:2][:] -> TRUE                                       */
/*               a[:][1]   -> FALSE                                      */
/*************************************************************************/
static int _check_continuous(const _XMP_array_section_t *array_info, const int dims, const long elmts)
{
  // Only 1 elements is transferred.
  // e.g.) a[2]
  if(elmts == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension is transferred.
  // e.g.) a[1][2][2:3]
  if(array_info[dims-1].length == elmts && array_info[dims-1].stride == 1)
    return _XMP_N_INT_TRUE;

  // Does non-continuous dimension exist ?
  for(int i=0;i<dims;i++)
    if(array_info[i].stride != 1 && array_info[i].length != 1)
      return _XMP_N_INT_FALSE;

  // (.., d-3, d-2)-th dimension's length is "1" &&
  // d-1-th stride is "1" &&
  // (d, d+1, ..)-th dimensions are ":".
  // e.g.) a[1][3][1:2][:]    // d == 3
  //       a[1][3:2:2][:][:]  // d == 2
  //       a[2][:][:][:]      // d == 1
  //       a[:][:][:][:]      // d == 0
  
  int d = _XMP_get_dim_of_allelmts(dims, array_info);
  if(d == 0){
    return _XMP_N_INT_TRUE;
  }
  else if(d == 1){
    if(array_info[0].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 2){
    if(array_info[0].length == 1 && array_info[1].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 3){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 4){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 5){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 6){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].length == 1 && array_info[5].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
  else if(d == 7){
    if(array_info[0].length == 1 && array_info[1].length == 1 &&
       array_info[2].length == 1 && array_info[3].length == 1 &&
       array_info[4].length == 1 && array_info[5].length == 1 &&
       array_info[6].stride == 1)
      return _XMP_N_INT_TRUE;
    else
      return _XMP_N_INT_FALSE;
  }
 
  _XMP_fatal("Unexpected Error!\n");
  return -1; // dummy
}

/*****************************************************************************/
/* DESCRIPTION : Execute put/get operation                                   */
/* ARGUMENT    : [IN] rdma_code      : _XMP_N_COARRAY_PUT/_XMP_N_COARRAY_GET */
/*               [IN/OUT] *remote_coarray : Descriptor of remote coarray     */
/*               [IN/OUT] *local_array    : Descriptor of local coarray      */
/*               [IN/OUT] *local_coarray  : Descriptor of local coarray      */
/* NOTE        :                                                             */
/*     If a local_array is NOT a coarray, local_coarray == NULL.             */
/*****************************************************************************/
void _XMP_coarray_rdma_do(const int rdma_code, void *remote_coarray, void *local_array, void *local_coarray)
{
  if(_transfer_coarray_elmts == 0 || _transfer_array_elmts == 0) return;

  if(rdma_code == _XMP_N_COARRAY_GET){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_coarray_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;
    // e.g. a[0:3] = b[0:2]:[3] is NG, but a[0:3] = b[0:1]:[3] is OK
  }
  else if(rdma_code == _XMP_N_COARRAY_PUT){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_array_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    // e.g. a[0:3]:[3] = b[0:2] is NG, but a[0:3]:[3] = b[0:1] is OK.
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

  // _XMP_local_put(), _XMP_gasnet_put(), ... don't support to transfer long data size now.
  // _XMP_check_less_than_SIZE_MAX() checks whether the trasfer size is less than SIZE_MAX, defined in <limits.h>.
  size_t elmt_size  = ((_XMP_coarray_t*)remote_coarray)->elmt_size;
  _XMP_check_less_than_SIZE_MAX(_transfer_coarray_elmts*elmt_size); // fix me
  _XMP_check_less_than_SIZE_MAX(_transfer_array_elmts*elmt_size);   // fix me
  
  if(rdma_code == _XMP_N_COARRAY_PUT){
    if(target_rank == _XMP_world_rank){
      _XMP_local_put(remote_coarray, local_array, remote_coarray_is_continuous, local_array_is_continuous, 
		     _coarray_dims, _array_dims, _coarray, _array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims,
		   _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts,
		   _XMP_N_INT_FALSE);
#endif
    }
  }
  else if(rdma_code == _XMP_N_COARRAY_GET){
    if(target_rank == _XMP_world_rank){
      _XMP_local_get(local_array, remote_coarray, local_array_is_continuous, remote_coarray_is_continuous,
		     _array_dims, _coarray_dims, _array, _coarray, (size_t)_transfer_array_elmts, (size_t)_transfer_coarray_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims,
		   _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts,
		   _XMP_N_INT_FALSE);
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


void _XMP_coarray_rdma_do2(const int rdma_code, void *remote_coarray, void *local_array, void *local_coarray,
			   const long coarray_elmts[], const long coarray_distance[])
{
  if(_transfer_coarray_elmts == 0 || _transfer_array_elmts == 0) return;

  if(rdma_code == _XMP_N_COARRAY_GET){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_coarray_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;
    // e.g. a[0:3] = b[0:2]:[3] is NG, but a[0:3] = b[0:1]:[3] is OK
  }
  else if(rdma_code == _XMP_N_COARRAY_PUT){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_array_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    // e.g. a[0:3]:[3] = b[0:2] is NG, but a[0:3]:[3] = b[0:1] is OK.
  }

  int target_rank = 0;
  for(int i=0;i<_image_dims;i++)
    target_rank += ((_XMP_coarray_t*)remote_coarray)->distance_of_image_elmts[i] * (_image_num[i] - 1);

  for(int i=0;i<_coarray_dims;i++){
    _coarray[i].elmts    = coarray_elmts[i];
    _coarray[i].distance = coarray_distance[i];
  }

  int remote_coarray_is_continuous = _check_continuous(_coarray, _coarray_dims, _transfer_coarray_elmts);
  int local_array_is_continuous    = _check_continuous(_array,   _array_dims,   _transfer_array_elmts); 

  // _XMP_local_put(), _XMP_gasnet_put(), ... don't support to transfer long data size now.
  // _XMP_check_less_than_SIZE_MAX() checks whether the trasfer size is less than SIZE_MAX, defined in <limits.h>.
  size_t elmt_size  = ((_XMP_coarray_t*)remote_coarray)->elmt_size;
  _XMP_check_less_than_SIZE_MAX(_transfer_coarray_elmts*elmt_size); // fix me
  _XMP_check_less_than_SIZE_MAX(_transfer_array_elmts*elmt_size);   // fix me
  
  if(rdma_code == _XMP_N_COARRAY_PUT){
    if(target_rank == _XMP_world_rank){
      _XMP_local_put(remote_coarray, local_array, remote_coarray_is_continuous, local_array_is_continuous, 
		     _coarray_dims, _array_dims, _coarray, _array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims,
		   _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts,
		   _XMP_N_INT_FALSE);
#endif
    }
  }
  else if(rdma_code == _XMP_N_COARRAY_GET){
    if(target_rank == _XMP_world_rank){
      _XMP_local_get(local_array, remote_coarray, local_array_is_continuous, remote_coarray_is_continuous,
		     _array_dims, _coarray_dims, _array, _coarray, (size_t)_transfer_array_elmts, (size_t)_transfer_coarray_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims,
		   _coarray, _array, remote_coarray, local_array, (size_t)_transfer_coarray_elmts, (size_t)_transfer_array_elmts,
		   _XMP_N_INT_FALSE);
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


/**
   Wrapper function of _XMP_coarray_rdma_do()
*/
void _XMP_coarray_rdma_do_f(const int *rdma_code, void *remote_coarray, void *local_array, void *local_coarray)
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
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_sync_all();
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
#elif _XMP_TCA
  _XMP_tca_sync_memory();
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_sync_memory();
#endif
}

/**
   Execute sync_memory()
*/
void xmp_sync_memory(const int* status)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_memory();
#elif _XMP_TCA
  _XMP_tca_sync_memory();
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_sync_memory();
#endif
}

/**
   Execute sync_all()
*/
void xmp_sync_all(const int* status)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_all();
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_sync_all();
#endif
}

/**
   Execute sync_images()
*/
void xmp_sync_images(const int num, int* image_set, int* status)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_sync_images(num, image_set, status);
#elif _XMP_FJRDMA
  _XMP_fjrdma_sync_images(num, image_set, status);
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_sync_images(num, image_set, status);
#endif
}

/**
   Wrapper function of xmp_sync_images()
 */
void xmp_sync_images_f(const int *num, int* image_set, int* status)
{
  xmp_sync_images(*num, image_set, status);
}

/**
   Execute sync_image()
*/
void xmp_sync_image(int image, int* status)
{
  xmp_sync_images(1, &image, status);
}

/**
   Wrapper function of xmp_sync_image()
*/
void xmp_sync_image_f(int *image, int* status)
{
  xmp_sync_images(1, image, status);
}

/**
   Execute sync_images_all()
*/
void xmp_sync_images_all(int* status)
{
  _XMP_fatal("Not implement xmp_sync_images_all()");
}

/************************************************************************/
/* DESCRIPTION : Execute put operation without preprocessing            */
/* ARGUMENT    : [IN] target_image : Target image                       */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/* NOTE       : Both dst and src are continuous coarrays                */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_coarray_shortcut_put(const int target_image, _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
			       const long dst_offset, const long src_offset, const long dst_elmts, const long src_elmts)
{
  int target_rank = target_image - 1;
  size_t elmt_size = dst_desc->elmt_size;
  
  if(target_rank == _XMP_world_rank){
    _XMP_local_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src_desc->real_addr+src_offset, 
			       dst_elmts, src_elmts, elmt_size);
  }
  else{
    _XMP_check_less_than_SIZE_MAX(dst_elmts);
    _XMP_check_less_than_SIZE_MAX(src_elmts);
#ifdef _XMP_GASNET
    _XMP_gasnet_shortcut_put(target_rank, dst_desc, src_desc->addr[_XMP_world_rank]+src_offset,
			     (size_t)dst_offset, (size_t)dst_elmts, (size_t)src_elmts, elmt_size);
#elif _XMP_FJRDMA
    _XMP_fjrdma_shortcut_put(target_rank, (uint64_t)dst_offset, (uint64_t)src_offset, dst_desc, src_desc, 
			     (size_t)dst_elmts, (size_t)src_elmts, elmt_size);
#elif _XMP_TCA
    _XMP_fatal("_XMP_tca_shortcut_put is unimplemented");
#elif _XMP_MPI3_ONESIDED
    _XMP_mpi_shortcut_put(target_rank, dst_desc, src_desc, (size_t)dst_offset, (size_t)src_offset,
			  (size_t)dst_elmts, (size_t)src_elmts, elmt_size, false, false);
#endif
  }
}

/************************************************************************/
/* DESCRIPTION : Execute get operation without preprocessing            */
/* ARGUMENT    : [IN] target_image : Target image                       */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/* NOTE       : Both dst and src are continuous coarrays                */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_coarray_shortcut_get(const int target_image, _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
			       const long dst_offset, const long src_offset, const long dst_elmts,
			       const long src_elmts)
{
  int target_rank = target_image - 1;
  size_t elmt_size = dst_desc->elmt_size;
 
  if(target_rank == _XMP_world_rank){
    _XMP_local_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src_desc->real_addr+src_offset,
			       dst_elmts, src_elmts, elmt_size);
  }
  else{
    _XMP_check_less_than_SIZE_MAX(dst_elmts);
    _XMP_check_less_than_SIZE_MAX(src_elmts);
#ifdef _XMP_GASNET
    _XMP_gasnet_shortcut_get(target_rank, dst_desc, src_desc->addr[target_rank]+src_offset, (size_t)dst_offset,
                             (size_t)dst_elmts, (size_t)src_elmts, elmt_size);
#elif _XMP_FJRDMA
    _XMP_fjrdma_shortcut_get(target_rank, dst_desc, src_desc, (uint64_t)dst_offset, (uint64_t)src_offset, 
			     (size_t)dst_elmts, (size_t)src_elmts, elmt_size);
#elif _XMP_MPI3_ONESIDED
    _XMP_mpi_shortcut_get(target_rank, dst_desc, src_desc, (size_t)dst_offset, (size_t)src_offset,
			  (size_t)dst_elmts, (size_t)src_elmts, elmt_size, false, false);
#endif
  }
}

/**
   Wrapper function of _XMP_coarray_shortcut_put()
*/
void _XMP_coarray_shortcut_put_f(const int *target, void *dst, const void *src, const long *dst_offset, 
				 const long *src_offset, const long *dst_elmts, const long *src_elmts)
{
  _XMP_coarray_shortcut_put(*target, dst, src, *dst_offset, *src_offset, *dst_elmts, *src_elmts);
}

/**
   Wrapper function of _XMP_coarray_shortcut_get()
*/
void _XMP_coarray_shortcut_get_f(const int *target, void *dst, const void *src, const long *dst_offset, 
				 const long *src_offset, const long *dst_elmts, const long *src_elmts)
{
  _XMP_coarray_shortcut_get(*target, dst, src, *dst_offset, *src_offset, *dst_elmts, *src_elmts);
}

/**
   Build table for sync images
*/
void _XMP_build_sync_images_table()
{
#ifdef _XMP_GASNET
  _XMP_gasnet_build_sync_images_table();
#elif _XMP_FJRDMA
  _XMP_fjrdma_build_sync_images_table();
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_build_sync_images_table();
#endif
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
#if !defined(_XMP_GASNET) && !defined(_XMP_MPI3_ONESIDED)
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
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_lastly_deallocate(false);
#endif

  _XMP_coarray_t *_last_coarray_ptr = _pop_coarray_queue();
  _XMP_coarray_deallocate(_last_coarray_ptr);
}

/*****************************************************************************/
/* DESCRIPTION : Execute put/get operation                                   */
/* ARGUMENT    : [IN] rdma_code      : _XMP_N_COARRAY_PUT/_XMP_N_COARRAY_GET */
/*               [IN/OUT] *remote_coarray : Descriptor of remote coarray     */
/*               [IN/OUT] *local_array    : Descriptor of local coarray      */
/*               [IN/OUT] *local_coarray  : Descriptor of local coarray      */
/* NOTE        :                                                             */
/*     If a local_array is NOT a coarray, local_coarray == NULL.             */
/*****************************************************************************/
void _XMP_coarray_rdma_do_acc(const int rdma_code, void *remote_coarray, void *local_array,
			      void *local_coarray, const int is_remote_on_acc, const int is_local_on_acc)
{
  if(_transfer_coarray_elmts == 0 || _transfer_array_elmts == 0) return;

  if(rdma_code == _XMP_N_COARRAY_GET){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_coarray_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;
    // e.g. a[0:3] = b[0:2]:[3] is NG, but a[0:3] = b[0:1]:[3] is OK
  }
  else if(rdma_code == _XMP_N_COARRAY_PUT){
    if(_transfer_coarray_elmts != _transfer_array_elmts && _transfer_array_elmts != 1)
      _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
    // e.g. a[0:3]:[3] = b[0:2] is NG, but a[0:3]:[3] = b[0:1] is OK.
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
      _XMP_fatal("_XMP_coarray_rdma_do_acc: local_put is unimplemented");
      /* _XMP_local_put(remote_coarray, local_array, remote_coarray_is_continuous, local_array_is_continuous,  */
      /* 		     _coarray_dims, _array_dims, _coarray, _array, _transfer_coarray_elmts, _transfer_array_elmts); */
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_put(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		   _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts,
		   is_remote_on_acc);
#endif
    }
  }
  else if(rdma_code == _XMP_N_COARRAY_GET){
    if(target_rank == _XMP_world_rank){
      _XMP_local_get(local_array, remote_coarray, local_array_is_continuous, remote_coarray_is_continuous,
		     _array_dims, _coarray_dims, _array, _coarray, _transfer_array_elmts, _transfer_coarray_elmts);
    }
    else{
#ifdef _XMP_GASNET
      _XMP_gasnet_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank,
		      _coarray_dims, _array_dims, _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_FJRDMA
      _XMP_fjrdma_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims, 
		      _coarray, _array, remote_coarray, local_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts);
#elif _XMP_MPI3_ONESIDED
      _XMP_mpi_get(remote_coarray_is_continuous, local_array_is_continuous, target_rank, _coarray_dims, _array_dims,
		   _coarray, _array, remote_coarray, local_array, _transfer_coarray_elmts, _transfer_array_elmts,
		   is_remote_on_acc);
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
