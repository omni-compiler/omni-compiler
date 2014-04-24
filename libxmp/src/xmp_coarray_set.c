#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "xmp_internal.h"

static unsigned long long _xmp_heap_size, _xmp_stride_size;
static int _elmt_size, _coarray_dims, _image_dims, *_image_elmts;
static long long *_coarray_elmts, _total_coarray_elmts;

void _XMP_coarray_initialize(int argc, char **argv)
{
  char *env_heap_size, *env_stride_size;
  int i;

  if((env_heap_size = getenv("XMP_COARRAY_HEAP_SIZE")) != NULL){
    for(i=0;i<strlen(env_heap_size);i++){
      if(isdigit(env_heap_size[i]) == 0){
        fprintf(stderr, "%s : ", env_heap_size);
        _XMP_fatal("Unexpected Charactor in XMP_COARRAY_HEAP_SIZE");
      }
    }
    _xmp_heap_size = atoi(env_heap_size) * 1024 * 1024;
    if(_xmp_heap_size <= 0){
      _XMP_fatal("XMP_COARRAY_HEAP_SIZE is less than 0 !!");
    }
  }
  else{
    _xmp_heap_size = _XMP_DEFAULT_COARRAY_HEAP_SIZE;
  }

  if((env_stride_size = getenv("XMP_COARRAY_STRIDE_SIZE")) != NULL){
    for(i=0;i<strlen(env_stride_size);i++){
      if(isdigit(env_stride_size[i]) == 0){
        fprintf(stderr, "%s : ", env_stride_size);
        _XMP_fatal("Unexpected Charactor in XMP_COARRAY_STRIDE_SIZE");
      }
    }
    _xmp_stride_size = atoi(env_stride_size) * 1024 * 1024;
    if(_xmp_stride_size <= 0){
      _XMP_fatal("XMP_COARRAY_STRIDE_SIZE is less than 0 !!");
    }
  }
  else{
    _xmp_stride_size = _XMP_DEFAULT_COARRAY_STRIDE_SIZE;
  }

#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_initialize(argc, argv, _xmp_heap_size, _xmp_stride_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_initialize();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_finalize(int return_val)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_finalize(return_val);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_finalize();
#endif
}

void _XMP_coarray_malloc_set(int elmt_size, int coarray_dims, int image_dims)
{
  _elmt_size     = elmt_size;
  _coarray_dims  = coarray_dims;
  _coarray_elmts = malloc(sizeof(long long) * coarray_dims);
  _image_dims    = image_dims;
  _image_elmts   = malloc(sizeof(int) * image_dims);
  _total_coarray_elmts = 1;
}

void _XMP_coarray_malloc_set_f(int *elmt_size, int *coarray_dims, int *image_dims)
{
  _XMP_coarray_malloc_set(*elmt_size, *coarray_dims, *image_dims);
}

void _XMP_coarray_malloc_array_info(int dim, long long coarray_elmts)
{
  _coarray_elmts[dim]   = coarray_elmts;
  _total_coarray_elmts *= coarray_elmts;
}

void _XMP_coarray_malloc_array_info_f(int *dim, long long *coarray_elmts)
{
  _XMP_coarray_malloc_array_info(*dim, *coarray_elmts);
}

void _XMP_coarray_malloc_image_info(int dim, int image_elmts)
{
  _image_elmts[dim] = image_elmts;
}

void _XMP_coarray_malloc_image_info_f(int *dim, int *image_elmts)
{
  _XMP_coarray_malloc_image_info(*dim, *image_elmts);
}

void _XMP_coarray_malloc_do(void **coarray, void *addr)
{
  long long *distance_of_coarray_elmts = _XMP_alloc(sizeof(long long) * _coarray_dims);

  for(int i=0;i<_coarray_dims-1;i++){
    long long distance = 1;
    for(int j=i+1;j<_coarray_dims;j++){
      distance *= _coarray_elmts[j];
    }
    distance_of_coarray_elmts[i] = distance * _elmt_size;
  }
  distance_of_coarray_elmts[_coarray_dims-1] = _elmt_size;

  int total_node_size  = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = 1;
  for(int i=0;i<_image_dims-1;i++)
    total_image_size *= _image_elmts[i];

  if(total_node_size % total_image_size != 0)
    _XMP_fatal("Wrong coarray image size.");

  _image_elmts[_image_dims-1] = total_node_size / total_image_size;

  int *distance_of_image_elmts = _XMP_alloc(sizeof(int) * _image_dims);
  for(int i=_image_dims-1;i>=1;i--){
    int distance = 1;  
    for(int j=0;j<i;j++){
      distance *= _image_elmts[j];
    }
    distance_of_image_elmts[i] = distance;
  }
  distance_of_image_elmts[0] = 1;

  _XMP_coarray_t* c      = _XMP_alloc(sizeof(_XMP_coarray_t));
  c->elmt_size     = _elmt_size;
  c->coarray_dims  = _coarray_dims;
  c->coarray_elmts = _coarray_elmts;
  c->image_dims    = _image_dims;
  c->distance_of_coarray_elmts = distance_of_coarray_elmts;
  c->distance_of_image_elmts   = distance_of_image_elmts;
  *coarray         = c;

#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_malloc_do(*coarray, addr, _total_coarray_elmts*_elmt_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_malloc_do(*coarray, addr, _total_coarray_elmts*_elmt_size);
#endif
  
  free(_image_elmts);
  // Note: Do not free() _coarray_elmts.
}

void _XMP_coarray_malloc_do_f(void **coarray, void *addr)
{
  _XMP_coarray_malloc_do(coarray, addr);
}
