#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "xmp_internal.h"
#include "xmp_constant.h"

static size_t _xmp_heap_size, _xmp_stride_size;
static size_t _elmt_size;
static int _coarray_dims, _image_dims, *_image_elmts;
static int *_coarray_elmts, _total_coarray_elmts;

// The last word must be "M"
static void check_last_word(char *env, char *env_val)
{
  int len = strlen(env_val);
  char last_char = env_val[len-1];
  if(last_char != 'M'){
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Unexpected Charactor in %s=%s\n", env, env_val);
      fprintf(stderr, "        The last Character must be M (e.g. %s=16M)\n", env);
    }
    _XMP_fatal_nomsg();
  }
}

// Is "env_val" all number except for the last word ?
static void check_num(char *env, char *env_val)
{
  int len = strlen(env_val);

  for(int i=0;i<len-1;i++){
    if(! isdigit(env_val[i])){
      if(_XMP_world_rank == 0)
	fprintf(stderr, "[ERROR] Unexpected Charactor in %s=%s\n", env, env_val);
      _XMP_fatal_nomsg();
    }
  }
}

static size_t get_coarray_memory_size(char *env, char *env_val){
  check_last_word(env, env_val);
  check_num(env, env_val);
  return (size_t)atoi(env_val) * 1024 * 1024;
}

static size_t check_env_size_coarray(char *env){
  size_t size = 0;
  char *env_val;

  if((env_val = getenv(env)) != NULL){
    size = get_coarray_memory_size(env, env_val);
  }
  else{
    if(strcmp(env, "XMP_COARRAY_HEAP_SIZE") == 0){
      env_val = _XMP_DEFAULT_COARRAY_HEAP_SIZE;
      size = get_coarray_memory_size(env, env_val);
    }
    else if(strcmp(env, "XMP_COARRAY_STRIDE_SIZE") == 0){
      env_val = _XMP_DEFAULT_COARRAY_STRIDE_SIZE;
      size = get_coarray_memory_size(env, env_val);
    }
    else
      _XMP_fatal("Internal Error in xmp_coarray_set.c");
  }

  if(size <= 0){
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Unexpected value of %s=%s\n", env, env_val);
    }
    _XMP_fatal_nomsg();
  }

  return size;
}

void _XMP_coarray_initialize(int argc, char **argv)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &_XMP_world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_XMP_world_size);

  _xmp_heap_size   = check_env_size_coarray("XMP_COARRAY_HEAP_SIZE");
  _xmp_stride_size = check_env_size_coarray("XMP_COARRAY_STRIDE_SIZE");
  _xmp_heap_size  += _xmp_stride_size;

#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_initialize(argc, argv, _xmp_heap_size, _xmp_stride_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_initialize();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_finalize(const int return_val)
{
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_finalize(return_val);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_finalize();
#endif
}

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

/********************/
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
  *coarray          = c;

#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_malloc_do(*coarray, addr, (size_t)_total_coarray_elmts*_elmt_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_malloc_do(*coarray, addr, (size_t)_total_coarray_elmts*_elmt_size);
#endif
  
  free(_image_elmts);
  // Note: Do not free() _coarray_elmts.
}

void _XMP_coarray_malloc_do_f(void **coarray, void *addr)
{
  _XMP_coarray_malloc_do(coarray, addr);
}

