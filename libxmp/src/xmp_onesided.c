#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "xmp_internal.h"

#ifdef _XMP_GASNET
static void _check_unit(char *env, char *env_val)
{
  int len = strlen(env_val);
  char last_char = env_val[len-1];
  if(last_char != 'M'){ // The last word must be "M"
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Unexpected Charactor in %s=%s\n", env, env_val);
      fprintf(stderr, "        The last Character must be M (e.g. %s=16M)\n", env);
    }
    _XMP_fatal_nomsg();
  }
}

static void _check_num(char *env, char *env_val)  // Is "env_val" all number except for the last word ?
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

static size_t _get_environment_val(char *env, char *env_val)
{
  _check_unit(env, env_val);
  _check_num(env, env_val);
  return (size_t)atoi(env_val) * 1024 * 1024;
}

static size_t _get_size(char *env)
{
  size_t size = 0;
  char *env_val;

  if((env_val = getenv(env)) != NULL){
    size = _get_environment_val(env, env_val);
  }
  else{
    if(strcmp(env, "XMP_ONESIDED_HEAP_SIZE") == 0){
      env_val = _XMP_DEFAULT_ONESIDED_HEAP_SIZE;
      size = _get_environment_val(env, env_val);
    }
    else if(strcmp(env, "XMP_ONESIDED_STRIDE_SIZE") == 0){
      env_val = _XMP_DEFAULT_ONESIDED_STRIDE_SIZE;
      size = _get_environment_val(env, env_val);
    }
    else
      _XMP_fatal("Internal Error");
  }

  return size;
}
#endif

void _XMP_onesided_initialize(int argc, char **argv)
{
#ifdef _XMP_FJRDMA
  if(_XMP_world_size > _XMP_FJRDMA_MAX_PROCS){
    if(_XMP_world_rank == 0)
      fprintf(stderr, "Warning : Onesided operations cannot be not used in %d processes (up to %d processes)\n", 
	      _XMP_world_size, _XMP_FJRDMA_MAX_PROCS);

    return;
  }
#endif

#ifdef _XMP_GASNET
  size_t _xmp_heap_size, _xmp_stride_size;
  _xmp_heap_size   = _get_size("XMP_ONESIDED_HEAP_SIZE");
  _xmp_stride_size = _get_size("XMP_ONESIDED_STRIDE_SIZE");
  _xmp_heap_size  += _xmp_stride_size;
  _XMP_gasnet_initialize(argc, argv, _xmp_heap_size, _xmp_stride_size);
#elif _XMP_FJRDMA
  _XMP_fjrdma_initialize(argc, argv);
#endif

#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA)
  _XMP_build_coarray_queue();
  _XMP_post_wait_initialize();
#endif
}

void _XMP_onesided_finalize(const int return_val)
{
#ifdef _XMP_GASNET
  _XMP_gasnet_finalize(return_val);
#elif _XMP_FJRDMA
  if(_XMP_world_size > _XMP_FJRDMA_MAX_PROCS) return;
  else _XMP_fjrdma_finalize();
#endif
}
