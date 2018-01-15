#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "xmp_internal.h"
int _XMP_flag_put_nb      = false; // This variable is temporal
int _XMP_flag_get_nb      = false; // This variable is temporal
#if defined(_XMP_FJRDMA)
int _XMP_flag_put_nb_rr   = false; // This variable is temporal
int _XMP_flag_put_nb_rr_i = false; // This variable is temporal
#endif

#if defined(_XMP_GASNET) || defined(_XMP_MPI3_ONESIDED)
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

void _XMP_initialize_onesided_functions()
{
#ifdef _XMP_FJRDMA
  if(_XMP_world_size > _XMP_ONESIDED_MAX_PROCS){
    if(_XMP_world_rank == 0)
      fprintf(stderr, "Warning : Onesided operations cannot be not used in %d processes (up to %d processes)\n", 
	      _XMP_world_size, _XMP_ONESIDED_MAX_PROCS);

    return;
  }
#endif

#ifdef _XMP_GASNET
  size_t _xmp_heap_size, _xmp_stride_size;
  _xmp_heap_size   = _get_size("XMP_ONESIDED_HEAP_SIZE");
  _xmp_stride_size = _get_size("XMP_ONESIDED_STRIDE_SIZE");
  _xmp_heap_size  += _xmp_stride_size;
  _XMP_gasnet_initialize(_xmp_heap_size, _xmp_stride_size);
  _XMP_gasnet_intrinsic_initialize();
#elif _XMP_FJRDMA
  _XMP_fjrdma_initialize();
#elif _XMP_MPI3_ONESIDED
  size_t _xmp_heap_size;
  _xmp_heap_size   = _get_size("XMP_ONESIDED_HEAP_SIZE");
  _XMP_mpi_onesided_initialize(_xmp_heap_size);
#endif

#ifdef _XMP_TCA
  _XMP_tca_initialize();
#endif

#if defined(_XMP_GASNET) || defined(_XMP_FJRDMA) || defined(_XMP_TCA) || defined(_XMP_MPI3_ONESIDED)
  _XMP_build_sync_images_table();
  _XMP_build_coarray_queue();
  _XMP_post_wait_initialize();
#endif

  /* Temporary  */
  if(getenv("XMP_PUT_NB") != NULL){
    _XMP_flag_put_nb = true;  // defalt false
    if(_XMP_world_rank == 0)
      printf("*** _XMP_coarray_contiguous_put() is Non-Blocking ***\n");
  }

  if(getenv("XMP_GET_NB") != NULL){
    _XMP_flag_get_nb = true;  // defalt false
    if(_XMP_world_rank == 0)
      printf("*** _XMP_coarray_contiguous_get() is Non-Blocking ***\n");
  }

#if defined(_XMP_FJRDMA)
  if(getenv("XMP_PUT_NB_RR") != NULL){
    _XMP_flag_put_nb    = true;  // defalt false
    _XMP_flag_put_nb_rr = true;  // defalt false
    if(_XMP_world_rank == 0)
      printf("*** _XMP_coarray_contiguous_put() is Non-Blocking and Round-Robin ***\n");
  }

  if(getenv("XMP_PUT_NB_RR_I") != NULL){
    _XMP_flag_put_nb      = true;  // defalt false
    _XMP_flag_put_nb_rr   = true;  // defalt false
    _XMP_flag_put_nb_rr_i = true;  // defalt false
    if(_XMP_world_rank == 0)
      printf("*** _XMP_coarray_contiguous_put() is Non-Blocking, Round-Robin and Immediately ***\n");
  }
#endif
  /* End Temporary */
}

void _XMP_finalize_onesided_functions()
{
#ifdef _XMP_GASNET
  _XMP_gasnet_finalize();
#elif _XMP_FJRDMA
  if(_XMP_world_size > _XMP_ONESIDED_MAX_PROCS) return;
  else _XMP_fjrdma_finalize();
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_onesided_finalize();
#endif

#ifdef _XMP_TCA
  _XMP_tca_finalize();
#endif
}
