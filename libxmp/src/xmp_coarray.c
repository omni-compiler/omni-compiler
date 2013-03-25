#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"
#ifdef _XMP_COARRAY_FJRDMA
#include "mpi-ext.h"
#endif

unsigned long long _xmp_heap_size;
static int _elmt_size, _coarray_dims, _image_dims, *_image_size, _array_dims;
static long long *_coarray_size, _total_coarray_size;
static long long _total_coarray_length, _total_array_length;
static _XMP_array_section_t *_coarray, *_array;

void _XMP_coarray_initialize(int argc, char **argv){
#ifdef _XMP_COARRAY_GASNET
  char *env_heap_size;

  if((env_heap_size = getenv("XMP_COARRAY_HEAP_SIZE")) != NULL){
    int i;
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
  } else{
    _xmp_heap_size = _XMP_DEFAULT_COARRAY_HEAP_SIZE;
  }

  _XMP_gasnet_initialize(argc, argv, _xmp_heap_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_initialize();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_finalize(){
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
  _XMP_gasnet_finalize(0);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_finalize();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_malloc_set(int elmt_size, int coarray_dims, int image_dims){
  _elmt_size          = elmt_size;
  _coarray_dims       = coarray_dims;
  _coarray_size       = _XMP_alloc(sizeof(long long) * coarray_dims);
  _image_dims         = image_dims;
  _image_size         = _XMP_alloc(sizeof(int) * image_dims);
  _total_coarray_size = 1;
}

void _XMP_coarray_malloc_set_f(int *elmt_size, int *coarray_dims, int *image_dims){
  _XMP_coarray_malloc_set(*elmt_size, *coarray_dims, *image_dims);
}

void _XMP_coarray_malloc_array_info(int dim, long long coarray_size){
  _coarray_size[dim] = coarray_size;
  _total_coarray_size *= coarray_size;
}

void _XMP_coarray_malloc_array_info_f(int *dim, long long *coarray_size){
  _XMP_coarray_malloc_array_info(*dim, *coarray_size);
}

void _XMP_coarray_malloc_image_info(int dim, int image_size){
  _image_size[dim] = image_size;
}

void _XMP_coarray_malloc_image_info_f(int *dim, int *image_size){
  _XMP_coarray_malloc_image_info(*dim, *image_size);
}

void _XMP_coarray_malloc_do(void **coarray, void *addr){
  int i, j;

  long long *distance_of_coarray_elmt = _XMP_alloc(sizeof(long long) * _coarray_dims);
  for(i=0;i<_coarray_dims-1;i++){
    long long distance = 1;
    for(j=i+1;j<_coarray_dims;j++){
      distance       *= _coarray_size[j];
    }
    distance_of_coarray_elmt[i] = distance * _elmt_size;
  }
  distance_of_coarray_elmt[_coarray_dims-1] = _elmt_size;

  int total_node_size = _XMP_get_execution_nodes()->comm_size;
  int total_image_size = 1;
  for(i=0;i<_image_dims-1;i++)
    total_image_size *= _image_size[i];

#if 0
  fprintf(stderr, "total_node_size=%d, total_image_size=%d\n", total_node_size, total_image_size);
#endif
  if(total_image_size > total_node_size){
    _XMP_fatal("Too large coarray image size.");
  }
  else if(total_node_size % total_image_size != 0){
    _XMP_fatal("Wrong coarray image size.");
  }
  _image_size[_image_dims-1] = total_node_size / total_image_size;

  int *distance_of_image_elmt = _XMP_alloc(sizeof(int) * _image_dims);
  for(i=_image_dims-1;i>=1;i--){
    int distance = 1;  
    for(j=0;j<i;j++){
      distance *= _image_size[j]; 
    }
    distance_of_image_elmt[i] = distance;
  }
  distance_of_image_elmt[0] = 1; 

  _XMP_coarray_t* c = _XMP_alloc(sizeof(_XMP_coarray_t));
  c->elmt_size = _elmt_size;
  c->coarray_dims = _coarray_dims;
  c->size         = _coarray_size;
  c->distance_of_array_elmt = distance_of_coarray_elmt;
  c->image_dims = _image_dims;
  c->distance_of_image_elmt = distance_of_image_elmt;
  *coarray = c;
#if 0
  fprintf(stderr, "elmt_size=%d coarray_dims=%d, image_dims=%d\n", ((_XMP_coarray_t*)(*coarray))->elmt_size, ((_XMP_coarray_t*)(*coarray))->coarray_dims, ((_XMP_coarray_t*)(*coarray))->image_dims);
  fprintf(stderr, "corray_size");
  for (i=0; i<((_XMP_coarray_t*)(*coarray))->coarray_dims; i++) {
    fprintf(stderr, "[%d]=%d  ", i, ((_XMP_coarray_t*)(*coarray))->size[i]);
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "distance_of_image_elmt");
  for (i=0; i<((_XMP_coarray_t*)(*coarray))->image_dims; i++) {
    fprintf(stderr, "[%d]=%d  ", i, ((_XMP_coarray_t*)(*coarray))->distance_of_image_elmt[i]);
  }
  fprintf(stderr, "\nchech fin\n");
#endif

#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_set_coarray(*coarray, addr, _total_coarray_size, _elmt_size);
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_reg_mem(*coarray, addr, _total_coarray_size);
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
  free(_image_size);
}

void _XMP_coarray_malloc_do_f(void **coarray, void *addr){
  _XMP_coarray_malloc_do(coarray, addr);
}

void _XMP_coarray_rma_set(int coarray_dims, int array_dims, int image_dims){
  _coarray              = malloc(sizeof(_XMP_array_section_t) * coarray_dims);
  _array                = malloc(sizeof(_XMP_array_section_t) * array_dims);
  _coarray_dims         = coarray_dims;
  _array_dims           = array_dims;
  _image_size           = malloc(sizeof(int) * image_dims);
  _image_dims           = image_dims;
  _total_coarray_length = 1;
  _total_array_length   = 1;
}

void _XMP_coarray_rma_set_f(int *coarray_dims, int *array_dims, int *image_dims){
  _XMP_coarray_rma_set(*coarray_dims, *array_dims, *image_dims);
}

void _XMP_coarray_rma_coarray_set(int dim, long long start, long long length, long long stride){
  _coarray[dim].start    = start;
  _coarray[dim].length   = length;
  _total_coarray_length *= length;
  _coarray[dim].stride   = stride;
}

void _XMP_coarray_rma_coarray_set_f(int *dim, long long *start, long long *length, long long *stride){
  _XMP_coarray_rma_coarray_set(*dim, *start, *length, *stride);
}

void _XMP_coarray_rma_array_set(int dim, long long start, long long length, long long stride, long long size, long long distance){
  _array[dim].start    = start;
  _array[dim].length   = length;
  _total_array_length *= length;
  _array[dim].stride   = stride;
  _array[dim].size     = size;
  _array[dim].distance = distance;
}

void _XMP_coarray_rma_array_set_f(int *dim, long long *start, long long *length, long long *stride, long long *size, long long *distance){
  _XMP_coarray_rma_array_set(*dim, *start, *length, *stride, *size, *distance);
}

void _XMP_coarray_rma_node_set(int dim, int image_num){
  _image_size[dim]  = image_num;
}

void _XMP_coarray_rma_node_set_f(int *dim, int *image_num){
  _XMP_coarray_rma_node_set(*dim, *image_num);
}

static int check_continuous(_XMP_array_section_t *a, int dims, long long total_length){
  // If only 1 elements is transferred.
  if(_total_coarray_length == 1)
    return _XMP_N_INT_TRUE;

  // Only the last dimension length is transferred.
  // ex) a[1][2][:]
  if(_total_coarray_length == (a+dims-1)->length && (a+dims-1)->stride == 1)
    return _XMP_N_INT_TRUE;

  // The last dimension is not continuous ?
  if((a+dims-1)->length > 1 && (a+dims-1)->stride != 1){
    return _XMP_N_INT_FALSE;
  }
  else{
    int i, flag = _XMP_N_INT_TRUE, th = 0;
    // (i+1, i+2, ..)-th dimensions are ":" && i-th dimension's stride is "1" &&
    // (i-1, i-2, ..)-th dimension's length is "1" ?
    // ex1) a[1][3][1:2][:]   // (i = 2)
    // ex2) a[2][:][:]        // (i = 0)
    for(i=dims-1;i>=0;i--){
      if((a+i)->start != 0 || (a+i)->length != (a+i)->size){
	th = i;
	break;
      }
    }
    
    if(th == 0 && a->stride == 1){  //  ex) a[1:2][:][:] or a[:][:][:]
      return _XMP_N_INT_TRUE;
    }
    else if(th == dims-1){          // The last dimension is not ":".  ex) a[:][:][1:2]
      return _XMP_N_INT_FALSE;
    }
    else if((a+th)->stride == 1){
      for(i=0;i<th;i++)
	if((a+i)->length != 1)
	  flag = _XMP_N_INT_FALSE;
    }
    else{
      flag = _XMP_N_INT_FALSE;
    }
    
    if(flag){
      return _XMP_N_INT_TRUE;
    }
    else{
      return _XMP_N_INT_FALSE; 
    }
  }
}

void _XMP_coarray_rma_do(int rma_code, void *coarray, void *array){
  int i, target_image = 0;

  if(_total_coarray_length != _total_array_length){
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n") ;
  }

  for(i=0;i<_image_dims-1;i++)
    target_image += ((_XMP_coarray_t*)coarray)->distance_of_image_elmt[i] * (_image_size[i] - 1);
  target_image += _image_size[_image_dims-1] - 1;

  for(i=0;i<_array_dims;i++)
    _array[i].distance *= ((_XMP_coarray_t*)coarray)->elmt_size;

  for(i=0;i<_coarray_dims;i++){
    _coarray[i].size     = ((_XMP_coarray_t*)coarray)->size[i];
    _coarray[i].distance = ((_XMP_coarray_t*)coarray)->distance_of_array_elmt[i];
  }

  int coarray_continuous, array_continuous;
  coarray_continuous = check_continuous(_coarray, _coarray_dims, _total_coarray_length);
  array_continuous   = check_continuous(_array, _array_dims, _total_coarray_length); 

#if 0
  fprintf(stderr, "===coarray_rma_do===\n");
  fprintf(stderr, "target_image=%d\n", target_image);
  fprintf(stderr, "coarray_continuous=%d  array_continuous=%d\n", coarray_continuous, array_continuous);
#endif

  if(coarray_continuous == _XMP_N_INT_FALSE || coarray_continuous == _XMP_N_INT_FALSE){
    _XMP_fatal("Sorry! Not continuous array is not supported.");
  }

#ifdef _XMP_COARRAY_GASNET
  if(_XMP_N_COARRAY_PUT == rma_code){
    _XMP_gasnet_put(target_image, coarray_continuous, array_continuous,
		    _coarray_dims, _array_dims, _coarray, _array, coarray, array, _total_coarray_length);
  }
  else if(_XMP_N_COARRAY_GET == rma_code){
    _XMP_gasnet_get(target_image, coarray_continuous, array_continuous,
                    _coarray_dims, _array_dims, _coarray, _array, coarray, array, _total_coarray_length);
  }
  else{
    _XMP_fatal("Unexpected Operation !!");
  }
#elif _XMP_COARRAY_FJRDMA
  if (_XMP_N_COARRAY_PUT == rma_code) {
    _XMP_fjrdma_put(target_image,
		    coarray_continuous,
		    array_continuous,
		    _coarray_dims,
		    _array_dims,
		    _coarray,
		    _array,
		    coarray,
		    array,
		    _total_coarray_length,
		    _total_array_length,
		    _image_size);
  } else if (_XMP_N_COARRAY_GET == rma_code) {
    _XMP_fjrdma_get(target_image,
		    coarray_continuous,
		    array_continuous,
                    _coarray_dims,
		    _array_dims,
		    _coarray,
		    _array,
		    coarray,
		    array,
		    _total_coarray_length,
		    _total_array_length,
		    _image_size);
  } else {
    _XMP_fatal("Unexpected Operation !!");
  }
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif

  free(_coarray);  free(_array);  free(_image_size);
}

void _XMP_coarray_rma_do_f(int *rma_code, void *coarray, void *array){
  _XMP_coarray_rma_do(*rma_code, coarray, array);
}

void _XMP_coarray_sync_all(){
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_sync_memory(){
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void xmp_sync_memory(int* status){
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_memory();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void xmp_sync_all(int* status){
#ifdef _XMP_COARRAY_GASNET
  _XMP_gasnet_sync_all();
#elif _XMP_COARRAY_FJRDMA
  _XMP_fjrdma_sync_all();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void xmp_sync_image(int image, int* status){
  _XMP_fatal("Not implement xmp_sync_images()");
}

void xmp_sync_image_f(int *image, int* status){
  xmp_sync_image(*image, status);
}

void xmp_sync_images(int num, int* image_set, int* status){
  _XMP_fatal("Not implement xmp_sync_images_images()");
}

void xmp_sync_images_f(int *num, int* image_set, int* status){
  xmp_sync_images(*num, image_set, status);
}

void xmp_sync_images_all(int* status){
  _XMP_fatal("Not implement xmp_sync_images_all()");
}

