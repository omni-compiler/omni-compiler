#include <stdarg.h>
#include "xmp_internal.h"

void _XMP_coarray_malloc(void **coarray, void *addr, long number_of_elements, size_t type_size) {

#ifdef _COARRAY_GASNET
  *coarray = (_XMP_coarray_t*)_XMP_alloc(sizeof(_XMP_coarray_t));
  _XMP_gasnet_set_coarray(*coarray, addr, number_of_elements, type_size);
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_initialize(int argc, char **argv){
#ifdef _COARRAY_GASNET
	char *env_heap_size;
	int heap_size;
	
	if((env_heap_size = getenv("XMP_COARRAY_HEAP_SIZE")) != NULL){
		int i;
		for(i=0;i<strlen(env_heap_size);i++){
			if(isdigit(env_heap_size[i]) == 0){
				fprintf(stderr, "%s : ", env_heap_size);
				_XMP_fatal("Unexpected Charactor in XMP_COARRAY_HEAP_SIZE");
			}
		}
		heap_size = atoi(env_heap_size) * 1024 * 1024;
		if(heap_size <= 0){
			_XMP_fatal("XMP_COARRAY_HEAP_SIZE is less than 0 !!");
		}
	} else{
		heap_size = _XMP_DEFAULT_COARRAY_HEAP_SIZE;
	}

  _XMP_gasnet_initialize(argc, argv, heap_size);
#else
	_XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_finalize(){
#ifdef _COARRAY_GASNET
  _XMP_gasnet_finalize(0);
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_rma_SCALAR(int rma_code, void *rma_addr, int rma_offset, void* local_addr, int local_offset, int node){
#ifdef _COARRAY_GASNET
	if(_XMP_N_COARRAY_PUT == rma_code){
		_XMP_gasnet_put(node, (_XMP_coarray_t*)rma_addr, rma_offset, local_addr, local_offset, 1);
	} else if(_XMP_N_COARRAY_GET == rma_code){
		_XMP_gasnet_get(local_addr, local_offset, node, (_XMP_coarray_t*)rma_addr, rma_offset, 1);
	}
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_put(int dest_node, void* dest, int dest_point, void *src_ptr, int src_point, int length){
#ifdef _COARRAY_GASNET
  _XMP_gasnet_put(dest_node, (_XMP_coarray_t*)dest, dest_point, src_ptr, src_point, length);
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_get(void *dest_ptr, int dest_point, int src_node, void* src, int src_point, int length){
#ifdef _COARRAY_GASNET
  _XMP_gasnet_get(dest_ptr, dest_point, src_node, (_XMP_coarray_t*)src, src_point, length);
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_sync_all(){
#ifdef _COARRAY_GASNET
  _XMP_gasnet_sync_all();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_sync_memory(){
#ifdef _COARRAY_GASNET
  _XMP_gasnet_sync_memory();
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}

void _XMP_coarray_rma_ARRAY(int rma_code, void *coarray, void *rma_addr, ...){
	int i;
  va_list args;
  va_start(args, rma_addr);

  // get coarray info
  int coarray_dim = va_arg(args, int);
  int coarray_start[coarray_dim], coarray_length[coarray_dim], coarray_stride[coarray_dim];
  unsigned long long coarray_dim_acc[coarray_dim];
  for (i = 0; i < coarray_dim; i++) {
    coarray_start[i] = va_arg(args, int);
    coarray_length[i] = va_arg(args, int);
    coarray_stride[i] = va_arg(args, int);
    coarray_dim_acc[i] = va_arg(args, unsigned long long);
  }

  // get rma_array info
  int rma_array_dim = va_arg(args, int);
  int rma_array_start[rma_array_dim], rma_array_length[rma_array_dim], rma_array_stride[rma_array_dim];
  unsigned long long rma_array_dim_acc[rma_array_dim];
  for (i = 0; i < rma_array_dim; i++) {
    rma_array_start[i] = va_arg(args, int);
    rma_array_length[i] = va_arg(args, int);
    rma_array_stride[i] = va_arg(args, int);
    rma_array_dim_acc[i] = va_arg(args, unsigned long long);
  }

  // get coarray ref info
	//	_XMP_nodes_t *coarray_nodes = coarray->nodes;   // Fix me
	//	int coarray_nodes_dim = coarray_nodes->dim;
	int coarray_nodes_dim = 1;
	int coarray_nodes_ref[coarray_nodes_dim];
  for (i = 0; i < coarray_nodes_dim; i++) {
    // translate 1-origin to 0-rigin
		//    coarray_nodes_ref[i] = va_arg(args, int) - 1;
		coarray_nodes_ref[i] = va_arg(args, int);
  }

  va_end(args);

#ifdef _COARRAY_GASNET
	int rma_array_start_point = 0, coarray_start_point = 0;
	for(i=0;i<rma_array_dim;i++)
		rma_array_start_point += rma_array_start[i] * rma_array_dim_acc[i];
	for(i=0;i<coarray_dim;i++)
		coarray_start_point   += coarray_start[i] * coarray_dim_acc[i];

  if(_XMP_N_COARRAY_PUT == rma_code){
		_XMP_gasnet_put(coarray_nodes_ref[0], (_XMP_coarray_t*)coarray, coarray_start_point, 
										rma_addr, rma_array_start_point, coarray_length[coarray_dim-1]);
  } else if(_XMP_N_COARRAY_GET == rma_code){
		_XMP_gasnet_get(rma_addr, rma_array_start_point, coarray_nodes_ref[0], 
										(_XMP_coarray_t*)coarray, coarray_start_point, coarray_length[coarray_dim-1]);
  } else{
		_XMP_fatal("Unexpected Operation !!");
	}
#else
  _XMP_fatal("Cannt use Coarray Function");
#endif
}
