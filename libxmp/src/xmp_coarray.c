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
  _XMP_gasnet_initialize(argc, argv, _XMP_COARRAY_MALLOC_SIZE);
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

void _XMP_coarray_rma_SCALAR(int rma_code, void *coarray, int offset, void* local_addr, int node){
#ifdef _COARRAY_GASNET
	if(_XMP_N_COARRAY_PUT == rma_code){
		_XMP_gasnet_put(node, (_XMP_coarray_t*)coarray, offset, local_addr, 0, 1);
	} else if(_XMP_N_COARRAY_GET == rma_code){
		_XMP_gasnet_get(local_addr, 0, node, (_XMP_coarray_t*)coarray, offset, 1);
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
