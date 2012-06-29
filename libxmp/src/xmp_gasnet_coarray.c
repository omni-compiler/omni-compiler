#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include <gasnet.h>
//#include "xmp_gasnet_lock.h"

//gasnet_handlerentry_t htable[] = {
//  { XMP_GASNET_LOCK_REQUEST,   _xmp_gasnet_lock_request },
//  { XMP_GASNET_SETLOCKSTATE,   _xmp_gasnet_setlockstate },
//  { XMP_GASNET_UNLOCK_REQUEST, _xmp_gasnet_unlock_request },
//  { XMP_GASNET_LOCKHANDOFF,    _xmp_gasnet_lockhandoff }
//};
static gasnet_seginfo_t *_test_seginfo;
static unsigned long long _xmp_coarray_shift = 0;
static void **_xmp_gasnet_buf;

void _XMP_gasnet_set_coarray(_XMP_coarray_t *coarray, void **addr, unsigned long long number_of_elements, size_t type_size){
  int numprocs;
  void **each_addr;   // head address of a local array on each node

  numprocs = gasnet_nodes();
  each_addr = _XMP_alloc(sizeof(void*) * numprocs);

  gasnet_node_t i;
  for(i=0;i<numprocs;i++)
    each_addr[i] = (char *)(_xmp_gasnet_buf[i]) + _xmp_coarray_shift;

  _xmp_coarray_shift += type_size * number_of_elements;

  coarray->addr = (char **)each_addr;
  coarray->type_size = type_size;

  *addr = each_addr[gasnet_mynode()];
}

static void *_gasnet_getseg(gasnet_node_t node) {
  gasnet_seginfo_t *s = (gasnet_seginfo_t *)malloc(gasnet_nodes()*sizeof(gasnet_seginfo_t));
  gasnet_getSegmentInfo(s, gasnet_nodes());
  _test_seginfo = s;

  return _test_seginfo[node].addr;
}

void _XMP_gasnet_initialize(int argc, char **argv, unsigned long long malloc_size){
  int numprocs;

  gasnet_init(&argc, &argv);

  if(malloc_size % GASNET_PAGESIZE != 0)
    malloc_size = (malloc_size/GASNET_PAGESIZE -1) * GASNET_PAGESIZE;

  //gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), malloc_size, 0);
  gasnet_attach(NULL, 0, malloc_size, 0);
  numprocs = gasnet_nodes();

  _xmp_gasnet_buf = (void **)malloc(sizeof(void*) * numprocs);

  gasnet_node_t i;
  for(i=0;i<numprocs;i++)
    _xmp_gasnet_buf[i] = (void*)_gasnet_getseg( i );

}

void _XMP_gasnet_finalize(int val){
  gasnet_exit(val);
}

void _XMP_gasnet_sync_memory(){
  gasnet_wait_syncnbi_all();
}

void _XMP_gasnet_sync_all(){
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

void _XMP_gasnet_put(int dest_node, _XMP_coarray_t* dest, unsigned long long dest_point, void *src_ptr, 
										 unsigned long long src_point, unsigned long long length){
  dest_point *= dest->type_size;
  src_point  *= dest->type_size;
  gasnet_put_nbi_bulk(dest_node, dest->addr[dest_node]+dest_point, (char *)(src_ptr)+src_point, dest->type_size*length);
}

void _XMP_gasnet_get(void *dest_ptr, unsigned long long dest_point, int src_node, _XMP_coarray_t *src, 
										 unsigned long long src_point, unsigned long long length){
  dest_point *= src->type_size;
  src_point  *= src->type_size;
  gasnet_get_bulk((char *)(dest_ptr)+dest_point, src_node, src->addr[src_node]+src_point, src->type_size*length);
}
