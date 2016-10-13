#include "xmp_internal.h"
extern void _XMP_gasnet_build_shift_queue();
size_t _xmp_gasnet_heap_size, _xmp_gasnet_stride_size, _xmp_gasnet_coarray_shift = 0;
char **_xmp_gasnet_buf;
int *_xmp_gasnet_stride_queue;

gasnet_handlerentry_t htable[] = {
  { _XMP_GASNET_LOCK_REQUEST,               _xmp_gasnet_lock_request },
  { _XMP_GASNET_SETLOCKSTATE,               _xmp_gasnet_setlockstate },
  { _XMP_GASNET_UNLOCK_REQUEST,             _xmp_gasnet_unlock_request },
  { _XMP_GASNET_LOCKHANDOFF,                _xmp_gasnet_lockhandoff },
  { _XMP_GASNET_POSTREQ,                    _xmp_gasnet_postreq },
  { _XMP_GASNET_UNPACK,                     _xmp_gasnet_unpack },
  { _XMP_GASNET_UNPACK_USING_BUF,           _xmp_gasnet_unpack_using_buf },
  { _XMP_GASNET_UNPACK_REPLY,               _xmp_gasnet_unpack_reply },
  { _XMP_GASNET_PACK,                       _xmp_gasnet_pack },
  { _XMP_GASNET_UNPACK_GET_REPLY,           _xmp_gasnet_unpack_get_reply},
  { _XMP_GASNET_PACK_USING_BUF,             _xmp_gasnet_pack_using_buf},
  { _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF, _xmp_gasnet_unpack_get_reply_using_buf},
  { _XMP_GASNET_PACK_GET_HANDLER,           _xmp_gasnet_pack_get },
  { _XMP_GASNET_UNPACK_GET_REPLY_NONC,      _xmp_gasnet_unpack_get_reply_nonc },
  { _XMP_GASNET_ADD_NOTIFY,                 _xmp_gasnet_add_notify },
  { _XMP_GASNET_ATOMIC_DEFINE_DO,           _xmp_gasnet_atomic_define_do },
  { _XMP_GASNET_ATOMIC_DEFINE_REPLY_DO,     _xmp_gasnet_atomic_define_reply_do },
  { _XMP_GASNET_ATOMIC_REF_DO,              _xmp_gasnet_atomic_ref_do },
  { _XMP_GASNET_ATOMIC_REF_REPLY_DO,        _xmp_gasnet_atomic_ref_reply_do }
};

/**
   Initialize GASNet job
*/
void _XMP_gasnet_initialize(int argc, char **argv, const size_t xmp_gasnet_heap_size, 
			    const size_t xmp_gasnet_stride_size)
{
  if(argc != 0)
    gasnet_init(&argc, &argv);
  else{
    // In XMP/Fortran, this function is called with "argc == 0" & "**argv == NULL".
    // But if the second argument of gasnet_init() is NULL, gasnet_init() returns error.
    // So dummy argument is created and used.
    char **s;
    s = malloc(sizeof(char *));
    s[0] = malloc(sizeof(char));
    gasnet_init(&argc, &s);
  }

  _XMP_world_rank = gasnet_mynode();
  _XMP_world_size = gasnet_nodes();

  if(xmp_gasnet_heap_size % GASNET_PAGESIZE != 0){
    if(xmp_gasnet_heap_size <= GASNET_PAGESIZE){
      _xmp_gasnet_heap_size = GASNET_PAGESIZE;
    }
    else{
      _xmp_gasnet_heap_size = (xmp_gasnet_heap_size/GASNET_PAGESIZE + 1) * GASNET_PAGESIZE;
    }
  }
  else{
    _xmp_gasnet_heap_size = xmp_gasnet_heap_size;
  }

  _xmp_gasnet_stride_size = xmp_gasnet_stride_size;

  gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), (uintptr_t)_xmp_gasnet_heap_size, 0);

  _xmp_gasnet_buf = (char **)malloc(sizeof(char*) * _XMP_world_size);

  gasnet_node_t i;
  gasnet_seginfo_t *s = (gasnet_seginfo_t *)malloc(_XMP_world_size*sizeof(gasnet_seginfo_t));
  gasnet_getSegmentInfo(s, _XMP_world_size);
  for(i=0;i<_XMP_world_size;i++)
    _xmp_gasnet_buf[i] =  (char*)s[i].addr;

  _xmp_gasnet_coarray_shift = xmp_gasnet_stride_size;
  _xmp_gasnet_stride_queue = malloc(sizeof(int) * _XMP_GASNET_STRIDE_INIT_SIZE);

  _XMP_gasnet_build_shift_queue();
}

/**
   Finalize GASNet job
*/
void _XMP_gasnet_finalize(const int val)
{
  _XMP_gasnet_sync_all();
  gasnet_exit(val);
}

