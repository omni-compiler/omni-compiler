#include "xmpf_internal.h"

//_Bool xmpf_test_task_on__(_XMP_object_ref_t **r_desc);
void xmpf_create_task_nodes__(_XMP_nodes_t **n, _XMP_object_ref_t **r_desc);
_Bool xmpf_test_task_on_nodes__(_XMP_nodes_t **n);
void xmpf_end_task__(void);
void xmpf_nodes_dealloc__(_XMP_nodes_t **n_desc);

#ifdef _XMP_MPI3
extern _Bool is_async;
extern int _async_id;
#endif


/* void xmpf_wait_async__(int *async_id) */
/* { */
/*   _XMP_wait_async__(*async_id); */
/* } */




void xmpf_wait_async__(int *async_id, _XMP_object_ref_t **on_desc)
{
  if (*on_desc){
    /* if (xmpf_test_task_on__(on_desc)){ */
    /*   _XMP_wait_async__(*async_id); */
    /*   xmpf_end_task__(); */
    /* } */
    _XMP_nodes_t *n;
    xmpf_create_task_nodes__(&n, on_desc);
    if (xmpf_test_task_on_nodes__(&n)){
      _XMP_barrier_EXEC();
      xmpf_end_task__();
    }
    xmpf_nodes_dealloc__(&n);
  }
  else {
    _XMP_wait_async__(*async_id);
  }

}


#ifdef _XMP_MPI3

void xmpf_init_async__(int *async_id){
  is_async = true;
  _async_id = *async_id;
}


void xmpf_start_async__(int *async_id){
  is_async = false;
}

#endif
