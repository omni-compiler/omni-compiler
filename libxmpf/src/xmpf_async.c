#include "xmpf_internal.h"

_Bool xmpf_test_task_on__(_XMP_object_ref_t **r_desc);
void xmpf_end_task__(void);

#define ASYNC_COMM 1

#ifdef ASYNC_COMM
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
    if (xmpf_test_task_on__(on_desc)){
      _XMP_wait_async__(*async_id);
      xmpf_end_task__();
    }
  }
  else {
    _XMP_wait_async__(*async_id);
  }

}


#ifdef ASYNC_COMM

void xmpf_init_async__(int *async_id){
  is_async = true;
  _async_id = *async_id;
}


void xmpf_start_async__(int *async_id){
  is_async = false;
}

#endif
