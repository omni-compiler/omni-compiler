#include "xmp_internal.h"
#include "tca-api.h"

void _XMP_alloc_tca(_XMP_array_t *adesc)
{
  adesc->set_handle = _XMP_N_INT_FALSE;
  int array_dim = adesc->dim;
  for(int i=0;i<array_dim;i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;
    ai->reflect_acc_sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
  }

  adesc->wait_slot = 0;  // No change ?
  adesc->wait_tag  = 0x100;  // No change ?
}

