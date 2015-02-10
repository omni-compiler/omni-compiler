#include "xmp_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tca-api.h"

int useTCAHybrid;
int useTCAHybridFlag;

void _XMP_init_tca()
{
  useTCAHybridFlag = 0;
  if (!useTCAHybridFlag) {
    char *mode_str = getenv("USE_TCA_HYBRID");
    if(mode_str !=  NULL){
      int mode = atoi(mode_str);
      switch(mode){
      default:
      case 0:
	useTCAHybrid = 0;
	break;
      case 1:
	useTCAHybrid = 1;
	break;
      }
    }
    useTCAHybridFlag = 1;
  }

  if (!useTCAHybridFlag) {
    if(_XMP_world_size > 16)
      _XMP_fatal("TCA reflect has been not implemented in 16 more than nodes.");
  }

  TCA_SAFE_CALL(tcaInit());
  tcaDMADescInt_Init(); // Initialize Descriptor (Internal Memory) Mode
}

void _XMP_alloc_tca(_XMP_array_t *adesc)
{
  adesc->set_handle = _XMP_N_INT_FALSE;
  int array_dim = adesc->dim;
  for(int i = 0; i < array_dim; i++){
    _XMP_array_info_t *ai = &(adesc->info[i]);
    if(ai->shadow_type == _XMP_N_SHADOW_NONE)
      continue;
    ai->reflect_acc_sched = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
  }

  adesc->wait_slot = 0;  // No change ?
  adesc->wait_tag  = 0x100;  // No change ?
}

void _xmp_set_network_id (_XMP_array_t *adesc)
{
  int my_rank = adesc->align_template->onto_nodes->comm_rank;
  int my_id;
  int *network_id = _XMP_alloc(sizeof(int) * _XMP_world_size);
  
  // temporary, FIX me
  if ((my_rank % 2) == 0) {
    my_id = 0;
  } else {
    my_id = 1;
  }

  MPI_Allgather(&my_id, 1, MPI_INT, network_id, 1, MPI_INT, MPI_COMM_WORLD);

  /* if (my_rank == 0) */
  /*   for (int i = 0; i < 8; i++)  */
  /*     printf("%d\n", network_id[i]); */

  adesc->network_id = network_id;
}

int _xmp_is_same_network_id(_XMP_array_t *adesc, int dst_rank)
{
  int my_rank = adesc->align_template->onto_nodes->comm_rank;
  int src_id = adesc->network_id[my_rank];
  int dst_id = adesc->network_id[dst_rank];

  assert(dst_rank >= 0);

  assert(useTCAHybridFlag == 1);
  if (useTCAHybrid) {
    if (src_id == dst_id) {
      return 1;
    } else {
      return 0;
    }
  } else {
    return 1;
  }
}
