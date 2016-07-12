#include "xmpf_internal.h"

void xmpf_reduction__(void *data_addr, int *count, int *datatype, int *op,
		      _XMP_object_ref_t **r_desc, int *num_locs);

static char comm_mode = -1;

static void set_comm_mode()
{
  if(comm_mode < 0){
    char *mode_str = getenv("XACC_COMM_MODE");
    if(mode_str !=  NULL){
      comm_mode = atoi(mode_str);
    }else{
      comm_mode = 0;
    }
  }
}

void xaccf_reduction__(void *data_addr, int *count, int *datatype, int *op,
		       _XMP_object_ref_t **r_desc, int *num_locs)
{
  set_comm_mode();
  
  if(*num_locs != 0){
    _XMP_fatal("reduce_FLMM is not implemented for XACC");
  }

  if(comm_mode >= 1){
    xmpf_reduction__(data_addr, count, datatype, op, r_desc, num_locs);
  }else{
    _XMP_fatal("reduce is not implemented for XACC_COMM_MODE=0");
  }
}

void xaccf_reduction_loc__(int *dim, void *loc, int *datatype)
{
  _XMP_fatal("reduction_loc is not implemented for XACC");
}


void xaccf_bcast__(void *data_addr, int *count, int *datatype,
		   _XMP_object_ref_t **from_desc, _XMP_object_ref_t **on_desc)
{
  _XMP_fatal("bcast is not implemented for XACC");
}
