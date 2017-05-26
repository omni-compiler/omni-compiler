#ifndef MPI_PORTABLE_PLATFORM_H
#define MPI_PORTABLE_PLATFORM_H
#endif

#include <stdio.h>
#include <stdlib.h>
#include "xmp_internal.h"

void _XMP_bcast(void *data_addr, int count, int size,
		_XMP_object_ref_t *from_desc, _XMP_object_ref_t *on_desc);

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

void _XMP_bcast_acc(void *data_addr, int count, int size,
		    _XMP_object_ref_t *from_desc, _XMP_object_ref_t *on_desc)
{
  set_comm_mode();

#ifdef _XMP_TCA
  _XMP_fatal("XACC bcast is not implemented for TCA");

#else //default MPI
  if(comm_mode >= 1){
    _XMP_bcast(data_addr, count, size, from_desc, on_desc);
  }else{
    _XMP_fatal("XACC bcast is not implemented for comm_mode = 0");
  }
#endif
}
