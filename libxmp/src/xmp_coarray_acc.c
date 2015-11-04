//#include <stdarg.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <ctype.h>
#include "xmp_internal.h"
#include "xmp_constant.h"

int _XMP_coarray_get_total_elmts(void *coarray_desc)
{
  _XMP_coarray_t* c = (_XMP_coarray_t*)coarray_desc;
    
  int total_coarray_elmts = 1;
  for(int i = 0; i < c->coarray_dims; i++){
    total_coarray_elmts *= c->coarray_elmts[i];
  }
  return total_coarray_elmts;
}

/** 
   Attach memory to coarray
 */
void _XMP_coarray_attach_acc(_XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size)
{
  _XMP_coarray_set_info(coarray_desc);

#ifdef _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_attach(coarray_desc, addr, coarray_size, true);
#else
  _XMP_fatal("_XMP_coarray_attach_acc is unavailable");
#endif
}

/** 
   Detach memory from coarray
 */
void _XMP_coarray_detach_acc(_XMP_coarray_t *coarray_desc)
{
#ifdef _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_detach(coarray_desc, true);
#else
  _XMP_fatal("_XMP_coarray_detach_acc is unavailable");
#endif
}

void _XMP_coarray_malloc_do_acc(void **coarray_desc, void *addr)
{
  _XMP_coarray_t* c = *coarray_desc;
    
  int total_coarray_elmts = _XMP_coarray_get_total_elmts(c);

#ifdef _XMP_TCA
  _XMP_tca_malloc_do(*coarray_desc, addr, total_coarray_elmts * c->elmt_size);
#elif _XMP_MPI3_ONESIDED
  _XMP_mpi_coarray_malloc_do(*coarray_desc, addr, total_coarray_elmts * c->elmt_size, true);
#else
  _XMP_fatal("_XMP_coarray_malloc_do_acc is unavailable");
#endif
}


/************************************************************************/
/* DESCRIPTION : Execute put operation without preprocessing            */
/* ARGUMENT    : [IN] target_image : Target image                       */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] is_dst_on_acc: Whether dst is on acc or not       */
/*               [IN] is_src_on_acc: Whether src is on acc or not       */
/* NOTE       : Both dst and src are continuous coarrays                */
/* EXAMPLE    :                                                         */
/*     a[0:100]:[1] = b[0:100]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_coarray_shortcut_put_acc(const int target_image, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc, 
				   const size_t dst_offset, const size_t src_offset, 
				   const size_t dst_elmts, const size_t src_elmts,
  				   const int is_dst_on_acc, const int is_src_on_acc)
{
  int target_rank = target_image - 1;
  size_t elmt_size = dst_desc->elmt_size;

  if(!is_dst_on_acc || !is_src_on_acc){
    _XMP_fatal("device to host and host to device put are umimplemented");
  }
  
  if(target_rank == _XMP_world_rank){
#ifdef _XMP_MPI3_ONESIDED
    _XMP_mpi_shortcut_put(target_rank, dst_desc, src_desc, dst_offset, src_offset,
			  dst_elmts, src_elmts, elmt_size, is_dst_on_acc, is_src_on_acc);
#else
    _XMP_fatal("local_continuous_copy is unimplemented");
#endif
    //_XMP_local_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src_desc->real_addr+src_offset, 
    //dst_elmts, src_elmts, elmt_size);
  }
  else{
#ifdef _XMP_TCA
    _XMP_tca_shortcut_put(target_rank, dst_offset, src_offset, dst_desc, src_desc, 
			  dst_elmts, src_elmts, elmt_size);
#elif _XMP_MPI3_ONESIDED
    _XMP_mpi_shortcut_put(target_rank, dst_desc, src_desc, dst_offset, src_offset,
			  dst_elmts, src_elmts, elmt_size, is_dst_on_acc, is_src_on_acc);
#else
    _XMP_fatal("_XMP_coarray_shortcut_put_acc is unavailable");
#endif
  }
}

/************************************************************************/
/* DESCRIPTION : Execute get operation without preprocessing            */
/* ARGUMENT    : [IN] target_image : Target image                       */
/*               [OUT] *dst_desc   : Descriptor of destination coarray  */
/*               [IN] *src_desc    : Descriptor of source coarray       */
/*               [IN] dst_offset   : Offset size of destination coarray */
/*               [IN] src_offset   : Offset size of source coarray      */
/*               [IN] dst_elmts    : Number of elements of destination  */
/*               [IN] src_elmts    : Number of elements of source       */
/*               [IN] is_dst_on_acc: Whether dst is on acc or not       */
/*               [IN] is_src_on_acc: Whether src is on acc or not       */
/* NOTE       : Both dst and src are continuous coarrays                */
/* EXAMPLE    :                                                         */
/*     a[0:100] = b[0:100]:[1]; // a[] is a dst, b[] is a src           */
/************************************************************************/
void _XMP_coarray_shortcut_get_acc(const int target_image, _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				   const size_t dst_offset, const size_t src_offset, 
				   const size_t dst_elmts, const size_t src_elmts,
				   const int is_dst_on_acc, const int is_src_on_acc)
{
  int target_rank = target_image - 1;
  size_t elmt_size = dst_desc->elmt_size;

  if(!is_dst_on_acc || !is_src_on_acc){
    _XMP_fatal("device to host and host to device put are umimplemented");
  }

  if(target_rank == _XMP_world_rank){
    _XMP_fatal("local_continuous_copy is unimplemented");
    /* _XMP_local_continuous_copy((char *)dst_desc->real_addr+dst_offset, (char *)src_desc->real_addr+src_offset, */
    /* 			       dst_elmts, src_elmts, elmt_size); */
  }
  else{
#ifdef _XMP_TCA
    /* _XMP_tca_shortcut_get(target_rank, dst_offset, src_offset, dst_desc, src_desc,  */
    /* 			  dst_elmts, src_elmts, elmt_size); */
    _XMP_fatal("_XMP_tca_shortcut_get is unimplemented");
#elif _XMP_MPI3_ONESIDED
    _XMP_mpi_shortcut_get(target_rank, dst_desc, src_desc, dst_offset, src_offset,
			  dst_elmts, src_elmts, elmt_size, is_dst_on_acc, is_src_on_acc);
#else
    _XMP_fatal("_XMP_coarray_shortcut_get_acc is unavailable");
#endif
  }
}
