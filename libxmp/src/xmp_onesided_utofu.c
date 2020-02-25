#include <stdlib.h>
#define _XMP_UTOFU_EXTERN_
#include "xmp_internal.h"

void _XMP_utofu_initialize(void)
{
  int ret;
  size_t num_tnis;
  utofu_tni_id_t *tni_ids;

  // Get TNIID that can be used for one-sided comm.
  ret = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
  if( ret != UTOFU_SUCCESS || num_tnis == 0) {
    _XMP_fatal("utofu_get_onesided_tnis");
  }

  _xmp_utofu_tni_id = tni_ids[0];
  free(tni_ids);

  // Queries the TNI function related to one-sided communication.
  ret = utofu_query_onesided_caps(_xmp_utofu_tni_id, &(_xmp_utofu_onesided_caps));
  if( ret != UTOFU_SUCCESS ) {
    _XMP_fatal("utofu_query_onesided_caps");
  }

  // Create a VCQ and get the corresponding VCQ ID
  ret = utofu_create_vcq(_xmp_utofu_tni_id, 0, &(_xmp_utofu_vcq_hdl));
  if( ret != UTOFU_SUCCESS ) {
    switch( ret ) {
      case UTOFU_ERR_FULL:
        _XMP_fatal("utofu_create_vcq : No more VCQs can be created for this TNI.");
        break;
      case UTOFU_ERR_NOT_AVAILABLE:
        _XMP_fatal("utofu_create_vcq : The type of VCQ specified by the flags parameter cannot be created.");
        break;
      case UTOFU_ERR_NOT_SUPPORTED:
        _XMP_fatal("utofu_create_vcq : The type of VCQ specified by the flags parameter is not supported.");
        break;
      default:
        _XMP_fatal("utofu_create_vcq : other error");
    }
  }

  utofu_vcq_id_t vcq_ids;
  ret = utofu_query_vcq_id(_xmp_utofu_vcq_hdl, &vcq_ids);
  if( ret != UTOFU_SUCCESS ) {
    _XMP_fatal("utofu_query_vcq_id");
  }

  _xmp_utofu_vcq_ids = (utofu_vcq_id_t*)malloc(sizeof(utofu_vcq_id_t) * _XMP_world_size);
  _xmp_utofu_vcq_ids_org = (utofu_vcq_id_t*)malloc(sizeof(utofu_vcq_id_t) * _XMP_world_size);

  // Send VCQ ID to all ranks
  MPI_Allgather(&vcq_ids, 1, MPI_UINT64_T,
                _xmp_utofu_vcq_ids, 1, MPI_UINT64_T,
                MPI_COMM_WORLD);

  // Copy vcq id original
  for( int i = 0; i < _XMP_world_size; i++ )
    _xmp_utofu_vcq_ids_org[i] = _xmp_utofu_vcq_ids[i];

  // Set default communication path coordinates by remote VCQ ID
  for( int i = 0; i < _XMP_world_size; i++ ) {
    if( i != _XMP_world_rank ) {
      utofu_set_vcq_id_path( &(_xmp_utofu_vcq_ids[i]), NULL );
    }
  }

  _xmp_utofu_num_of_puts = 0;
  _xmp_utofu_num_of_gets = 0;

  _xmp_utofu_edata_flag_sync_images = ((1 << (8 * _xmp_utofu_onesided_caps->max_edata_size)) - 1);
  _xmp_utofu_edata_flag_armw_puts   = ((1 << (8 * _xmp_utofu_onesided_caps->max_edata_size)) - 1);
  _xmp_utofu_edata_flag_armw_gets   = ((1 << (8 * _xmp_utofu_onesided_caps->max_edata_size)) - 2);

  MPI_Barrier(MPI_COMM_WORLD);
}

void _XMP_utofu_finalize(void)
{
  _XMP_utofu_sync_all();
  utofu_free_vcq(_xmp_utofu_vcq_hdl);
}
