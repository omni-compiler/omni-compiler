#include <stdlib.h>
#include "xmp.h"
#include "xmp_internal.h"

// Check return code of utofu function
void _XMP_utofu_error_check( int utofu_ret )
{
  fprintf(stderr, "utofu return code : ");
  switch( utofu_ret ) {
    case UTOFU_SUCCESS:
      fprintf(stderr, "UTOFU_SUCCESS\n");
      break;
    case UTOFU_ERR_NOT_FOUND:
      fprintf(stderr, "UTOFU_ERR_NOT_FOUND\n");
      break;
    case UTOFU_ERR_NOT_COMPLETED:
      fprintf(stderr, "UTOFU_ERR_NOT_COMPLETED\n");
      break;
    case UTOFU_ERR_NOT_PROCESSED:
      fprintf(stderr, "UTOFU_ERR_NOT_PROCESSED\n");
      break;
    case UTOFU_ERR_BUSY:
      fprintf(stderr, "UTOFU_ERR_BUSY\n");
      break;
    case UTOFU_ERR_USED:
      fprintf(stderr, "UTOFU_ERR_USED\n");
      break;
    case UTOFU_ERR_FULL:
      fprintf(stderr, "UTOFU_ERR_FULL\n");
      break;
    case UTOFU_ERR_NOT_AVAILABLE:
      fprintf(stderr, "UTOFU_ERR_NOT_AVAILABLE\n");
      break;
    case UTOFU_ERR_NOT_SUPPORTED:
      fprintf(stderr, "UTOFU_ERR_NOT_SUPPORTED\n");
      break;
    case UTOFU_ERR_TCQ_OTHER:
      fprintf(stderr, "UTOFU_ERR_TCQ_OTHER\n");
      break;
    case UTOFU_ERR_TCQ_DESC:
      fprintf(stderr, "UTOFU_ERR_TCQ_DESC\n");
      break;
    case UTOFU_ERR_TCQ_MEMORY:
      fprintf(stderr, "UTOFU_ERR_TCQ_MEMORY\n");
      break;
    case UTOFU_ERR_TCQ_STADD:
      fprintf(stderr, "UTOFU_ERR_TCQ_STADD\n");
      break;
    case UTOFU_ERR_TCQ_LENGTH:
      fprintf(stderr, "UTOFU_ERR_TCQ_LENGTH\n");
      break;
    case UTOFU_ERR_MRQ_OTHER:
      fprintf(stderr, "UTOFU_ERR_MRQ_OTHER\n");
      break;
    case UTOFU_ERR_MRQ_PEER:
      fprintf(stderr, "UTOFU_ERR_MRQ_PEER\n");
      break;
    case UTOFU_ERR_MRQ_LCL_MEMORY:
      fprintf(stderr, "UTOFU_ERR_MRQ_LCL_MEMORY\n");
      break;
    case UTOFU_ERR_MRQ_RMT_MEMORY:
      fprintf(stderr, "UTOFU_ERR_MRQ_RMT_MEMORY\n");
      break;
    case UTOFU_ERR_MRQ_LCL_STADD:
      fprintf(stderr, "UTOFU_ERR_MRQ_LCL_STADD\n");
      break;
    case UTOFU_ERR_MRQ_RMT_STADD:
      fprintf(stderr, "UTOFU_ERR_MRQ_RMT_STADD\n");
      break;
    case UTOFU_ERR_MRQ_LCL_LENGTH:
      fprintf(stderr, "UTOFU_ERR_MRQ_LCL_LENGTH\n");
      break;
    case UTOFU_ERR_MRQ_RMT_LENGTH:
      fprintf(stderr, "UTOFU_ERR_MRQ_RMT_LENGTH\n");
      break;
    case UTOFU_ERR_BARRIER_OTHER:
      fprintf(stderr, "UTOFU_ERR_BARRIER_OTHER\n");
      break;
    case UTOFU_ERR_BARRIER_MISMATCH:
      fprintf(stderr, "UTOFU_ERR_BARRIER_MISMATCH\n");
      break;
    case UTOFU_ERR_INVALID_ARG:
      fprintf(stderr, "UTOFU_ERR_INVALID_ARG\n");
      break;
    case UTOFU_ERR_INVALID_POINTER:
      fprintf(stderr, "UTOFU_ERR_INVALID_POINTER\n");
      break;
    case UTOFU_ERR_INVALID_FLAGS:
      fprintf(stderr, "UTOFU_ERR_INVALID_FLAGS\n");
      break;
    case UTOFU_ERR_INVALID_COORDS:
      fprintf(stderr, "UTOFU_ERR_INVALID_COORDS\n");
      break;
    case UTOFU_ERR_INVALID_PATH:
      fprintf(stderr, "UTOFU_ERR_INVALID_PATH\n");
      break;
    case UTOFU_ERR_INVALID_TNI_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_TNI_ID\n");
      break;
    case UTOFU_ERR_INVALID_CQ_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_CQ_ID\n");
      break;
    case UTOFU_ERR_INVALID_BG_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_BG_ID\n");
      break;
    case UTOFU_ERR_INVALID_CMP_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_CMP_ID\n");
      break;
    case UTOFU_ERR_INVALID_VCQ_HDL:
      fprintf(stderr, "UTOFU_ERR_INVALID_VCQ_HDL\n");
      break;
    case UTOFU_ERR_INVALID_VCQ_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_VCQ_ID\n");
      break;
    case UTOFU_ERR_INVALID_VBG_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_VBG_ID\n");
      break;
    case UTOFU_ERR_INVALID_PATH_ID:
      fprintf(stderr, "UTOFU_ERR_INVALID_PATH_ID\n");
      break;
    case UTOFU_ERR_INVALID_STADD:
      fprintf(stderr, "UTOFU_ERR_INVALID_STADD\n");
      break;
    case UTOFU_ERR_INVALID_ADDRESS:
      fprintf(stderr, "UTOFU_ERR_INVALID_ADDRESS\n");
      break;
    case UTOFU_ERR_INVALID_SIZE:
      fprintf(stderr, "UTOFU_ERR_INVALID_SIZE\n");
      break;
    case UTOFU_ERR_INVALID_STAG:
      fprintf(stderr, "UTOFU_ERR_INVALID_STAG\n");
      break;
    case UTOFU_ERR_INVALID_EDATA:
      fprintf(stderr, "UTOFU_ERR_INVALID_EDATA\n");
      break;
    case UTOFU_ERR_INVALID_NUMBER:
      fprintf(stderr, "UTOFU_ERR_INVALID_NUMBER\n");
      break;
    case UTOFU_ERR_INVALID_OP:
      fprintf(stderr, "UTOFU_ERR_INVALID_OP\n");
      break;
    case UTOFU_ERR_INVALID_DESC:
      fprintf(stderr, "UTOFU_ERR_INVALID_DESC\n");
      break;
    case UTOFU_ERR_INVALID_DATA:
      fprintf(stderr, "UTOFU_ERR_INVALID_DATA\n");
      break;
    case UTOFU_ERR_OUT_OF_RESOURCE:
      fprintf(stderr, "UTOFU_ERR_OUT_OF_RESOURCE\n");
      break;
    case UTOFU_ERR_OUT_OF_MEMORY:
      fprintf(stderr, "UTOFU_ERR_OUT_OF_MEMORY\n");
      break;
    case UTOFU_ERR_FATAL:
      fprintf(stderr, "UTOFU_ERR_FATAL\n");
      break;
    default:
      fprintf(stderr, "unknown return code : %d\n", utofu_ret);
  }
}


// for sync images >>>
typedef struct _XMP_utofu_vcqid_table {
  int rank;
  utofu_vcq_id_t vcqid;
} _XMP_utofu_vcqid_t;

static _XMP_utofu_vcqid_t *_xmp_utofu_vcqid_t;
static unsigned int *_sync_images_table;
static utofu_stadd_t _lcl_stadd;
static utofu_stadd_t *_rmt_stadds;

int compare_vcqid_table(const void *a, const void *b)
{
  if(      ((_XMP_utofu_vcqid_t*)a)->vcqid < ((_XMP_utofu_vcqid_t*)b)->vcqid ) return -1;
  else if( ((_XMP_utofu_vcqid_t*)a)->vcqid > ((_XMP_utofu_vcqid_t*)b)->vcqid ) return  1;
  else return 0;
}

static void _add_sync_images_table_rank(const int rank)
{
  _sync_images_table[rank]++;
}

static void _add_sync_images_table_vcqid(const utofu_vcq_id_t vcqid)
{
  _XMP_utofu_vcqid_t *p;
  _XMP_utofu_vcqid_t key;
  key.vcqid = vcqid;
  p = (_XMP_utofu_vcqid_t *)bsearch( &key, _xmp_utofu_vcqid_t, _XMP_world_size, sizeof(_XMP_utofu_vcqid_t), compare_vcqid_table);
  if( p == NULL ) {
    _XMP_fatal("_add_sync_images_table_vcqid : invalid vcqid");
  }
  else {
    _sync_images_table[p->rank]++;
  }
}
// <<< for sync images

uint64_t _XMP_utofu_check_mrq_notice( struct utofu_mrq_notice *notice )
{
  uint64_t rmt_value = 0;

  if( notice->notice_type == UTOFU_MRQ_TYPE_RMT_PUT ) {
    if( notice->edata == _xmp_utofu_edata_flag_sync_images ) {
      _add_sync_images_table_vcqid( notice->vcq_id );
    }
    else {
      _xmp_utofu_add_postreq( notice->vcq_id, (int)notice->edata );
    }
  }
  else if( notice->notice_type == UTOFU_MRQ_TYPE_LCL_GET ) {
    _xmp_utofu_num_of_gets--;
  }
  else if( notice->notice_type == UTOFU_MRQ_TYPE_LCL_PUT ) {
    _xmp_utofu_num_of_puts--;
  }
  else if( notice->notice_type == UTOFU_MRQ_TYPE_LCL_ARMW ) {
    if( notice->edata == _xmp_utofu_edata_flag_armw_puts ) {
      _xmp_utofu_num_of_puts--;
    }
    else if( notice->edata == _xmp_utofu_edata_flag_armw_gets ) {
      _xmp_utofu_num_of_gets--;
    }
    rmt_value = notice->rmt_value;
  }

  return rmt_value;
}

void _XMP_utofu_coarray_malloc( _XMP_coarray_t *coarray_desc, void **addr, const size_t coarray_size )
{
  *addr = _XMP_alloc( coarray_size );
  _XMP_utofu_regmem( coarray_desc, *addr, coarray_size );
}

void _XMP_utofu_regmem( _XMP_coarray_t *coarray_desc, void *addr, const size_t coarray_size)
{
  utofu_stadd_t *each_addr = _XMP_alloc(sizeof(utofu_stadd_t) * _XMP_world_size);

  coarray_desc->stadds = (utofu_stadd_t*)malloc(sizeof(utofu_stadd_t) * _XMP_world_size);
  utofu_stadd_t stadds;
  utofu_reg_mem(_xmp_utofu_vcq_hdl, addr, coarray_size, 0, &stadds);

  MPI_Comm comm = xmp_get_mpi_comm();
  int image_size = xmp_num_images();
  if( image_size == _XMP_world_size ) {
    MPI_Allgather(&stadds, 1, MPI_UINT64_T,
                  coarray_desc->stadds, 1, MPI_UINT64_T,
                  comm);
  }
  else {
    utofu_stadd_t *tmp_stadds;
    tmp_stadds = (utofu_stadd_t*)malloc(sizeof(utofu_stadd_t) * image_size);
    MPI_Allgather(&stadds, 1, MPI_UINT64_T,
                  tmp_stadds, 1, MPI_UINT64_T,
                  comm);

    MPI_Group grp, world_grp;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    MPI_Comm_group(comm, &grp);
    int grp_size;
    MPI_Group_size(grp, &grp_size);
    int *ranks = malloc(grp_size * sizeof(int));
    int *world_ranks = malloc(grp_size * sizeof(int));
    for( int i = 0; i < grp_size; i++ )
      ranks[i] = i;
    MPI_Group_translate_ranks(grp, grp_size, ranks, world_grp, world_ranks);

    for( int i = 0; i < grp_size; i++ )
      coarray_desc->stadds[world_ranks[i]] = tmp_stadds[ranks[i]];

    free(ranks);
    free(world_ranks);
    MPI_Group_free(&grp);
    MPI_Group_free(&world_grp);
    free(tmp_stadds);
  }

  coarray_desc->real_addr = addr;
  coarray_desc->addr = (void *)each_addr;
}

void _XMP_utofu_deallocate( _XMP_coarray_t *coarray_desc )
{
  _XMP_utofu_sync_all();

  int image_size = xmp_num_images();
  if( image_size == _XMP_world_size ) {
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, coarray_desc->stadds[_XMP_world_rank], 0);
  }
  else {
    int rank = xmpc_node_num();
    int world_rank;
    MPI_Comm comm = xmp_get_mpi_comm();
    MPI_Group grp, world_grp;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    MPI_Comm_group(comm, &grp);
    MPI_Group_translate_ranks(grp, 1, &rank, world_grp, &world_rank);
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, coarray_desc->stadds[world_rank], 0);
    MPI_Group_free(&grp);
    MPI_Group_free(&world_grp);
  }
  free( coarray_desc->stadds );
}

static void _utofu_put( const int target_rank,
                        utofu_stadd_t src_stadd, utofu_stadd_t dst_stadd,
                        const size_t transfer_size, const uint64_t edata,
                        const unsigned long int post_flags, const uintptr_t cbvalue )
{
  utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];

  const size_t max_putget_size = _xmp_utofu_onesided_caps->max_putget_size;

  if( transfer_size <= max_putget_size ) {
    _xmp_utofu_num_of_puts++;
    utofu_put( _xmp_utofu_vcq_hdl, rmt_vcq_id, src_stadd, dst_stadd, transfer_size,
               edata, post_flags, (void *)cbvalue );
  }
  else {
    int times = transfer_size / max_putget_size;
    int rest  = transfer_size - (max_putget_size * times);

    for( int i = 0; i < times; i++ ) {
      _xmp_utofu_num_of_puts++;
      utofu_put( _xmp_utofu_vcq_hdl, rmt_vcq_id, src_stadd, dst_stadd, max_putget_size,
                 edata, post_flags, (void *)cbvalue );
      src_stadd += max_putget_size;
      dst_stadd += max_putget_size;
    }

    if( rest != 0 ) {
      _xmp_utofu_num_of_puts++;
      utofu_put( _xmp_utofu_vcq_hdl, rmt_vcq_id, src_stadd, dst_stadd, rest,
                 edata, post_flags, (void *)cbvalue );
    }
  }
}

static void _utofu_mput(const size_t target_rank,
                        utofu_stadd_t* dst_stadds, utofu_stadd_t* src_stadds,
                        // size_t* lengths, const int stride, const size_t transfer_elmts,
                        size_t* lengths, const long stride, const size_t transfer_elmts,
                        const uint64_t edata, const unsigned long int post_flags,
                        const uintptr_t cbvalue)
{
  if( stride == 0 ) {
    for( size_t i = 0; i < transfer_elmts; i++ ) {
      _utofu_put(target_rank, src_stadds[i], dst_stadds[i],
                 lengths[i], edata, post_flags, cbvalue);
    }
  }
  else if((stride > 0) && (lengths[0] <= _xmp_utofu_onesided_caps->max_putget_size)) {
    utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];
    _xmp_utofu_num_of_puts += transfer_elmts;
    utofu_put_stride(_xmp_utofu_vcq_hdl, rmt_vcq_id, src_stadds[0], dst_stadds[0],
                     lengths[0], stride, transfer_elmts, edata, post_flags, (void *)cbvalue);
  }
  else{
    for( size_t i = 0; i < transfer_elmts; i++ ) {
      _utofu_put(target_rank, src_stadds[0], dst_stadds[0],
                 lengths[0], edata, post_flags, cbvalue);
      dst_stadds[0] += stride;
      src_stadds[0] += stride;
    }
  }
}

static void _utofu_scalar_mput_do(const size_t target_rank,
                                  utofu_stadd_t* dst_stadds, utofu_stadd_t* src_stadds,
                                  size_t* lengths, const size_t transfer_elmts,
                                  const uint64_t edata,
                                  const unsigned long int post_flags, const uintptr_t cbvalue)
{
  _utofu_mput(target_rank, dst_stadds, src_stadds, lengths, 0, transfer_elmts,
              edata, post_flags, cbvalue);
}


static void _XMP_utofu_sync_memory_get( const unsigned long int post_flags )
{
  int ret;

  if( post_flags & UTOFU_ONESIDED_FLAG_TCQ_NOTICE ) {
    void *cbdata;
    ret = utofu_poll_tcq(_xmp_utofu_vcq_hdl, 0, &cbdata);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_fatal("_XMP_utofu_sync_memory_get : utofu_poll_tcq not success");
    }
  }

  if( post_flags & UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE ) {
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_utofu_error_check( ret );
      _XMP_fatal("_XMP_utofu_sync_memory_get : utofu_poll_mrq not success");
    }
    if( ret == UTOFU_SUCCESS )
      _XMP_utofu_check_mrq_notice( &notice );
  }
}

static void _XMP_utofu_sync_memory_get_all( const unsigned long int post_flags )
{
  while( _xmp_utofu_num_of_gets != 0 ) {
    _XMP_utofu_sync_memory_get( post_flags );
  }
}

static void _XMP_utofu_sync_memory_put( const unsigned long int post_flags )
{
  int ret;

  if( post_flags & UTOFU_ONESIDED_FLAG_TCQ_NOTICE ) {
    void *cbdata;
    ret = utofu_poll_tcq(_xmp_utofu_vcq_hdl, 0, &cbdata);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_fatal("_XMP_utofu_sync_memory_put : utofu_poll_tcq not success");
    }
  }

  if( post_flags & UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE ) {
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_utofu_error_check( ret );
      _XMP_fatal("_XMP_utofu_sync_memory_put : utofu_poll_mrq not success");
    }
    if( ret == UTOFU_SUCCESS )
      _XMP_utofu_check_mrq_notice( &notice );
  }
}

static void _XMP_utofu_sync_memory_put_all( const unsigned long int post_flags )
{
  while( _xmp_utofu_num_of_puts != 0 ) {
    _XMP_utofu_sync_memory_put( post_flags );
  }
}

void _XMP_utofu_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
                               const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                               const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  int ident_val = _xmp_utofu_num_of_puts;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_PUT_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  size_t transfer_size = dst_elmts * elmt_size;

  utofu_stadd_t src_stadd = src_desc->stadds[_XMP_world_rank] + src_offset;
  utofu_stadd_t dst_stadd = dst_desc->stadds[target_rank] + dst_offset;

  if( dst_elmts == src_elmts){
    _utofu_put( target_rank, src_stadd, dst_stadd,
                transfer_size, edata, post_flags, cbvalue );
  }
  else if(src_elmts == 1){
    utofu_stadd_t dst_stadds[dst_elmts], src_stadds[dst_elmts];
    size_t lengths[dst_elmts];
    for( size_t i = 0; i < dst_elmts; i++ ) dst_stadds[i] = dst_stadd + i * elmt_size;
    for( size_t i = 0; i < dst_elmts; i++ ) src_stadds[i] = src_stadd;
    for( size_t i = 0; i < dst_elmts; i++ ) lengths[i]    = elmt_size;
    _utofu_scalar_mput_do(target_rank, dst_stadds, src_stadds, lengths, dst_elmts,
                          edata, post_flags, cbvalue);
  }
  else{
    _XMP_fatal("_XMP_utofu_contiguous_put : error");
  }

  _XMP_utofu_sync_memory_put_all( post_flags );
}


static void _utofu_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
                                  const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                                  char *src, const size_t transfer_size)
{
  int ident_val = _xmp_utofu_num_of_puts;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_PUT_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t dst_stadd = dst_desc->stadds[target_rank] + dst_offset;
  utofu_stadd_t src_stadd;

  if(src_desc == NULL){
    utofu_reg_mem(_xmp_utofu_vcq_hdl, src + src_offset, transfer_size, 0, &src_stadd);
  }
  else{
    src_stadd = src_desc->stadds[_XMP_world_rank] + src_offset;
  }

  _utofu_put( target_rank, src_stadd, dst_stadd,
              transfer_size, edata, post_flags, cbvalue );

  _XMP_utofu_sync_memory_put_all( post_flags );

  if(src_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, src_stadd, 0);
}

static void _utofu_NON_contiguous_the_same_stride_mput(const int target_rank,
                                                       utofu_stadd_t dst_stadd, utofu_stadd_t src_stadd,
                                                       const size_t transfer_elmts, const _XMP_array_section_t *array_info,
                                                       const int array_dims, size_t elmt_size,
                                                       const uint64_t edata,
                                                       const unsigned long int post_flags,
                                                       const uintptr_t cbvalue)
{
  size_t copy_chunk_dim = (size_t)_XMP_get_dim_of_allelmts(array_dims, array_info);
  size_t copy_chunk     = (size_t)_XMP_calc_copy_chunk(copy_chunk_dim, array_info);
  size_t copy_elmts     = transfer_elmts/(copy_chunk/elmt_size);
  long   stride         = _XMP_calc_stride(array_info, array_dims, copy_chunk);

  _utofu_mput(target_rank, &dst_stadd, &src_stadd, &copy_chunk, stride, copy_elmts,
              edata, post_flags, cbvalue);
}

static void _utofu_NON_contiguous_general_mput(const int target_rank,
                                               utofu_stadd_t dst_stadd, utofu_stadd_t src_stadd,
                                               const size_t transfer_elmts,
                                               const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
                                               const int dst_dims, const int src_dims, size_t elmt_size,
                                               const uint64_t edata,
                                               const unsigned long int post_flags,
                                               const uintptr_t cbvalue)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  utofu_stadd_t dst_stadds[copy_elmts], src_stadds[copy_elmts];
  size_t lengths[copy_elmts];

  _XMP_set_coarray_addresses_with_chunk(dst_stadds, dst_stadd, dst_info, dst_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(src_stadds, src_stadd, src_info, src_dims, copy_chunk, copy_elmts);
  for(size_t i=0;i<copy_elmts;i++) lengths[i] = copy_chunk;

  _utofu_scalar_mput_do(target_rank, dst_stadds, src_stadds, lengths, copy_elmts,
                        edata, post_flags, cbvalue);
}

static void _utofu_NON_contiguous_put(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
                                      const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
                                      const int dst_dims, const int src_dims,
                                      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                                      void *src, const size_t transfer_elmts)
{
  int ident_val = _xmp_utofu_num_of_puts;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_PUT_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t src_stadd;
  utofu_stadd_t dst_stadd = dst_desc->stadds[target_rank] + dst_offset;
  size_t elmt_size = dst_desc->elmt_size;

  if(src_desc == NULL){
    utofu_reg_mem(_xmp_utofu_vcq_hdl, src + src_offset, transfer_elmts * elmt_size, 0, &src_stadd);
  }
  else{
    src_stadd = src_desc->stadds[_XMP_world_rank] + src_offset;
  }

  if(_XMP_is_the_same_constant_stride(dst_info, src_info, dst_dims, src_dims)){
    _utofu_NON_contiguous_the_same_stride_mput(target_rank, dst_stadd, src_stadd, transfer_elmts,
                                               dst_info, dst_dims, elmt_size,
                                               edata, post_flags, cbvalue);
  }
  else{
    _utofu_NON_contiguous_general_mput(target_rank, dst_stadd, src_stadd, transfer_elmts,
                                       dst_info, src_info, dst_dims, src_dims, elmt_size,
                                       edata, post_flags, cbvalue);
  }

  _XMP_utofu_sync_memory_put_all( post_flags );

  if(src_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, src_stadd, 0);
}

static void _utofu_scalar_mput(const int target_rank,
                               const uint64_t dst_offset, const uint64_t src_offset,
                               const _XMP_array_section_t *dst_info, const int dst_dims,
                               const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                               char *src, const size_t transfer_elmts)
{
  int ident_val = _xmp_utofu_num_of_puts;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_PUT_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t dst_stadd = dst_desc->stadds[target_rank] + dst_offset;
  utofu_stadd_t src_stadd;
  size_t elmt_size = dst_desc->elmt_size;
  utofu_stadd_t dst_stadds[transfer_elmts], src_stadds[transfer_elmts];
  size_t lengths[transfer_elmts];

  if(src_desc == NULL) {
    utofu_reg_mem(_xmp_utofu_vcq_hdl, src + src_offset, transfer_elmts * elmt_size, 0, &src_stadd);
  }
  else {
    src_stadd = src_desc->stadds[_XMP_world_rank] + src_offset;
  }

  _XMP_set_coarray_addresses(dst_stadd, dst_info, dst_dims, transfer_elmts, dst_stadds);
  for( size_t i = 0; i < transfer_elmts; i++ ) src_stadds[i] = src_stadd;
  for( size_t i = 0; i < transfer_elmts; i++ ) lengths[i] = elmt_size;

  _utofu_scalar_mput_do(target_rank, dst_stadds, src_stadds, lengths, transfer_elmts,
                        edata, post_flags, cbvalue);
  _XMP_utofu_sync_memory_put_all( post_flags );

  if(src_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, src_stadd, 0);
}

void _XMP_utofu_put(const int dst_contiguous, const int src_contiguous, const int target_rank,
                    const int dst_dims, const int src_dims, const _XMP_array_section_t *dst_info,
                    const _XMP_array_section_t *src_info, const _XMP_coarray_t *dst_desc,
                    const _XMP_coarray_t *src_desc, void *src, const size_t dst_elmts, const size_t src_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = dst_desc->elmt_size * dst_elmts;

  if(dst_elmts == src_elmts){
    if(dst_contiguous == _XMP_N_INT_TRUE && src_contiguous == _XMP_N_INT_TRUE){
      _utofu_contiguous_put(target_rank, dst_offset, src_offset,
                            dst_desc, src_desc, src, transfer_size);
    }
    else{
      _utofu_NON_contiguous_put(target_rank, dst_offset, src_offset,
                                dst_info, src_info, dst_dims, src_dims,
                                dst_desc, src_desc, src, dst_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _utofu_scalar_mput(target_rank, dst_offset, src_offset,
                         dst_info, dst_dims, dst_desc, src_desc, src, dst_elmts);
    }
    else{
      _XMP_fatal("_XMP_utofu_put : Number of elements is invalid");
    }
  }
}


static void _utofu_get( const int target_rank,
                        utofu_stadd_t lcl_stadd, utofu_stadd_t rmt_stadd,
                        const size_t transfer_size, const uint64_t edata,
                        const unsigned long int post_flags, const uintptr_t cbvalue )
{
  utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];

  const size_t max_putget_size = _xmp_utofu_onesided_caps->max_putget_size;

  if( transfer_size <= max_putget_size ) {
    _xmp_utofu_num_of_gets++;
    utofu_get( _xmp_utofu_vcq_hdl, rmt_vcq_id, lcl_stadd, rmt_stadd, transfer_size,
               edata, post_flags, (void *)cbvalue );
  }
  else {
    int times = transfer_size / max_putget_size;
    int rest  = transfer_size - (max_putget_size * times);

    for( int i = 0; i < times; i++ ) {
      _xmp_utofu_num_of_gets++;
      utofu_get( _xmp_utofu_vcq_hdl, rmt_vcq_id, lcl_stadd, rmt_stadd, max_putget_size,
                 edata, post_flags, (void *)cbvalue );
      lcl_stadd += max_putget_size;
      rmt_stadd += max_putget_size;
    }

    if( rest != 0 ) {
      _xmp_utofu_num_of_gets++;
      utofu_get( _xmp_utofu_vcq_hdl, rmt_vcq_id, lcl_stadd, rmt_stadd, rest,
                 edata, post_flags, (void *)cbvalue );
    }
  }
}

static void _utofu_mget(const size_t target_rank,
                        utofu_stadd_t* lcl_stadds, utofu_stadd_t* rmt_stadds,
                        size_t* lengths, const long stride, const size_t transfer_elmts,
                        const uint64_t edata, const unsigned long int post_flags,
                        const uintptr_t cbvalue)
{
  if( stride == 0 ) {
    for( size_t i = 0; i < transfer_elmts; i++ ) {
      _utofu_get(target_rank, lcl_stadds[i], rmt_stadds[i],
                 lengths[i], edata, post_flags, cbvalue);
    }
  }
  else if((stride > 0) && (lengths[0] <= _xmp_utofu_onesided_caps->max_putget_size)) {
    utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[target_rank];
    _xmp_utofu_num_of_gets += transfer_elmts;
    utofu_get_stride(_xmp_utofu_vcq_hdl, rmt_vcq_id, lcl_stadds[0], rmt_stadds[0],
                     lengths[0], stride, transfer_elmts, edata, post_flags, (void *)cbvalue);
  }
  else {
    for( size_t i = 0; i < transfer_elmts; i++ ) {
      _utofu_get(target_rank, lcl_stadds[0], rmt_stadds[0],
                 lengths[0], edata, post_flags, cbvalue);
      lcl_stadds[0] += stride;
      rmt_stadds[0] += stride;
    }
  }
}

static void _utofu_scalar_mget_do(const size_t target_rank,
                                  utofu_stadd_t* lcl_stadds, utofu_stadd_t* rmt_stadds,
                                  size_t* lengths, const size_t transfer_elmts,
                                  const uint64_t edata,
                                  const unsigned long int post_flags, const uintptr_t cbvalue)
{
  _utofu_mget(target_rank, lcl_stadds, rmt_stadds, lengths, 0, transfer_elmts,
              edata, post_flags, cbvalue);
}

void _XMP_utofu_contiguous_get(const int target_rank, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
				const uint64_t dst_offset, const uint64_t src_offset,
				const size_t dst_elmts, const size_t src_elmts, const size_t elmt_size)
{
  int ident_val = _xmp_utofu_num_of_gets;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_GET_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  size_t transfer_size = dst_elmts * elmt_size;
  utofu_stadd_t rmt_stadd = src_desc->stadds[target_rank] + src_offset;
  utofu_stadd_t lcl_stadd = dst_desc->stadds[_XMP_world_rank] + dst_offset;

  if(dst_elmts == src_elmts){
    _utofu_get( target_rank, lcl_stadd, rmt_stadd,
                transfer_size, edata, post_flags, cbvalue );
    _XMP_utofu_sync_memory_get_all( post_flags );
  }
  else if(src_elmts == 1){
    _utofu_get( target_rank, lcl_stadd, rmt_stadd,
                elmt_size, edata, post_flags, cbvalue );
    _XMP_utofu_sync_memory_get_all( post_flags );

    char *dst = dst_desc->real_addr + dst_offset;
    for( size_t i = 1; i < dst_elmts; i++ )
      memcpy( dst + i * elmt_size, dst, elmt_size );
  }
  else{
    _XMP_fatal("Coarray Error ! transfer size is wrong.\n");
  }
}

static void _utofu_contiguous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
                                  char *dst, const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                                  const size_t transfer_size)
{
  int ident_val = _xmp_utofu_num_of_gets;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_GET_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t rmt_stadd = src_desc->stadds[target_rank] + src_offset;
  utofu_stadd_t lcl_stadd;

  if(dst_desc == NULL){
    utofu_reg_mem(_xmp_utofu_vcq_hdl, dst + dst_offset, transfer_size, 0, &lcl_stadd);
  }
  else{
    lcl_stadd = dst_desc->stadds[_XMP_world_rank] + dst_offset;
  }

  _utofu_get( target_rank, lcl_stadd, rmt_stadd,
              transfer_size, edata, post_flags, cbvalue );

  _XMP_utofu_sync_memory_get_all( post_flags );

  if(dst_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, lcl_stadd, 0);
}

static void _utofu_NON_contiguous_the_same_stride_mget(const int target_rank,
                                                       utofu_stadd_t lcl_stadd, utofu_stadd_t rmt_stadd,
                                                       const size_t transfer_elmts, const _XMP_array_section_t *array_info,
                                                       const int array_dims, size_t elmt_size,
                                                       const uint64_t edata,
                                                       const unsigned long int post_flags,
                                                       const uintptr_t cbvalue)
{
  size_t copy_chunk_dim = (size_t)_XMP_get_dim_of_allelmts(array_dims, array_info);
  size_t copy_chunk     = (size_t)_XMP_calc_copy_chunk(copy_chunk_dim, array_info);
  size_t copy_elmts     = transfer_elmts/(copy_chunk/elmt_size);
  long   stride         = _XMP_calc_stride(array_info, array_dims, copy_chunk);

  _utofu_mget(target_rank, &lcl_stadd, &rmt_stadd, &copy_chunk, stride, copy_elmts,
              edata, post_flags, cbvalue);
}

static void _utofu_NON_contiguous_general_mget(const int target_rank,
                                               utofu_stadd_t lcl_stadd, utofu_stadd_t rmt_stadd,
                                               const size_t transfer_elmts,
                                               const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
                                               const int dst_dims, const int src_dims, size_t elmt_size,
                                               const uint64_t edata,
                                               const unsigned long int post_flags,
                                               const uintptr_t cbvalue)
{
  size_t copy_chunk = _XMP_calc_max_copy_chunk(dst_dims, src_dims, dst_info, src_info);
  size_t copy_elmts = transfer_elmts/(copy_chunk/elmt_size);
  utofu_stadd_t lcl_stadds[copy_elmts], rmt_stadds[copy_elmts];
  size_t lengths[copy_elmts];

  _XMP_set_coarray_addresses_with_chunk(lcl_stadds, lcl_stadd, dst_info, dst_dims, copy_chunk, copy_elmts);
  _XMP_set_coarray_addresses_with_chunk(rmt_stadds, rmt_stadd, src_info, src_dims, copy_chunk, copy_elmts);
  for(size_t i=0;i<copy_elmts;i++) lengths[i] = copy_chunk;

  _utofu_scalar_mget_do(target_rank, lcl_stadds, rmt_stadds, lengths, copy_elmts,
                        edata, post_flags, cbvalue);
}

static void _utofu_NON_contiguous_get(const int target_rank, const uint64_t dst_offset, const uint64_t src_offset,
                                      const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info,
                                      const int dst_dims, const int src_dims,
                                      const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                                      void *dst, const size_t transfer_elmts)
{
  int ident_val = _xmp_utofu_num_of_gets;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_GET_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t rmt_stadd = src_desc->stadds[target_rank] + src_offset;
  utofu_stadd_t lcl_stadd;
  size_t elmt_size = src_desc->elmt_size;

  if(dst_desc == NULL){
    utofu_reg_mem(_xmp_utofu_vcq_hdl, dst + dst_offset, transfer_elmts * elmt_size, 0, &lcl_stadd);
  }
  else{
    lcl_stadd = dst_desc->stadds[_XMP_world_rank] + dst_offset;
  }

  if(_XMP_is_the_same_constant_stride(dst_info, src_info, dst_dims, src_dims)){
    _utofu_NON_contiguous_the_same_stride_mget(target_rank, lcl_stadd, rmt_stadd, transfer_elmts,
                                               src_info, src_dims, elmt_size,
                                               edata, post_flags, cbvalue);
  }
  else{
    _utofu_NON_contiguous_general_mget(target_rank, lcl_stadd, rmt_stadd, transfer_elmts,
                                       dst_info, src_info, dst_dims, src_dims, elmt_size,
                                       edata, post_flags, cbvalue);
  }

  _XMP_utofu_sync_memory_get_all( post_flags );

  if(dst_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, lcl_stadd, 0);
}

static void _utofu_scalar_mget(const int target_rank,
                               const uint64_t dst_offset, const uint64_t src_offset,
                               const _XMP_array_section_t *dst_info, const int dst_dims,
                               const _XMP_coarray_t *dst_desc, const _XMP_coarray_t *src_desc,
                               char *dst, const size_t transfer_elmts)
{
  int ident_val = _xmp_utofu_num_of_gets;
  unsigned long int post_flags;
  post_flags = _XMP_UTOFU_COARRAY_GET_POST_FLAGS;
  uint64_t edata = ident_val % (1UL << (8 * _xmp_utofu_onesided_caps->max_edata_size));
  uintptr_t cbvalue = ident_val;

  utofu_stadd_t rmt_stadd = src_desc->stadds[target_rank] + src_offset;
  utofu_stadd_t lcl_stadd;
  size_t elmt_size = src_desc->elmt_size;

  if(dst_desc == NULL){
    utofu_reg_mem(_xmp_utofu_vcq_hdl, dst + dst_offset, transfer_elmts * elmt_size, 0, &lcl_stadd);
  }
  else{
    lcl_stadd = dst_desc->stadds[_XMP_world_rank] + dst_offset;
  }

  _utofu_get( target_rank, lcl_stadd, rmt_stadd, elmt_size, edata, post_flags, cbvalue );
  _XMP_utofu_sync_memory_get_all( post_flags );

  char *src_addr = dst + dst_offset;
  char *dst_addr = src_addr;
  switch( dst_dims ) {
  case 1:
    _XMP_stride_memcpy_1dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 2:
    _XMP_stride_memcpy_2dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 3:
    _XMP_stride_memcpy_3dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 4:
    _XMP_stride_memcpy_4dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 5:
    _XMP_stride_memcpy_5dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 6:
    _XMP_stride_memcpy_6dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  case 7:
    _XMP_stride_memcpy_7dim(dst_addr, src_addr, dst_info, elmt_size, _XMP_SCALAR_MCOPY);
    break;
  default:
    _XMP_fatal("Coarray Error ! Dimension is too big.\n");
    break;
  }

  if(dst_desc == NULL)
    utofu_dereg_mem(_xmp_utofu_vcq_hdl, lcl_stadd, 0);
}

void _XMP_utofu_get(const int src_contiguous, const int dst_contiguous, const int target_rank, 
		    const int src_dims, const int dst_dims, 
		    const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		    const _XMP_coarray_t *src_desc, const _XMP_coarray_t *dst_desc, void *dst,
		    const size_t src_elmts, const size_t dst_elmts)
{
  uint64_t dst_offset = (uint64_t)_XMP_get_offset(dst_info, dst_dims);
  uint64_t src_offset = (uint64_t)_XMP_get_offset(src_info, src_dims);
  size_t transfer_size = src_desc->elmt_size * src_elmts;

  if(src_elmts == dst_elmts){
    if(dst_contiguous == _XMP_N_INT_TRUE && src_contiguous == _XMP_N_INT_TRUE){
      _utofu_contiguous_get(target_rank, dst_offset, src_offset, dst, dst_desc, src_desc, transfer_size);
    }
    else{
      _utofu_NON_contiguous_get(target_rank, dst_offset, src_offset,
                                dst_info, src_info, dst_dims, src_dims,
                                dst_desc, src_desc, dst, dst_elmts);
    }
  }
  else{
    if(src_elmts == 1){
      _utofu_scalar_mget(target_rank, dst_offset, src_offset,
                         dst_info, dst_dims, dst_desc, src_desc, (char *)dst, dst_elmts);
    }
    else{
      _XMP_fatal("Number of elements is invalid");
    }
  }
}

void _XMP_utofu_sync_memory(void)
{
  int ret;
  while( _xmp_utofu_num_of_puts > 0 || _xmp_utofu_num_of_gets > 0 ) {
    struct utofu_mrq_notice notice;
      ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret != UTOFU_SUCCESS && ret != UTOFU_ERR_NOT_FOUND ) {
      _XMP_utofu_error_check( ret );
      _XMP_fatal("_XMP_utofu_sync_memory : utofu_poll_mrq not success");
    }
    if( ret == UTOFU_SUCCESS )
      _XMP_utofu_check_mrq_notice( &notice );
  }
}

void _XMP_utofu_sync_all(void)
{
  _XMP_utofu_sync_memory();

  MPI_Comm comm = xmp_get_mpi_comm();
  MPI_Barrier(comm);
}

void _XMP_utofu_build_sync_images_table(void)
{
  _sync_images_table = malloc(sizeof(unsigned int) * _XMP_world_size);

  for( int i = 0; i < _XMP_world_size; i++ )
    _sync_images_table[i] = 0;

  double *token     = _XMP_alloc(sizeof(double));
  utofu_reg_mem(_xmp_utofu_vcq_hdl, token, sizeof(double), 0, &_lcl_stadd);
  _rmt_stadds       = _XMP_alloc(sizeof(utofu_stadd_t) * _XMP_world_size);
  MPI_Allgather(&_lcl_stadd, 1, MPI_UINT64_T, _rmt_stadds, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  _xmp_utofu_vcqid_t = (_XMP_utofu_vcqid_t*)malloc(sizeof(_XMP_utofu_vcqid_t) * _XMP_world_size);
  for( int i = 0; i < _XMP_world_size; i++ ) {
    _xmp_utofu_vcqid_t[i].rank  = i;
    _xmp_utofu_vcqid_t[i].vcqid = _xmp_utofu_vcq_ids_org[i];
  }
  qsort( _xmp_utofu_vcqid_t, _XMP_world_size, sizeof(_XMP_utofu_vcqid_t), compare_vcqid_table );
}

static void _notify_sync_images(const int num, int *rank_set)
{
  int num_of_requests = 0;

  for( int i = 0; i < num; i++ ) {
    if( rank_set[i] == _XMP_world_rank ) {
      _add_sync_images_table_rank( _XMP_world_rank );
    }
    else {
      int ret;
      int node_num = rank_set[i];
      utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[node_num];

      int ident_val = num_of_requests;
      uint64_t edata = _xmp_utofu_edata_flag_sync_images;
      unsigned long int post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_STRONG_ORDER;
      uintptr_t cbvalue = ident_val;
      num_of_requests++;
      utofu_put(_xmp_utofu_vcq_hdl, rmt_vcq_id, _lcl_stadd, _rmt_stadds[node_num],
                sizeof(double), edata, post_flags, (void *)cbvalue);

      void *cbdata;
      do {
        ret = utofu_poll_tcq(_xmp_utofu_vcq_hdl, 0, &cbdata);
      } while (ret == UTOFU_ERR_NOT_FOUND);
      if( ret != UTOFU_SUCCESS ) {
        _XMP_fatal("_notify_sync_images : utofu_put, utofu_poll_tcq not success");
      }
    }
  }
}

static _Bool _check_sync_images_table(const int num, int *rank_set)
{
  int checked = 0;
  for( int i = 0; i < num; i++ ) {
    if( _sync_images_table[rank_set[i]] > 0 ) {
      checked++;
    }
  }

  if( checked == num ) return true;
  else                 return false;
}

static void _wait_sync_images(const int num, int *rank_set)
{
  while(1) {
    if( _check_sync_images_table( num, rank_set ) ) break;

    int ret;
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret == UTOFU_SUCCESS ) {
      _XMP_utofu_check_mrq_notice( &notice );
    }
    else if( ret != UTOFU_ERR_NOT_FOUND) {
      _XMP_utofu_error_check( ret );
    }
  }
}

void _XMP_utofu_sync_images(const int num, int* image_set, int* status)
{
  _XMP_utofu_sync_memory();

  if( num == 0 ) {
    return;
  }
  else if( num < 0 ) {
    fprintf(stderr, "Invalid value is used in xmp_sync_memory. The first argument is %d\n", num);
    _XMP_fatal_nomsg();
  }

  _notify_sync_images( num, image_set );
  _wait_sync_images( num, image_set );

  // post-processing
  for( int i = 0; i < num; i++ )
    _sync_images_table[image_set[i]]--;
}

