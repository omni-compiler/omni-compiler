#include "xmp_internal.h"
#include <stdlib.h>

typedef struct _XMP_postreq_info{
  int tag;
  utofu_vcq_id_t vcqid;
} _XMP_postreq_info_t;

typedef struct _XMP_postreq{
  _XMP_postreq_info_t *table;   /**< Table for post requests */
  int                 num;      /**< How many post requests are in table */
  int                 max_size; /**< Max size of table */
} _XMP_postreq_t;

static _XMP_postreq_t _postreq;
static utofu_stadd_t _lcl_stadd;
static utofu_stadd_t *_rmt_stadds;
static int _num_of_puts = 0;

void _xmp_utofu_post_wait_initialize(void)
{
  _postreq.num      = 0;
  _postreq.max_size = _XMP_POSTREQ_TABLE_INITIAL_SIZE;
  _postreq.table    = malloc(sizeof(_XMP_postreq_info_t) * _postreq.max_size);

  double *token     = _XMP_alloc(sizeof(double));
  utofu_reg_mem(_xmp_utofu_vcq_hdl, token, sizeof(double), 0, &_lcl_stadd);
  _rmt_stadds       = _XMP_alloc(sizeof(utofu_stadd_t) * _XMP_world_size);
  MPI_Allgather(&_lcl_stadd, 1, MPI_UINT64_T, _rmt_stadds, 1, MPI_UINT64_T, MPI_COMM_WORLD);
}

void _xmp_utofu_add_postreq(const utofu_vcq_id_t vcqid, const int tag)
{
  if( _postreq.num == _postreq.max_size ) {
    _postreq.max_size *= _XMP_POSTREQ_TABLE_INCREMENT_RATIO;
    size_t next_size = sizeof(_XMP_postreq_info_t) * _postreq.max_size;
    _XMP_postreq_info_t *tmp;
    if((tmp = realloc(_postreq.table, next_size)) == NULL) {
      free(_postreq.table);
      _XMP_fatal("add_postreq : cannot allocate memory : _postreq.table");
    }
    else{
      _postreq.table = tmp;
    }
  }

  _postreq.table[_postreq.num].vcqid = vcqid;
  _postreq.table[_postreq.num].tag   = tag;
  _postreq.num++;
}

void _xmp_utofu_post(const int node_num, const int tag)
{
  uint64_t taglimit = _xmp_utofu_edata_flag_sync_images;
  if( tag < 0 || tag >= taglimit ) {
    fprintf(stderr, "_xmp_utofu_post : tag is %d : tag must be in the range 0 <= tag < %lu\n", tag, taglimit);
    _XMP_fatal_nomsg();
  }

  _XMP_utofu_sync_memory();

  if(node_num == _XMP_world_rank){
    _xmp_utofu_add_postreq( _xmp_utofu_vcq_ids_org[node_num], tag );
  }
  else{
    int ret;
    utofu_vcq_id_t rmt_vcq_id = _xmp_utofu_vcq_ids[node_num];

    int ident_val = _num_of_puts;
    uint64_t edata = tag;
    unsigned long int post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE | UTOFU_ONESIDED_FLAG_REMOTE_MRQ_NOTICE | UTOFU_ONESIDED_FLAG_STRONG_ORDER;
    uintptr_t cbvalue = ident_val;
    _num_of_puts++;
    utofu_put(_xmp_utofu_vcq_hdl, rmt_vcq_id, _lcl_stadd, _rmt_stadds[node_num],
              sizeof(double), edata, post_flags, (void *)cbvalue);

    void *cbdata;
    do {
      ret = utofu_poll_tcq(_xmp_utofu_vcq_hdl, 0, &cbdata);
    } while (ret == UTOFU_ERR_NOT_FOUND);
    if( ret != UTOFU_SUCCESS ) {
      _XMP_fatal("_xmp_utofu_post : utofu_put, utofu_poll_tcq not success");
    }
  }
}

void _xmp_utofu_wait_noargs(void)
{
  _XMP_utofu_sync_memory();

  while(1) {
    if(_postreq.num > 0) {
      _postreq.num--;
      break;
    }
    int ret;
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret == UTOFU_SUCCESS ) {
      _XMP_utofu_check_mrq_notice( &notice );
    }
  }
}

static void shift_postreq(const int index)
{
  if(index != _postreq.num-1){  // Not last request
    for(int i=index+1;i<_postreq.num;i++){
      _postreq.table[i-1] = _postreq.table[i];
    }
  }
  _postreq.num--;
}

static bool remove_postreq_node(const utofu_vcq_id_t vcqid)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(vcqid == _postreq.table[i].vcqid){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

static bool remove_postreq(const utofu_vcq_id_t vcqid, const int tag)
{
  for(int i=_postreq.num-1;i>=0;i--){
    if(vcqid == _postreq.table[i].vcqid && tag == _postreq.table[i].tag){
      shift_postreq(i);
      return _XMP_N_INT_TRUE;
    }
  }
  return _XMP_N_INT_FALSE;
}

void _xmp_utofu_wait(const int node_num, const int tag)
{
  _XMP_utofu_sync_memory();

  while(1) {
    bool table_has_postreq = remove_postreq(_xmp_utofu_vcq_ids_org[node_num], tag);
    if(table_has_postreq) break;

    int ret;
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret == UTOFU_SUCCESS ) {
      _XMP_utofu_check_mrq_notice( &notice );
    }
  }
}

void _xmp_utofu_wait_node(const int node_num)
{
  _XMP_utofu_sync_memory();

  while(1) {
    bool table_has_postreq = remove_postreq_node(_xmp_utofu_vcq_ids_org[node_num]);
    if(table_has_postreq) break;

    int ret;
    struct utofu_mrq_notice notice;
    ret = utofu_poll_mrq(_xmp_utofu_vcq_hdl, 0, &notice);
    if( ret == UTOFU_SUCCESS ) {
      _XMP_utofu_check_mrq_notice( &notice );
    }
  }
}
