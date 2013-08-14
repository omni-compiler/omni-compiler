#include "xmpf_internal.h"

typedef struct _XMP_async_comm {
  int async_id;
  int nreqs;
  MPI_Request *reqs;
  struct _XMP_async_comm *next;
} _XMP_async_comm_t;

#define _XMP_ASYNC_COMM_SIZE 511

_XMP_async_comm_t _XMP_async_comm_tab[_XMP_ASYNC_COMM_SIZE] = { 0 };

#define _XMP_MAX_ASYNC_REQS (4 * _XMP_N_MAX_DIM * 10)


static void _XMPF_reflect_sched(_XMP_array_t **a_desc, int *lwidth, int *uwidth,
				int *is_periodic, int is_async);

#if !defined(OMNI_TARGET_CPU_KCOMPUTER) || !defined(K_RDMA_REFLECT)
static void _XMPF_reflect_normal_sched_dim(_XMP_array_t *adesc, int target_dim,
					   int lwidth, int uwidth, int is_periodic);
static void _XMPF_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim,
					  int lwidth, int uwidth, int is_periodic);
static void _XMPF_reflect_start(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic,
				int dummy);
static void _XMPF_reflect_wait(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic);
#else
static void _XMPF_reflect_rdma_sched_dim(_XMP_array_t *adesc, int target_dim,
					 int lwidth, int uwidth, int is_periodic);
static void _XMPF_reflect_start(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic,
				 int tag);
static void _XMPF_reflect_wait(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic);
static void _XMPF_wait_async_rdma(_XMP_async_comm_t *async);
#endif

int _XMPF_get_owner_pos_BLOCK(_XMP_array_t *a, int dim, int index);

_XMP_async_comm_t *_XMPF_get_async(int async_id);
_XMP_async_comm_t *_XMPF_get_or_create_async(int async_id);
void _XMPF_pop_async(int async_id);

void _XMPF_reflect_pack(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic);
void _XMPF_reflect_unpack(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic);


//#define DBG 1

#ifdef _XMP_TIMING
double t0, t1;
#endif

extern int _xmp_reflect_pack_flag;

static int _xmpf_set_reflect_flag = 0;
static int _xmp_lwidth[_XMP_N_MAX_DIM];
static int _xmp_uwidth[_XMP_N_MAX_DIM];
static int _xmp_is_periodic[_XMP_N_MAX_DIM];


void xmpf_set_reflect__(_XMP_array_t **a_desc, int *dim, int *lwidth, int *uwidth,
			int *is_periodic)
{
  _xmpf_set_reflect_flag = 1;
  _xmp_lwidth[*dim] = *lwidth;
  _xmp_uwidth[*dim] = *uwidth;
  _xmp_is_periodic[*dim] = *is_periodic;
}

//double t0, t_sched = 0, t_start = 0, t_wait = 0;

void xmpf_reflect__(_XMP_array_t **a_desc)
{

  _XMP_RETURN_IF_SINGLE;

  _XMP_array_t *a = *a_desc;

  if (!a->is_allocated){
    _xmpf_set_reflect_flag = 0;
    return;
  }

  if (!_xmpf_set_reflect_flag){
    for (int i = 0; i < a->dim; i++){
      _XMP_array_info_t *ai = &(a->info[i]);
      _xmp_lwidth[i] = ai->shadow_size_lo;
      _xmp_uwidth[i] = ai->shadow_size_hi;
      _xmp_is_periodic[i] = 0;
    }
  }

  //  t0 = MPI_Wtime();
  _XMPF_reflect_sched(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 0);
  //  t_sched = t_sched + (MPI_Wtime() - t0);

  //  t0 = MPI_Wtime();
  _XMPF_reflect_start(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 0);
  //  t_start = t_start + (MPI_Wtime() - t0);

  //  t0 = MPI_Wtime();
  _XMPF_reflect_wait(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic);
  //  t_wait = t_wait + (MPI_Wtime() - t0);

  _xmpf_set_reflect_flag = 0;

}


void xmpf_wait_async__(int *async_id)
{
  _XMP_async_comm_t *async;

  if (!(async = _XMPF_get_async(*async_id))) _XMP_fatal("wrong async-id");

  int nreqs = async->nreqs;;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  if (nreqs && !reqs){
    _XMPF_wait_async_rdma(async);
    return;
  }
#endif

  _XMP_TSTART(t0);
  MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
  _XMP_TEND(xmptiming_.t_wait, t0);

  _XMPF_pop_async(*async_id);

}


int xmp_test_async_(int *async_id)
{
  _XMP_async_comm_t *async;

  if (!(async = _XMPF_get_async(*async_id))) _XMP_fatal("wrong async-id");

  int nreqs = async->nreqs;
  MPI_Request *reqs = async->reqs;

#if defined(OMNI_TARGET_CPU_KCOMPUTER) && defined(K_RDMA_REFLECT)
  // For RDMA reflects, async->nreqs > 0 and async->reqs == NULL.
  _XMP_fatal("xmp_test_async not supported for RDMA.");
  /* if (nreqs && !reqs){ */
  /*   _XMPF_test_async_rdma(async); */
  /*   return; */
  /* } */
#endif

  int flag;
  MPI_Testall(nreqs, reqs, &flag, MPI_STATUSES_IGNORE);

  if (flag){
    _XMPF_pop_async(*async_id);
    return 1;
  }
  else {
    return 0;
  }

}


static void _XMPF_reflect_sched(_XMP_array_t **a_desc, int *lwidth, int *uwidth,
				int *is_periodic, int is_async)
{
  _XMP_array_t *a = *a_desc;

  _XMP_TSTART(t0);
  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NONE){
      continue;
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      _XMP_reflect_sched_t *reflect = ai->reflect_sched;

      if (lwidth[i] || uwidth[i]){
	if (lwidth[i] != reflect->lo_width ||
	    uwidth[i] != reflect->hi_width ||
	    is_periodic[i] != reflect->is_periodic){

	  reflect->lo_width = lwidth[i];
	  reflect->hi_width = uwidth[i];
	  reflect->is_periodic = is_periodic[i];

#if !defined(OMNI_TARGET_CPU_KCOMPUTER) || !defined(K_RDMA_REFLECT)
	  if (_xmp_reflect_pack_flag && !is_async){
	    _XMPF_reflect_pcopy_sched_dim(a, i, lwidth[i], uwidth[i], is_periodic[i]);
	  }
	  else {
	    _XMPF_reflect_normal_sched_dim(a, i, lwidth[i], uwidth[i], is_periodic[i]);
	  }
#else
	  _XMPF_reflect_rdma_sched_dim(a, i, lwidth[i], uwidth[i], is_periodic[i]);
#endif

	}
      }

    }
    else { /* _XMP_N_SHADOW_FULL */
      _XMP_fatal("xmpf_reflect: not surpport full shadow");
    }
    
  }
  _XMP_TEND(xmptiming_.t_sched, t0);

}


#if !defined(OMNI_TARGET_CPU_KCOMPUTER) || !defined(K_RDMA_REFLECT)

//
// Reflect without RDMA
//

static void _XMPF_reflect_normal_sched_dim(_XMP_array_t *adesc, int target_dim,
					   int lwidth, int uwidth, int is_periodic){

  if (lwidth == 0 && uwidth == 0) return;

  _XMP_array_info_t *ai = &(adesc->info[target_dim]);
  _XMP_array_info_t *ainfo = adesc->info;
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  if (lwidth > ai->shadow_size_lo || uwidth > ai->shadow_size_hi){
    _XMP_fatal("reflect width is larger than shadow width.");
  }

  _XMP_reflect_sched_t *reflect = ai->reflect_sched;

  int target_tdim = ai->align_template_index;
  _XMP_nodes_info_t *ni = adesc->align_template->chunk[target_tdim].onto_nodes_info;

  int ndims = adesc->dim;

  // 0-origin
  int my_pos = ni->rank;
  int lb_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_lower);
  int ub_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_upper);

  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;

  MPI_Comm *comm = adesc->align_template->onto_nodes->comm;
  int my_rank = adesc->align_template->onto_nodes->comm_rank;

  int lo_rank = my_rank + (lo_pos - my_pos) * ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * ni->multiplier;

  int type_size = adesc->type_size;

  void *lo_recv_buf = adesc->array_addr_p;
  void *lo_send_buf = adesc->array_addr_p;
  void *hi_recv_buf = adesc->array_addr_p;
  void *hi_send_buf = adesc->array_addr_p;

  //
  // setup MPI_data_type
  //

  int count = 1;
  int blocklength = type_size;
  int stride = ainfo[0].alloc_size * type_size;

  for (int i = ndims - 2; i >= target_dim; i--){
    count *= ainfo[i+1].alloc_size;
  }

  for (int i = 1; i <= target_dim; i++){
    blocklength *= ainfo[i-1].alloc_size;
    stride *= ainfo[i].alloc_size;
  }

  // for lower reflect

  if (reflect->datatype_lo != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_lo);
  }

  if (lwidth){
    MPI_Type_vector(count, blocklength * lwidth, stride,
		    MPI_BYTE, &reflect->datatype_lo);
    MPI_Type_commit(&reflect->datatype_lo);
  }
  else {
    reflect->datatype_lo = MPI_BYTE; // dummy
  }

  // for upper reflect

  if (reflect->datatype_hi != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_hi);
  }

  if (uwidth){
    MPI_Type_vector(count, blocklength * uwidth, stride,
		    MPI_BYTE, &reflect->datatype_hi);
    MPI_Type_commit(&reflect->datatype_hi);
  }
  else {
    reflect->datatype_hi = MPI_BYTE; // dummy
  }

  //
  // calculate base address
  //

  // for lower reflect

  if (lwidth){
      
    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_upper - lwidth + 1;
	lb_recv = ainfo[i].shadow_size_lo - lwidth;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_buf = (void *)((char *)lo_send_buf + lb_send * dim_acc * type_size);
      lo_recv_buf = (void *)((char *)lo_recv_buf + lb_recv * dim_acc * type_size);
      
    }

  }

  // for upper reflect

  if (uwidth){

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_buf = (void *)((char *)hi_send_buf + lb_send * dim_acc * type_size);
      hi_recv_buf = (void *)((char *)hi_recv_buf + lb_recv * dim_acc * type_size);
      
    }

  }

  //
  // initialize communication
  //

  int src, dst;

  if (!is_periodic && my_pos == lb_pos){ // no periodic
    lo_rank = MPI_PROC_NULL;
  }

  if (!is_periodic && my_pos == ub_pos){ // no periodic
    hi_rank = MPI_PROC_NULL;
  }

  // for lower reflect

  if (lwidth){
    src = lo_rank;
    dst = hi_rank;
  }
  else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[0] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[0]);
  }
	
  if (reflect->req[1] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[1]);
  }

  MPI_Recv_init(lo_recv_buf, 1, reflect->datatype_lo, src,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[0]);
  MPI_Send_init(lo_send_buf, 1, reflect->datatype_lo, dst,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[1]);

  // for upper reflect

  if (uwidth){
    src = hi_rank;
    dst = lo_rank;
  }
  else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[2] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[2]);
  }
	
  if (reflect->req[3] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[3]);
  }

  MPI_Recv_init(hi_recv_buf, 1, reflect->datatype_hi, src,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[2]);
  MPI_Send_init(hi_send_buf, 1, reflect->datatype_hi, dst,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[3]);

}


static void _XMPF_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim,
					  int lwidth, int uwidth, int is_periodic){

  if (lwidth == 0 && uwidth == 0) return;

  _XMP_array_info_t *ai = &(adesc->info[target_dim]);
  _XMP_array_info_t *ainfo = adesc->info;
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  if (lwidth > ai->shadow_size_lo || uwidth > ai->shadow_size_hi){
    _XMP_fatal("reflect width is larger than shadow width.");
  }

  _XMP_reflect_sched_t *reflect = ai->reflect_sched;

  int target_tdim = ai->align_template_index;
  _XMP_nodes_info_t *ni = adesc->align_template->chunk[target_tdim].onto_nodes_info;

  int ndims = adesc->dim;

  // 0-origin
  int my_pos = ni->rank;
  int lb_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_lower);
  int ub_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_upper);

  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;

  MPI_Comm *comm = adesc->align_template->onto_nodes->comm;
  int my_rank = adesc->align_template->onto_nodes->comm_rank;

  int lo_rank = my_rank + (lo_pos - my_pos) * ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * ni->multiplier;

  int type_size = adesc->type_size;
  void *array_addr = adesc->array_addr_p;

  void *lo_send_array, *lo_recv_array;
  void *hi_send_array, *hi_recv_array;

  void *lo_send_buf = array_addr;
  void *lo_recv_buf = array_addr;
  void *hi_send_buf = array_addr;
  void *hi_recv_buf = array_addr;

  int lo_buf_size = 0;
  int hi_buf_size = 0;

  //
  // setup data_type
  //

  int count = 1;
  int blocklength = type_size;
  int stride = ainfo[0].alloc_size * type_size;

  for (int i = ndims - 2; i >= target_dim; i--){
    count *= ainfo[i+1].alloc_size;
  }

  for (int i = 1; i <= target_dim; i++){
    blocklength *= ainfo[i-1].alloc_size;
    stride *= ainfo[i].alloc_size;
  }

  //
  // calculate base address
  //

  // for lower reflect

  if (lwidth){

    lo_send_array = array_addr;
    lo_recv_array = array_addr;

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_upper - lwidth + 1;
	lb_recv = ainfo[i].shadow_size_lo - lwidth;;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_array = (void *)((char *)lo_send_array + lb_send * dim_acc * type_size);
      lo_recv_array = (void *)((char *)lo_recv_array + lb_recv * dim_acc * type_size);

    }

  }

  // for upper reflect

  if (uwidth){

    hi_send_array = array_addr;
    hi_recv_array = array_addr;

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv, dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_array = (void *)((char *)hi_send_array + lb_send * dim_acc * type_size);
      hi_recv_array = (void *)((char *)hi_recv_array + lb_recv * dim_acc * type_size);

    }

  }

  //
  // Allocate buffers

  // for lower reflect

  if (lwidth){

    lo_buf_size = lwidth * blocklength * count;

    if (target_dim == ndims - 1){
      lo_send_buf = lo_send_array;
      lo_recv_buf = lo_recv_array;
    }
    else {
      _XMP_TSTART(t0);
      lo_send_buf = _XMP_alloc(lo_buf_size);
      lo_recv_buf = _XMP_alloc(lo_buf_size);
      _XMP_TEND2(xmptiming_.t_mem, xmptiming_.tdim_mem[target_dim], t0);
    }

  }

  // for upper reflect

  if (uwidth){

    hi_buf_size = uwidth * blocklength * count;

    if (target_dim == ndims - 1){
      hi_send_buf = hi_send_array;
      hi_recv_buf = hi_recv_array;
    }
    else {
      _XMP_TSTART(t0);
      hi_send_buf = _XMP_alloc(hi_buf_size);
      hi_recv_buf = _XMP_alloc(hi_buf_size);
      _XMP_TEND2(xmptiming_.t_mem, xmptiming_.tdim_mem[target_dim], t0);
    }

  }

  //
  // initialize communication
  //

  int src, dst;

  if (!is_periodic && my_pos == lb_pos){ // no periodic
    lo_rank = MPI_PROC_NULL;
  }

  if (!is_periodic && my_pos == ub_pos){ // no periodic
    hi_rank = MPI_PROC_NULL;
  }

  // for lower shadow

  if (lwidth){
    src = lo_rank;
    dst = hi_rank;
  }
  else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[0] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[0]);
  }
	
  if (reflect->req[1] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[1]);
  }

  MPI_Recv_init(lo_recv_buf, lo_buf_size, MPI_BYTE, src,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[0]);
  MPI_Send_init(lo_send_buf, lo_buf_size, MPI_BYTE, dst,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[1]);

  // for upper shadow

  if (uwidth){
    src = hi_rank;
    dst = lo_rank;
  }
  else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[2] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[2]);
  }
	
  if (reflect->req[3] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[3]);
  }

  MPI_Recv_init(hi_recv_buf, hi_buf_size, MPI_BYTE, src,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[2]);
  MPI_Send_init(hi_send_buf, hi_buf_size, MPI_BYTE, dst,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[3]);

  //
  // cache schedule
  //

  reflect->count = count;
  reflect->blocklength = blocklength;
  reflect->stride = stride;

  reflect->lo_send_array = lo_send_array;
  reflect->lo_recv_array = lo_recv_array;
  reflect->hi_send_array = hi_send_array;
  reflect->hi_recv_array = hi_recv_array;

  reflect->lo_send_buf = lo_send_buf;
  reflect->lo_recv_buf = lo_recv_buf;
  reflect->hi_send_buf = hi_send_buf;
  reflect->hi_recv_buf = hi_recv_buf;

}


static void _XMPF_reflect_start(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic,
				int dummy)
{

  _XMP_array_t *a = *a_desc;

  if (_xmp_reflect_pack_flag){
    _XMP_TSTART(t0);
    _XMPF_reflect_pack(a, lwidth, uwidth, is_periodic);
    _XMP_TEND(xmptiming_.t_copy, t0);
  }

  for (int i = 0; i < a->dim; i++){

    if (!lwidth[i] && !uwidth[i]) continue;

    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;

    _XMP_TSTART(t0);
    MPI_Startall(4, reflect->req);
    _XMP_TEND2(xmptiming_.t_comm, xmptiming_.tdim_comm[i], t0);

  }

}


static void _XMPF_reflect_wait(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic)
{
  _XMP_array_t *a = *a_desc;

  for (int i = 0; i < a->dim; i++){

    if (!lwidth[i] && !uwidth[i]) continue;

    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;

    _XMP_TSTART(t0);
    MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
    _XMP_TEND2(xmptiming_.t_wait, xmptiming_.tdim_wait[i], t0);

  }

  if (_xmp_reflect_pack_flag){
    _XMP_TSTART(t0);
    _XMPF_reflect_unpack(a, lwidth, uwidth, is_periodic);
    _XMP_TEND(xmptiming_.t_copy, t0);
  }

}


void xmpf_reflect_async__(_XMP_array_t **a_desc, int *async_id)
{

  int nreqs = 0;
  MPI_Request *reqs;

  _XMP_array_t *a = *a_desc;

  if (!a->is_allocated){
    _xmpf_set_reflect_flag = 0;
    return;
  }

  if (!_xmpf_set_reflect_flag){
    for (int i = 0; i < a->dim; i++){
      _XMP_array_info_t *ai = &(a->info[i]);
      _xmp_lwidth[i] = ai->shadow_size_lo;
      _xmp_uwidth[i] = ai->shadow_size_hi;
      _xmp_is_periodic[i] = 0;
    }
  }
    
  _XMPF_reflect_sched(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 1);

  //
  // Start Comm.
  //

  _XMP_async_comm_t *async = _XMPF_get_or_create_async(*async_id);

  reqs = &async->reqs[async->nreqs];

  for (int i = 0; i < a->dim; i++){
    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;
    if (_xmp_lwidth[i] || _xmp_uwidth[i]){
      if (async->nreqs + nreqs + 4 > _XMP_MAX_ASYNC_REQS){
	_XMP_fatal("too many arrays in an asynchronous reflect");
      }
      memcpy(&reqs[nreqs], reflect->req, 4 * sizeof(MPI_Request));
      nreqs += 4;
    }
  }

  async->nreqs += nreqs;

  _XMP_TSTART(t0);
  MPI_Startall(nreqs, reqs);
  _XMP_TEND(xmptiming_.t_start, t0);

  _xmpf_set_reflect_flag = 0;

}


#else

//
// Reflect with RDMA
//

static void _XMPF_reflect_rdma_sched_dim(_XMP_array_t *adesc, int target_dim,
					 int lwidth, int uwidth, int is_periodic){

  if (lwidth == 0 && uwidth == 0) return;

  _XMP_array_info_t *ai = &(adesc->info[target_dim]);
  _XMP_array_info_t *ainfo = adesc->info;
  _XMP_ASSERT(ai->align_manner == _XMP_N_ALIGN_BLOCK);
  _XMP_ASSERT(ai->is_shadow_comm_member);

  if (lwidth > ai->shadow_size_lo || uwidth > ai->shadow_size_hi){
    _XMP_fatal("reflect width is larger than shadow width.");
  }

  _XMP_reflect_sched_t *reflect = ai->reflect_sched;

  int target_tdim = ai->align_template_index;
  _XMP_nodes_info_t *ni = adesc->align_template->chunk[target_tdim].onto_nodes_info;

  int ndims = adesc->dim;

  // 0-origin
  int my_pos = ni->rank;
  int lb_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_lower);
  int ub_pos = _XMPF_get_owner_pos_BLOCK(adesc, target_dim, ai->ser_upper);

  int my_rank = adesc->align_template->onto_nodes->comm_rank;

  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;

  int lo_rank = my_rank + (lo_pos - my_pos) * ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * ni->multiplier;

  int type_size = adesc->type_size;

  void *lo_send_array, *lo_recv_array;
  void *hi_send_array, *hi_recv_array;

  uint64_t rdma_raddr;

  //
  // calculate offset
  //

  int count = 1;
  int blocklength = type_size;
  int stride = ainfo[0].alloc_size * type_size;

  for (int i = ndims - 2; i >= target_dim; i--){
    count *= ainfo[i+1].alloc_size;
  }

  for (int i = 1; i <= target_dim; i++){
    blocklength *= ainfo[i-1].alloc_size;
    stride *= ainfo[i].alloc_size;
  }
  
  //
  // calculate base address
  //

  // for lower reflect
    
  while ((rdma_raddr = FJMPI_Rdma_get_remote_addr(hi_rank, adesc->rdma_memid)) == FJMPI_RDMA_ERROR);

  if (lwidth){
    
    lo_send_array = (void *)adesc->rdma_addr;
    lo_recv_array = (void *)rdma_raddr;

    for (int i = 0; i < ndims; i++) {
      
      int lb_send, lb_recv, dim_acc;
      
      if (i == target_dim) {
	lb_send = ainfo[i].local_upper - lwidth + 1;
	lb_recv = ainfo[i].shadow_size_lo - lwidth;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }
      
      dim_acc = ainfo[i].dim_acc;
      
      lo_send_array = (void *)((uint64_t)lo_send_array + lb_send * dim_acc * type_size);
      lo_recv_array = (void *)((uint64_t)lo_recv_array + lb_recv * dim_acc * type_size);
      
    }
    
  }
  
  // for upper reflect
  
  while ((rdma_raddr = FJMPI_Rdma_get_remote_addr(lo_rank, adesc->rdma_memid)) == FJMPI_RDMA_ERROR);

  if (uwidth){
    
    hi_send_array = (void *)adesc->rdma_addr;
    hi_recv_array = (void *)rdma_raddr;

    for (int i = 0; i < ndims; i++) {
      
      int lb_send, lb_recv, dim_acc;
      
      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      }
      else {
	// Note: including shadow area
	lb_send = 0;
	lb_recv = 0;
      }
      
      dim_acc = ainfo[i].dim_acc;
	
      hi_send_array = (void *)((uint64_t)hi_send_array + lb_send * dim_acc * type_size);
      hi_recv_array = (void *)((uint64_t)hi_recv_array + lb_recv * dim_acc * type_size);
      
    }
      
  }
  
  //
  // cache schedule
  //

  if (!is_periodic && my_pos == lb_pos){ // no periodic
    lo_rank = -1;
  }

  if (!is_periodic && my_pos == ub_pos){ // no periodic
    hi_rank = -1;
  }

  reflect->count = count;
  reflect->blocklength = blocklength;
  reflect->stride = stride;

  reflect->lo_send_array = lo_send_array;
  reflect->lo_recv_array = lo_recv_array;
  reflect->hi_send_array = hi_send_array;
  reflect->hi_recv_array = hi_recv_array;

  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;

}


/* static void _XMPF_reflect_sync(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic, */
/* 			       int tag) */
/* { */
/*   _XMP_array_t *a = *a_desc; */

/*   xmpf_dbg_printf("_XMPF_reflect_sync starts\n"); */

/*   int nrdmas0 = 0, nrdmas1 = 0, nrdmas2 = 0, nrdmas3 = 0; */

/*   for (int i = 0; i < a->dim; i++){ */

/*     _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched; */

/*     // for lower reflect */

/*     if (lwidth[i] && reflect->hi_rank != -1){ */

/*       int hi_rank = reflect->hi_rank; */

/*       uint64_t lo_recv_array = (uint64_t)reflect->lo_recv_array; */
/*       uint64_t lo_send_array = (uint64_t)reflect->lo_send_array; */

/*       FJMPI_Rdma_put(hi_rank, tag, */
/* 		     lo_recv_array, lo_send_array, */
/* 		     0, */
/* 		     FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC2 | FJMPI_RDMA_REMOTE_NOTICE); */

/*       nrdmas0++; */
/*     } */

/*     if (lwidth[i] && reflect->lo_rank != -1) nrdmas2++; */

/*     // for upper reflect */

/*     if (uwidth[i] && reflect->lo_rank != -1){ */

/*       int lo_rank = reflect->lo_rank; */

/*       uint64_t hi_recv_array = (uint64_t)reflect->hi_recv_array; */
/*       uint64_t hi_send_array = (uint64_t)reflect->hi_send_array; */

/*       FJMPI_Rdma_put(lo_rank, tag, */
/* 		     hi_recv_array, hi_send_array, */
/* 		     0, */
/* 		     FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC3 | FJMPI_RDMA_REMOTE_NOTICE); */

/*       nrdmas1++; */
/*     } */

/*     if (uwidth[i] && reflect->hi_rank != -1) nrdmas3++; */

/*   } */

/*   // flush send completion */
/*   while (nrdmas0 || nrdmas1){ */
/*     xmpf_dbg_printf("nrdmas0 = %d, nrdmas1 = %d, nrdmas2 = %d, nrdmas3 = %d\n", */
/* 		    nrdmas0, nrdmas1, nrdmas2, nrdmas3); */
/*     while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, NULL) == FJMPI_RDMA_NOTICE){ */
/*       nrdmas0--; */
/*     } */
/*     while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, NULL) == FJMPI_RDMA_NOTICE){ */
/*       nrdmas1--; */
/*     } */
/*   } */

/*   // check receive completion */
/*   while (nrdmas2 || nrdmas3){ */
/*     xmpf_dbg_printf("nrdmas0 = %d, nrdmas1 = %d, nrdmas2 = %d, nrdmas3 = %d\n", */
/* 		    nrdmas0, nrdmas1, nrdmas2, nrdmas3); */
/*     while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC2, NULL) == FJMPI_RDMA_HALFWAY_NOTICE){ */
/*       nrdmas2--; */
/*     } */
/*     while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC3, NULL) == FJMPI_RDMA_HALFWAY_NOTICE){ */
/*       nrdmas3--; */
/*     } */
/*   } */

/*   xmpf_dbg_printf("_XMPF_reflect_sync ends\n"); */

/* } */


static void _XMPF_reflect_start(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic,
				int tag)
{
  _XMP_array_t *a = *a_desc;

  _XMP_TSTART(t1);

  xmp_barrier();

  for (int i = 0; i < a->dim; i++){

    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;

    _XMP_TSTART(t0);

    // for lower reflect

    if (lwidth[i] && reflect->hi_rank != -1){
      for (int j = 0; j < reflect->count; j++){
	FJMPI_Rdma_put(reflect->hi_rank, tag,
		       (uint64_t)reflect->lo_recv_array + j * reflect->stride,
		       (uint64_t)reflect->lo_send_array + j * reflect->stride,
		       lwidth[i] * reflect->blocklength,
		       FJMPI_RDMA_LOCAL_NIC0 | FJMPI_RDMA_REMOTE_NIC2);
      }
    }

    // for upper reflect

    if (uwidth[i] && reflect->lo_rank != -1){
      for (int j = 0; j < reflect->count; j++){
	FJMPI_Rdma_put(reflect->lo_rank, tag,
		       (uint64_t)reflect->hi_recv_array + j * reflect->stride,
		       (uint64_t)reflect->hi_send_array + j * reflect->stride,
		       uwidth[i] * reflect->blocklength,
		       FJMPI_RDMA_LOCAL_NIC1 | FJMPI_RDMA_REMOTE_NIC3);
      }
    }

    _XMP_TEND(xmptiming_.tdim_comm[i], t0);

  }

  _XMP_TEND(xmptiming_.t_comm, t1);

}


static void _XMPF_reflect_wait(_XMP_array_t **a_desc, int *lwidth, int *uwidth, int *is_periodic)
{

  _XMP_array_t *a = *a_desc;
  int nrdmas0 = 0, nrdmas1 = 0;

  _XMP_TSTART(t0);

  for (int i = 0; i < a->dim; i++){
    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;
    if (lwidth[i] && reflect->hi_rank != -1) nrdmas0 += reflect->count;
    if (uwidth[i] && reflect->lo_rank != -1) nrdmas1 += reflect->count;
  }

  while (nrdmas0 || nrdmas1){
    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, NULL) == FJMPI_RDMA_NOTICE){
      nrdmas0--;
    }
    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, NULL) == FJMPI_RDMA_NOTICE){
      nrdmas1--;
    }
  }

  xmp_barrier();

  _XMP_TEND(xmptiming_.t_wait, t0);

}


void xmpf_reflect_async__(_XMP_array_t **a_desc, int *async_id)
{

  _XMP_array_t *a = *a_desc;

  if (!a->is_allocated){
    _xmpf_set_reflect_flag = 0;
    return;
  }

  if (!_xmpf_set_reflect_flag){
    for (int i = 0; i < a->dim; i++){
      _XMP_array_info_t *ai = &(a->info[i]);
      _xmp_lwidth[i] = ai->shadow_size_lo;
      _xmp_uwidth[i] = ai->shadow_size_hi;
      _xmp_is_periodic[i] = 0;
    }
  }
    
  _XMPF_reflect_sched(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 1);
  _XMPF_reflect_start(a_desc, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, *async_id);

  _XMP_async_comm_t *async = _XMPF_get_or_create_async(*async_id);
  _XMP_free(async->reqs); async->reqs = NULL; // reqs not needed in RDMA reflects.

  for (int i = 0; i < a->dim; i++){
    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;
    if (_xmp_lwidth[i] && reflect->hi_rank != -1) async->nreqs += reflect->count;
    if (_xmp_uwidth[i] && reflect->lo_rank != -1) async->nreqs += reflect->count;
  }

  _xmpf_set_reflect_flag = 0;

}


static void _XMPF_wait_async_rdma(_XMP_async_comm_t *async)
{
  int nreqs = async->nreqs;
  int async_id = async->async_id;

  _XMP_async_comm_t *async1;

  struct FJMPI_Rdma_cq cq;

  _XMP_TSTART(t0);

  while (nreqs){

    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC0, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else {
      //      if (!(async1 = _XMPF_get_async(cq.tag))) _XMP_fatal("wrong async-id");
      async1 = _XMPF_get_or_create_async(cq.tag);
      async1->nreqs--;
    }

    while (FJMPI_Rdma_poll_cq(FJMPI_RDMA_NIC1, &cq) != FJMPI_RDMA_NOTICE);
    if (cq.tag == async_id){
      nreqs--;
    }
    else {
      //      if (!(async1 = _XMPF_get_async(cq.tag))) _XMP_fatal("wrong async-id");
      async1 = _XMPF_get_or_create_async(cq.tag);
      async1->nreqs--;
    }

  }

  _XMP_TEND(xmptiming_.t_wait, t0);

  _XMPF_pop_async(async_id);

  xmp_barrier();

}

#endif


int _XMPF_get_owner_pos_BLOCK(_XMP_array_t *a, int dim, int index){

  _XMP_ASSERT(a->info[dim].align_manner == _XMP_N_ALIGN_BLOCK);

  int align_offset = a->info[dim].align_subscript;

  int tdim = a->info[dim].align_template_index;
  int tlb = a->align_template->info[tdim].ser_lower;
  int chunk = a->align_template->chunk[tdim].par_chunk_width;

  int pos = (index + align_offset - tlb) / chunk;

  return pos;
}


//
// for Asynchronous Communication
//

/* void _XMPF_set_async(int nreqs, MPI_Request *reqs, int async_id) */
/* { */
/*   int hash = async_id % _XMP_ASYNC_COMM_SIZE; */
/*   _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash]; */

/*   if (async->nreqs == 0){ */
/*     async->async_id = async_id; */
/*     async->nreqs = nreqs; */
/*     async->reqs = reqs; */
/*     async->next = NULL; */
/*   } */
/*   else { */
/*     while (async->next) async = async->next; */
/*     _XMP_async_comm_t *new_async = _XMP_alloc(sizeof(_XMP_async_comm_t)); */
/*     new_async->async_id = async_id; */
/*     new_async->nreqs = nreqs; */
/*     new_async->reqs = reqs; */
/*     new_async->next = NULL; */
/*     async->next = new_async; */
/*   } */

/* } */


_XMP_async_comm_t *_XMPF_get_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  if (async->nreqs != 0){
    if (async->async_id == async_id){
      return async;
    }
    else {
      while (async->next){
	async = async->next;
	if (async->async_id == async_id){
	  return async;
	}
      }
    }

  }

  return NULL;

}


_XMP_async_comm_t *_XMPF_get_or_create_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  if (async->nreqs != 0){
    if (async->async_id == async_id){
      return async;
    }
    else {

      while (async->next){
	async = async->next;
	if (async->async_id == async_id){
	  return async;
	}
      }

      async->next = _XMP_alloc(sizeof(_XMP_async_comm_t));
      async = async->next;

    }

  }

  async->async_id = async_id;
  async->nreqs = 0;
  async->reqs = _XMP_alloc(sizeof(MPI_Request) * _XMP_MAX_ASYNC_REQS);
  async->next = NULL;

  return async;

}


void _XMPF_pop_async(int async_id)
{
  int hash = async_id % _XMP_ASYNC_COMM_SIZE;
  _XMP_async_comm_t *async = &_XMP_async_comm_tab[hash];

  // The case no comm. registered for async_id 0 and _XMPF_pop_async called for 
  // async_id == 0, may occur and is inconsistent.
  // But, actually, the code below works without problems in such a case, even if
  // no hash == 0.
  if (async->async_id == async_id){

    if (async->next){
      _XMP_async_comm_t *t = async->next;
      async->async_id = t->async_id;
      async->nreqs = t->nreqs;
      async->reqs = t->reqs;
      async->next = t->next;
      _XMP_free(t);
    }
    else {
      async->nreqs = 0;
      _XMP_free(async->reqs);
    }

    return;

  }
  else {

    _XMP_async_comm_t *prev = async;

    while (async = prev->next){

      if (async->async_id == async_id){
	prev->next = async->next;
	_XMP_free(async->reqs);
	_XMP_free(async);
	return;
      }
      
      prev = async;

    }

  }

  _XMP_fatal("internal error: inconsistent async table");

}


void _XMPF_reflect_pack(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic)
{

  for (int i = 0; i < a->dim - 1; i++){

    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;

    // for lower reflect
    if (lwidth[i]){
      _XMPF_pack_vector((char *)reflect->lo_send_buf,
			(char *)reflect->lo_send_array,
			reflect->count, lwidth[i] * reflect->blocklength,
			reflect->stride);
    }

    // for upper reflect
    if (uwidth[i]){
      _XMPF_pack_vector((char *)reflect->hi_send_buf,
			(char *)reflect->hi_send_array,
			reflect->count, uwidth[i] * reflect->blocklength,
			reflect->stride);
    }

  }

}


void _XMPF_reflect_unpack(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic)
{

  for (int i = 0; i < a->dim - 1; i++){

    _XMP_reflect_sched_t *reflect = a->info[i].reflect_sched;

    // for lower reflect
    if (lwidth[i]){
      _XMPF_unpack_vector((char *)reflect->lo_recv_array,
			  (char *)reflect->lo_recv_buf,
			  reflect->count, lwidth[i] * reflect->blocklength,
			  reflect->stride);
    }

    // for upper reflect
    if (uwidth[i]){
      _XMPF_unpack_vector((char *)reflect->hi_recv_array,
			  (char *)reflect->hi_recv_buf,
			  reflect->count, uwidth[i] * reflect->blocklength,
			  reflect->stride);
    }

  }

}
