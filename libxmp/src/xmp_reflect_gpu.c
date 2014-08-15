#include "xmp_internal.h"
#include "include/cuda_runtime.h"
#include<stdio.h>
#include<stdlib.h>

void _XMP_reflect_do_gpu(_XMP_array_t *array_desc);
void _XMP_reflect_init_gpu(void *acc_addr, _XMP_array_t *array_desc);

static void _XMP_gpu_wait_async(void *);

void _XMP_gpu_pack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);
void _XMP_gpu_unpack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);

static void _XMP_reflect_gpu_unpack(_XMP_array_t *a);
static void _XMP_reflect_gpu_pack(_XMP_array_t *a);
static void _XMP_reflect_wait(_XMP_array_t *a);
static void _XMP_gpu_update_host_async(void *dst, void *src, size_t size, void* async_id);
static void _XMP_gpu_update_device_async(void *dst, void *src, size_t size, void* async_id);

static void _XMP_reflect_start(_XMP_array_t *a, int dummy);
static void _XMP_reflect_sched(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic, int is_async, void *dev_addr);
static void _XMP_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim,
					 int lwidth, int uwidth, int is_periodic, void *dev_array_addr);
static int _xmpf_set_reflect_flag = 0;
static int _xmp_lwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_uwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_is_periodic[_XMP_N_MAX_DIM] = {0};

void _XMP_gpu_update_host_async(void *dst, void *src, size_t size, void *async_id)
{
  cudaStream_t st = *((cudaStream_t*)async_id);
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, st);
}
void _XMP_gpu_update_device_async(void *dst, void *src, size_t size, void *async_id)
{
  cudaStream_t st = *((cudaStream_t*)async_id);
  cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, st);
}
void _XMP_gpu_wait_async(void *async_id)
{
  cudaStream_t st = *((cudaStream_t*)async_id);
  cudaStreamSynchronize(st);
}

void* _XMP_gpu_host_alloc(size_t size)
{
  void *p;
  cudaMallocHost((void**)&p, size);
  if(p == NULL){
    printf("null returned\n");
  }
  return p;
}
void _XMP_gpu_host_free(void *addr)
{
  cudaFreeHost(addr);
}

void _XMP_gpu_alloc(void **addr, size_t size) {
  if (cudaMalloc(addr, size) != cudaSuccess) {
    _XMP_fatal("failed to allocate data on GPU");
  }
}

void _XMP_gpu_free(void *addr) {
  if (cudaFree(addr) != cudaSuccess) {
    _XMP_fatal("failed to free data on GPU");
  }
}
void* _XMP_gpu_alloc_async_id()
{
  cudaStream_t *st = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(st);
  return st;
}

void _XMP_gpu_free_async_id(void *async_id)
{
  cudaStreamDestroy(*((cudaStream_t*)async_id));
  free(async_id);
}

void _XMP_reflect_init_gpu(void *dev_addr,_XMP_array_t *a)
{
  _XMP_RETURN_IF_SINGLE;
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

  _XMP_reflect_sched(a, _xmp_lwidth, _xmp_uwidth, _xmp_is_periodic, 0, dev_addr);

  _xmpf_set_reflect_flag = 0;
  for (int i = 0; i < a->dim; i++){
    _xmp_lwidth[i] = 0;
    _xmp_uwidth[i] = 0;
    _xmp_is_periodic[i] = 0;
  }
}

void _XMP_reflect_do_gpu(_XMP_array_t *array_desc){
  _XMP_RETURN_IF_SINGLE;

  _XMP_reflect_start(array_desc, 0);
  _XMP_reflect_wait(array_desc);
}


static void _XMP_reflect_sched(_XMP_array_t *a, int *lwidth, int *uwidth,
			       int *is_periodic, int is_async, void *dev_addr)
{
  _XMP_TSTART(t0);
  for (int i = 0; i < a->dim; i++){

    _XMP_array_info_t *ai = &(a->info[i]);

    if (ai->shadow_type == _XMP_N_SHADOW_NONE){
      continue;
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){

      _XMP_reflect_sched_t *reflect = ai->reflect_sched;

      if (lwidth[i] || uwidth[i]){

	_XMP_ASSERT(reflect);

	if (reflect->is_periodic == -1 /* not set yet */ ||
	    lwidth[i] != reflect->lo_width ||
	    uwidth[i] != reflect->hi_width ||
	    is_periodic[i] != reflect->is_periodic){

	  reflect->lo_width = lwidth[i];
	  reflect->hi_width = uwidth[i];
	  reflect->is_periodic = is_periodic[i];

	  if (/*_xmp_reflect_pack_flag && !is_async*/ 1){
	    _XMP_reflect_pcopy_sched_dim(a, i, lwidth[i], uwidth[i], is_periodic[i], dev_addr);
	  }
	  else {
	    //_XMP_reflect_normal_sched_dim(a, i, lwidth[i], uwidth[i], is_periodic[i]);
	  }
	}
      }

    }
    else { /* _XMP_N_SHADOW_FULL */
      ;
    }
    
  }
  _XMP_TEND(xmptiming_.t_sched, t0);

}

static void _XMP_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim,
					 int lwidth, int uwidth, int is_periodic, void *dev_array_addr){
  //printf("desc=%p, tardim=%d, lw=%d, uw=%d, devp=%p\n", adesc, target_dim, lwidth, uwidth, dev_array_addr);
  
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
  int lb_pos = _XMP_get_owner_pos(adesc, target_dim, ai->ser_lower);
  int ub_pos = _XMP_get_owner_pos(adesc, target_dim, ai->ser_upper);

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
  void *lo_send_dev_array, *lo_recv_dev_array;
  void *hi_send_dev_array, *hi_recv_dev_array;

  void *lo_send_buf = NULL;
  void *lo_recv_buf = NULL;
  void *hi_send_buf = NULL;
  void *hi_recv_buf = NULL;

  void *lo_send_dev_buf = NULL;
  void *lo_recv_dev_buf = NULL;
  void *hi_send_dev_buf = NULL;
  void *hi_recv_dev_buf = NULL;

  int lo_buf_size = 0;
  int hi_buf_size = 0;

  //
  // setup data_type
  //

  int count, blocklength;
  long long stride;
  int count_offset = 0;

  if (_XMPF_running & !_XMPC_running){ /* for XMP/F */

    count = 1;
    blocklength = type_size;
    stride = ainfo[0].alloc_size * type_size;

    for (int i = ndims - 2; i >= target_dim; i--){
      count *= ainfo[i+1].alloc_size;
    }

    for (int i = 1; i <= target_dim; i++){
      blocklength *= ainfo[i-1].alloc_size;
      stride *= ainfo[i].alloc_size;
    }

  }
  else if (!_XMPF_running & _XMPC_running){ /* for XMP/C */

    count = 1;
    blocklength = type_size;
    stride = ainfo[ndims-1].alloc_size * type_size;

    
    if(target_dim > 0){
      count *= ainfo[0].par_size;
      count_offset = ainfo[0].shadow_size_lo;
    }
    for (int i = 1; i < target_dim; i++){
      count *= ainfo[i].alloc_size;
    }

    for (int i = ndims - 2; i >= target_dim; i--){
      blocklength *= ainfo[i+1].alloc_size;
      stride *= ainfo[i].alloc_size;
    }

    /* for (int i = target_dim + 1; i < ndims; i++){ */
    /*   blocklength *= ainfo[i].alloc_size; */
    /* } */
    /* for (int i = target_dim; i < ndims - 1; i++){ */
    /*   stride *= ainfo[i].alloc_size; */
    /* } */

    //    printf("count =%d, blength=%d, stride=%d\n", count ,blocklength,stride);
    //    printf("ainfo[0].par_size=%d\n", ainfo[0].par_size);
    //    printf("count_ofset=%d,\n", count_offset);
  }
  else {
    _XMP_fatal("cannot determin the base language.");
  }

  //
  // calculate base address
  //

  // for lower reflect

  if (lwidth){

    lo_send_array = lo_recv_array = (void *)((char*)array_addr + count_offset * stride);
    lo_send_dev_array = lo_recv_dev_array = (void *)((char*)dev_array_addr + count_offset * stride);

    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv;
      unsigned long long dim_acc;

      if (i == target_dim) {
	//printf("ainfo[%d].local_upper=%d\n",i,ainfo[i].local_upper);
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
      lo_send_dev_array = (void *)((char *)lo_send_dev_array + lb_send * dim_acc * type_size);
      lo_recv_dev_array = (void *)((char *)lo_recv_dev_array + lb_recv * dim_acc * type_size);

    }

  }

  // for upper reflect

  if (uwidth){

    hi_send_array = hi_recv_array = (void *)((char*)array_addr + count_offset * stride);
    hi_send_dev_array = hi_recv_dev_array = (void *)((char*)dev_array_addr + count_offset * stride);


    for (int i = 0; i < ndims; i++) {

      int lb_send, lb_recv;
      unsigned long long dim_acc;

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
      hi_send_dev_array = (void *)((char *)hi_send_dev_array + lb_send * dim_acc * type_size);
      hi_recv_dev_array = (void *)((char *)hi_recv_dev_array + lb_recv * dim_acc * type_size);

    }

  }

  //
  // Allocate buffers
  //

  if ((_XMPF_running && target_dim != ndims - 1) ||
      (_XMPC_running && target_dim != 0)){
    _XMP_gpu_host_free(reflect->lo_send_buf);
    _XMP_gpu_host_free(reflect->lo_recv_buf);
    _XMP_gpu_host_free(reflect->hi_send_buf);
    _XMP_gpu_host_free(reflect->hi_recv_buf);
    _XMP_gpu_free(reflect->lo_send_dev_buf);
    _XMP_gpu_free(reflect->lo_recv_dev_buf);
    _XMP_gpu_free(reflect->hi_send_dev_buf);
    _XMP_gpu_free(reflect->hi_recv_dev_buf);
  }
  if ((_XMPF_running && target_dim == ndims - 1) ||
      (_XMPC_running && target_dim == 0)){
    _XMP_gpu_host_free(reflect->lo_send_buf);
    _XMP_gpu_host_free(reflect->lo_recv_buf);
    _XMP_gpu_host_free(reflect->hi_send_buf);
    _XMP_gpu_host_free(reflect->hi_recv_buf);
  }
  

  // for lower reflect

  if (lwidth){

    lo_buf_size = lwidth * blocklength * count;

    if ((_XMPF_running && target_dim == ndims - 1) ||
	(_XMPC_running && target_dim == 0)){
      lo_send_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_recv_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_send_dev_buf = lo_send_dev_array;
      lo_recv_dev_buf = lo_recv_dev_array;
    }
    else {
      _XMP_TSTART(t0);
      lo_send_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_recv_buf = _XMP_gpu_host_alloc(lo_buf_size);

      _XMP_gpu_alloc((void **)&lo_send_dev_buf, lo_buf_size); //lo_send_dev_buf = _XMP_gpu_alloc(lo_buf_size);
      _XMP_gpu_alloc((void **)&lo_recv_dev_buf, lo_buf_size); //lo_recv_dev_buf = _XMP_gpu_alloc(lo_buf_size);
      _XMP_TEND2(xmptiming_.t_mem, xmptiming_.tdim_mem[target_dim], t0);
    }

  }

  // for upper reflect

  if (uwidth){

    hi_buf_size = uwidth * blocklength * count;

    if ((_XMPF_running && target_dim == ndims - 1) ||
	(_XMPC_running && target_dim == 0)){
      hi_send_buf = _XMP_gpu_host_alloc(hi_buf_size);
      hi_recv_buf = _XMP_gpu_host_alloc(hi_buf_size);
      hi_send_dev_buf = hi_send_dev_array;
      hi_recv_dev_buf = hi_recv_dev_array;
    }
    else {
      _XMP_TSTART(t0);
      hi_send_buf = _XMP_gpu_host_alloc(hi_buf_size);
      hi_recv_buf = _XMP_gpu_host_alloc(hi_buf_size);
      _XMP_gpu_alloc((void **)&hi_send_dev_buf, hi_buf_size); //hi_send_dev_buf = _XMP_gpu_alloc(hi_buf_size);
      _XMP_gpu_alloc((void **)&hi_recv_dev_buf, hi_buf_size); //hi_recv_dev_buf = _XMP_gpu_alloc(hi_buf_size);
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

  reflect->lo_send_dev_array = lo_send_dev_array;
  reflect->lo_recv_dev_array = lo_recv_dev_array;
  reflect->hi_send_dev_array = hi_send_dev_array;
  reflect->hi_recv_dev_array = hi_recv_dev_array;

  reflect->lo_send_dev_buf = lo_send_dev_buf;
  reflect->lo_recv_dev_buf = lo_recv_dev_buf;
  reflect->hi_send_dev_buf = hi_send_dev_buf;
  reflect->hi_recv_dev_buf = hi_recv_dev_buf;

  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;

  // gpu async
  reflect->lo_async_id = _XMP_gpu_alloc_async_id();
  reflect->hi_async_id = _XMP_gpu_alloc_async_id();
}

static void _XMP_reflect_start(_XMP_array_t *a, int dummy)
//static void _XMP_reflect_start(_XMP_array_t *a, int *is_periodic, int dummy)
{
  //printf("reflect start\n");
  _XMP_TSTART(t0);
  _XMP_reflect_gpu_pack(a);
  _XMP_TEND(xmptiming_.t_copy, t0);

  for (int i = 0; i < a->dim; i++){
    //printf("dim=%d\n", i);

    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      size_t lo_buf_size = lo_width * reflect->blocklength * reflect->count;
      size_t hi_buf_size = hi_width * reflect->blocklength * reflect->count;
      //printf("move");
      if(lo_width && reflect->hi_rank != MPI_PROC_NULL){
	_XMP_gpu_update_host_async(reflect->lo_send_buf, reflect->lo_send_dev_buf, lo_buf_size, reflect->lo_async_id);
      }
      if(hi_width && reflect->lo_rank != MPI_PROC_NULL){
	_XMP_gpu_update_host_async(reflect->hi_send_buf, reflect->hi_send_dev_buf, hi_buf_size, reflect->hi_async_id);
      }
      //      _XMP_gpu_wait_async(reflect->lo_async_id);
      //      _XMP_gpu_wait_async(reflect->hi_async_id);
      //_XMP_TSTART(t0);
      //MPI_Startall(4, reflect->req);
      //_XMP_TEND2(xmptiming_.t_comm, xmptiming_.tdim_comm[i], t0);

    }

  }

  for (int i = 0; i < a->dim; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      if(lo_width && reflect->hi_rank != MPI_PROC_NULL){
	_XMP_gpu_wait_async(reflect->lo_async_id);
      }
      if(hi_width && reflect->lo_rank != MPI_PROC_NULL){
	_XMP_gpu_wait_async(reflect->hi_async_id);
      }
    }
  }

  for (int i = 0; i < a->dim; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      _XMP_TSTART(t0);
      MPI_Startall(4, reflect->req);
      _XMP_TEND2(xmptiming_.t_comm, xmptiming_.tdim_comm[i], t0);
    }
  }
}


static void _XMP_reflect_wait(_XMP_array_t *a)
{
  //for (int i = 0; i < a->dim; i++){
  for (int i = a->dim-1; i >= 0; i--){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;
    //    if (!lwidth[i] && !uwidth[i]) continue;
    //    _XMP_array_info_t *ai = &(a->info[i]);
    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      _XMP_TSTART(t0);
      MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
      _XMP_TEND2(xmptiming_.t_wait, xmptiming_.tdim_wait[i], t0);

      if(lo_width && reflect->lo_rank != MPI_PROC_NULL){
	size_t lo_buf_size = lo_width * reflect->blocklength * reflect->count;
	//_XMP_gpu_update_device_async(reflect->lo_recv_dev_buf, reflect->lo_recv_buf, lo_buf_size, reflect->lo_async_id);
	//	cudaStream_t lo_stream = 
	cudaMemcpyAsync(reflect->lo_recv_dev_buf, reflect->lo_recv_buf, lo_buf_size, 
			cudaMemcpyHostToDevice, *(cudaStream_t*)reflect->lo_async_id);
      }
      if(hi_width && reflect->hi_rank != MPI_PROC_NULL){
	size_t hi_buf_size = hi_width * reflect->blocklength * reflect->count;
	//_XMP_gpu_update_device_async(reflect->hi_recv_dev_buf, reflect->hi_recv_buf, hi_buf_size, reflect->hi_async_id);
	cudaMemcpyAsync(reflect->hi_recv_dev_buf, reflect->hi_recv_buf, hi_buf_size, 
			cudaMemcpyHostToDevice, *(cudaStream_t*)reflect->hi_async_id);
      }
    }
    else if (ai->shadow_type == _XMP_N_SHADOW_FULL){
      _XMP_reflect_shadow_FULL(a->array_addr_p, a, i);
    }

  }

  if (/*_xmp_reflect_pack_flag*/ 1){
    _XMP_TSTART(t0);
    _XMP_reflect_gpu_unpack(a);

    for(int i = a->dim-1; i >= 0; i--){
      _XMP_array_info_t *ai = &(a->info[i]);
      if(! ai->is_shadow_comm_member) continue;
      _XMP_reflect_sched_t *reflect = ai->reflect_sched;
      int lo_width = reflect->lo_width;
      int hi_width = reflect->hi_width;
      if (!lo_width && !hi_width) continue;

      if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
	if(lo_width && reflect->lo_rank != MPI_PROC_NULL){
	  //_XMP_gpu_wait_async(reflect->lo_async_id);
	  cudaStreamSynchronize(*(cudaStream_t*)reflect->lo_async_id);
	}
	if(hi_width && reflect->hi_rank != MPI_PROC_NULL){
	  //_XMP_gpu_wait_async(reflect->hi_async_id);
	  cudaStreamSynchronize(*(cudaStream_t*)reflect->hi_async_id);
	}
      }
    }
    _XMP_TEND(xmptiming_.t_copy, t0);
  }

}

static void _XMP_reflect_gpu_pack(_XMP_array_t *a)
{

  int lb, ub;

  if (_XMPF_running & !_XMPC_running){ /* for XMP/F */
    lb = 0;
    ub = a->dim - 1;
  }
  else if (!_XMPF_running & _XMPC_running){ /* for XMP/C */
    lb = 1;
    ub = a->dim;
  }
  else {
    _XMP_fatal("cannot determin the base language.");
  }

  for (int i = lb; i < ub; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      // for lower reflect
      if (lo_width && reflect->hi_rank != MPI_PROC_NULL){
	_XMP_gpu_pack_vector_async((char *)reflect->lo_send_dev_buf,
				   (char *)reflect->lo_send_dev_array,
				   reflect->count, lo_width * reflect->blocklength,
				   reflect->stride, a->type_size, reflect->lo_async_id);
      }

      // for upper reflect
      if (hi_width && reflect->lo_rank != MPI_PROC_NULL){
	_XMP_gpu_pack_vector_async((char *)reflect->hi_send_dev_buf,
				   (char *)reflect->hi_send_dev_array,
				   reflect->count, hi_width * reflect->blocklength,
				   reflect->stride, a->type_size, reflect->hi_async_id);
      }
    }
  }
}


static void _XMP_reflect_gpu_unpack(_XMP_array_t *a)
{

  int lb, ub;

  if (_XMPF_running & !_XMPC_running){ /* for XMP/F */
    lb = 0;
    ub = a->dim - 1;
  } else if (!_XMPF_running & _XMPC_running){ /* for XMP/C */
    lb = 1;
    ub = a->dim;
  } else {
    _XMP_fatal("cannot determin the base language.");
  }

  //for (int i = lb; i < ub; i++){
  for(int i = ub - 1; i >= lb; i--){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      // for lower reflect
      if (lo_width && reflect->lo_rank != MPI_PROC_NULL){
	_XMP_gpu_unpack_vector_async((char *)reflect->lo_recv_dev_array,
				     (char *)reflect->lo_recv_dev_buf,
				     reflect->count, lo_width * reflect->blocklength,
				     reflect->stride, a->type_size, reflect->lo_async_id);
      }

      // for upper reflect
      if (hi_width && reflect->hi_rank != MPI_PROC_NULL){
	_XMP_gpu_unpack_vector_async((char *)reflect->hi_recv_dev_array,
				     (char *)reflect->hi_recv_dev_buf,
				     reflect->count, hi_width * reflect->blocklength,
				     reflect->stride, a->type_size, reflect->hi_async_id);
      }
    }
  }
}
