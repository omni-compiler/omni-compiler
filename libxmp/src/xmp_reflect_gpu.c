#include "xmp_internal.h"
#include "include/cuda_runtime.h"
#include<stdio.h>
#include<stdlib.h>

void _XMP_reflect_do_gpu(_XMP_array_t *array_desc);
void _XMP_reflect_init_gpu(void *acc_addr, _XMP_array_t *array_desc);

void _XMP_gpu_pack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);
void _XMP_gpu_unpack_vector_async(char * restrict dst, char * restrict src, int count, int blocklength, long stride, size_t typesize, void* async_id);
void _XMP_gpu_pack_vector2_async(char * restrict dst0, char * restrict src0, int blocklength0, long stride0,
				  char * restrict dst1, char * restrict src1, int blocklength1, long stride1,
				  int count, size_t typesize, cudaStream_t st);
void _XMP_gpu_unpack_vector2_async(char * restrict dst0, char * restrict src0, int blocklength0, long stride0,
				    char * restrict dst1, char * restrict src1, int blocklength1, long stride1,
				    int count, size_t typesize, cudaStream_t st);

static void _XMP_reflect_wait(_XMP_array_t *a);

static void _XMP_reflect_start(_XMP_array_t *a, int dummy);
static void _XMP_reflect_sched(_XMP_array_t *a, int *lwidth, int *uwidth, int *is_periodic, int is_async, void *dev_addr);
static void _XMP_reflect_pcopy_sched_dim(_XMP_array_t *adesc, int target_dim,
					 int lwidth, int uwidth, int is_periodic, void *dev_array_addr);
static int _xmpf_set_reflect_flag = 0;
static int _xmp_lwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_uwidth[_XMP_N_MAX_DIM] = {0};
static int _xmp_is_periodic[_XMP_N_MAX_DIM] = {0};

static char useHostBuffer = 1;
static char packVector = 1;
//static const char useSingleKernel = 0;
static const int useSingleStreamLimit = 16 * 1024; //element

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)						\
  do {                                                                  \
    cudaError_t err = call;						\
    if (cudaSuccess != err) {						\
      fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	       __FILE__, __LINE__, cudaGetErrorString(err) );		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

//#define _TLOG
#ifdef _TLOG
#include "tlog.h"
#define TLOG_LOG(log) do{tlog_log((log));}while(0)
#else
#define TLOG_LOG(log) do{}while(0)
#endif

static void gpu_memcpy_async(void *dst, void *src, size_t size, void *async_id)
{
  cudaStream_t st = *((cudaStream_t*)async_id);
  CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, st));
}

static void gpu_wait_async(void *async_id)
{
  cudaStream_t st = *((cudaStream_t*)async_id);
  CUDA_SAFE_CALL(cudaStreamSynchronize(st));
}

void _XMP_reflect_init_gpu(void *dev_addr,_XMP_array_t *a)
{
  _XMP_RETURN_IF_SINGLE;

  static char isFlagSetted = 0;
  if(! isFlagSetted ){
    char *mode_str = getenv("XACC_COMM_MODE");
    if(mode_str !=  NULL){
      int mode = atoi(mode_str);
      switch(mode){
      default:
      case 0:
	packVector = 1;
	useHostBuffer = 1;
	break;
      case 1:
	packVector = 1;
	useHostBuffer = 0;
	break;
      case 2:
	packVector = 0;
	useHostBuffer = 0;
	break;
      }
    }
    isFlagSetted = 1;
  }
  //printf("reflect mode (%d, %d)\n", packVector, useHostBuffer);

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

      _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;

      if(reflect == NULL){
	reflect = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
	reflect->is_periodic = -1; /* not used yet */
	reflect->datatype_lo = MPI_DATATYPE_NULL;
	reflect->datatype_hi = MPI_DATATYPE_NULL;
	for (int j = 0; j < 4; j++) reflect->req[j] = MPI_REQUEST_NULL;
	reflect->lo_send_buf = NULL;
	reflect->lo_recv_buf = NULL;
	reflect->hi_send_buf = NULL;
	reflect->hi_recv_buf = NULL;
	reflect->lo_send_host_buf = NULL;
	reflect->lo_recv_host_buf = NULL;
	reflect->hi_send_host_buf = NULL;
	reflect->hi_recv_host_buf = NULL;
	ai->reflect_acc_sched = reflect;
      }else{
	//
      }

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

  _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;

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
  //void *array_addr = adesc->array_addr_p;

  void *lo_send_array = NULL;
  void *lo_recv_array = NULL;
  void *hi_send_array = NULL;
  void *hi_recv_array = NULL;

  void *lo_send_dev_buf = NULL;
  void *lo_recv_dev_buf = NULL;
  void *hi_send_dev_buf = NULL;
  void *hi_recv_dev_buf = NULL;
  void *lo_send_host_buf = NULL;
  void *lo_recv_host_buf = NULL;
  void *hi_send_host_buf = NULL;
  void *hi_recv_host_buf = NULL;

  void *mpi_lo_send_buf = NULL;
  void *mpi_lo_recv_buf = NULL;
  void *mpi_hi_send_buf = NULL;
  void *mpi_hi_recv_buf = NULL;

  int lo_buf_size = 0;
  int hi_buf_size = 0;

  //
  // setup data_type
  //

  int count = 0, blocklength = 0;
  long long stride = 0;
  int count_offset = 0;

  if (_XMPF_running && !_XMPC_running){ /* for XMP/F */

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
  else if (!_XMPF_running && _XMPC_running){ /* for XMP/C */

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
    lo_send_array = lo_recv_array = (void *)((char*)dev_array_addr + count_offset * stride);

    for (int i = 0; i < ndims; i++) {
      int lb_send, lb_recv;
      unsigned long long dim_acc;

      if (i == target_dim) {
	//printf("ainfo[%d].local_upper=%d\n",i,ainfo[i].local_upper);
	lb_send = ainfo[i].local_upper - lwidth + 1;
	lb_recv = ainfo[i].shadow_size_lo - lwidth; ////ainfo[i].local_lower - lwidth;
      } else {
	// Note: including shadow area
	lb_send = 0; //// ainfo[i].local_lower - ainfo[i].shadow_size_lo;
	lb_recv = 0; //// ainfo[i].local_lower - ainfo[i].shadow_size_lo;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_array = (void *)((char *)lo_send_array + lb_send * dim_acc * type_size);
      lo_recv_array = (void *)((char *)lo_recv_array + lb_recv * dim_acc * type_size);
    }
  }

  // for upper reflect

  if (uwidth){
    hi_send_array = hi_recv_array = (void *)((char*)dev_array_addr + count_offset * stride);

    for (int i = 0; i < ndims; i++) {
      int lb_send, lb_recv;
      unsigned long long dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      } else {
	// Note: including shadow area
	lb_send = 0; //ainfo[i].local_lower - ainfo[i].shadow_size_lo;
	lb_recv = 0; //ainfo[i].local_lower - ainfo[i].shadow_size_lo;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_array = (void *)((char *)hi_send_array + lb_send * dim_acc * type_size);
      hi_recv_array = (void *)((char *)hi_recv_array + lb_recv * dim_acc * type_size);
    }
  }

  // for lower reflect
  if (reflect->datatype_lo != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_lo);
  }
  if(packVector || count == 1){
    MPI_Type_contiguous(blocklength * lwidth * count, MPI_BYTE, &reflect->datatype_lo);
  }else{
    MPI_Type_vector(count, blocklength * lwidth, stride, MPI_BYTE, &reflect->datatype_lo);
  }
  MPI_Type_commit(&reflect->datatype_lo);

  // for upper reflect
  if (reflect->datatype_hi != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_hi);
  }
  if(packVector || count == 1){
    MPI_Type_contiguous(blocklength * uwidth * count, MPI_BYTE, &reflect->datatype_hi);
  }else{
    MPI_Type_vector(count, blocklength * uwidth, stride, MPI_BYTE, &reflect->datatype_hi);
  }
  MPI_Type_commit(&reflect->datatype_hi);


  //
  // Allocate buffers
  //
  if(useHostBuffer){
    CUDA_SAFE_CALL(cudaFreeHost(reflect->lo_send_host_buf));
    CUDA_SAFE_CALL(cudaFreeHost(reflect->lo_recv_host_buf));
  }    

  if ((_XMPF_running && target_dim != ndims - 1) ||
      (_XMPC_running && target_dim != 0)){
    if(packVector){
      CUDA_SAFE_CALL(cudaFree(reflect->lo_send_buf));
      CUDA_SAFE_CALL(cudaFree(reflect->lo_recv_buf));
    }
  }
  if ((_XMPF_running && target_dim == ndims - 1) ||
      (_XMPC_running && target_dim == 0)){
    //
  }
  

  // for lower reflect

  if (lwidth){

    lo_buf_size = lwidth * blocklength * count;
    hi_buf_size = uwidth * blocklength * count;

    if ((_XMPF_running && target_dim == ndims - 1) ||
	(_XMPC_running && target_dim == 0)){
      lo_send_dev_buf = lo_send_array;
      lo_recv_dev_buf = lo_recv_array;
      hi_send_dev_buf = hi_send_array;
      hi_recv_dev_buf = hi_recv_array;
    } else {
      _XMP_TSTART(t0);
      if(packVector){
	CUDA_SAFE_CALL(cudaMalloc((void **)&lo_send_dev_buf, lo_buf_size + hi_buf_size));
	hi_send_dev_buf = (char*)lo_send_dev_buf + lo_buf_size;
	CUDA_SAFE_CALL(cudaMalloc((void **)&lo_recv_dev_buf, lo_buf_size + hi_buf_size));
	hi_recv_dev_buf = (char*)lo_recv_dev_buf + lo_buf_size;	
      }else{
	lo_send_dev_buf = lo_send_array;
	lo_recv_dev_buf = lo_recv_array;
	hi_send_dev_buf = hi_send_array;
	hi_recv_dev_buf = hi_recv_array;
      }
      _XMP_TEND2(xmptiming_.t_mem, xmptiming_.tdim_mem[target_dim], t0);
    }

    if(useHostBuffer){
      CUDA_SAFE_CALL(cudaMallocHost((void**)&lo_send_host_buf, lo_buf_size + hi_buf_size));
      hi_send_host_buf = (char*)lo_send_host_buf + lo_buf_size;
      CUDA_SAFE_CALL(cudaMallocHost((void**)&lo_recv_host_buf, lo_buf_size + hi_buf_size));
      hi_recv_host_buf = (char*)lo_recv_host_buf + lo_buf_size;
      mpi_lo_send_buf = lo_send_host_buf;
      mpi_lo_recv_buf = lo_recv_host_buf;
      mpi_hi_send_buf = hi_send_host_buf;
      mpi_hi_recv_buf = hi_recv_host_buf;
    }else{
      mpi_lo_send_buf = lo_send_dev_buf;
      mpi_lo_recv_buf = lo_recv_dev_buf;
      mpi_hi_send_buf = hi_send_dev_buf;
      mpi_hi_recv_buf = hi_recv_dev_buf;
    }
  }

  // for upper reflect


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
  } else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[0] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[0]);
  }
	
  if (reflect->req[1] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[1]);
  }

  MPI_Recv_init(mpi_lo_recv_buf, 1, reflect->datatype_lo, src,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[0]);
  MPI_Send_init(mpi_lo_send_buf, 1, reflect->datatype_lo, dst,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[1]);

  // for upper shadow

  if (uwidth){
    src = hi_rank;
    dst = lo_rank;
  } else {
    src = MPI_PROC_NULL;
    dst = MPI_PROC_NULL;
  }

  if (reflect->req[2] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[2]);
  }
	
  if (reflect->req[3] != MPI_REQUEST_NULL){
    MPI_Request_free(&reflect->req[3]);
  }

  MPI_Recv_init(mpi_hi_recv_buf, 1, reflect->datatype_hi, src,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[2]);
  MPI_Send_init(mpi_hi_send_buf, 1, reflect->datatype_hi, dst,
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

  if(packVector){
    reflect->lo_send_buf = lo_send_dev_buf;
    reflect->lo_recv_buf = lo_recv_dev_buf;
    reflect->hi_send_buf = hi_send_dev_buf;
    reflect->hi_recv_buf = hi_recv_dev_buf;
  }

  if(useHostBuffer){
    reflect->lo_send_host_buf = lo_send_host_buf;
    reflect->lo_recv_host_buf = lo_recv_host_buf;
    reflect->hi_send_host_buf = hi_send_host_buf;
    reflect->hi_recv_host_buf = hi_recv_host_buf;
  }

  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;

  // gpu async
  reflect->lo_async_id = _XMP_alloc(sizeof(cudaStream_t));
  CUDA_SAFE_CALL(cudaStreamCreate(reflect->lo_async_id));

  if(target_dim != 0 &&
     (!useHostBuffer || (lo_rank != MPI_PROC_NULL && hi_rank != MPI_PROC_NULL && (lo_buf_size / type_size) <= useSingleStreamLimit)) ){
    reflect->hi_async_id = NULL;
  }else{
    cudaStream_t *hi_stream = (cudaStream_t*)_XMP_alloc(sizeof(cudaStream_t));
    CUDA_SAFE_CALL(cudaStreamCreate(hi_stream));
    reflect->hi_async_id = (void*)hi_stream;
  }

  reflect->event = _XMP_alloc(sizeof(cudaEvent_t));
  CUDA_SAFE_CALL(cudaEventCreateWithFlags(reflect->event, cudaEventDisableTiming));
}

static void gpu_update_host(_XMP_reflect_sched_t *reflect)
{
  if(reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL){
    size_t lo_buf_size = reflect->lo_width * reflect->blocklength * reflect->count;
    size_t hi_buf_size = reflect->hi_width * reflect->blocklength * reflect->count;    
    gpu_memcpy_async(reflect->lo_send_host_buf, reflect->lo_send_buf, lo_buf_size + hi_buf_size, reflect->lo_async_id);
  }else{
  if(reflect->hi_rank != MPI_PROC_NULL){
    size_t lo_buf_size = reflect->lo_width * reflect->blocklength * reflect->count;
    gpu_memcpy_async(reflect->lo_send_host_buf, reflect->lo_send_buf, lo_buf_size, reflect->lo_async_id);
  }
  if(reflect->lo_rank != MPI_PROC_NULL){
    size_t hi_buf_size = reflect->hi_width * reflect->blocklength * reflect->count;
    gpu_memcpy_async(reflect->hi_send_host_buf, reflect->hi_send_buf, hi_buf_size, reflect->hi_async_id);
  }
  }
}

static void gpu_update_device(_XMP_reflect_sched_t *reflect)
{
  if(reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL){
    size_t lo_buf_size = reflect->lo_width * reflect->blocklength * reflect->count;
    size_t hi_buf_size = reflect->hi_width * reflect->blocklength * reflect->count;
    gpu_memcpy_async(reflect->lo_recv_buf, reflect->lo_recv_host_buf, lo_buf_size + hi_buf_size, reflect->lo_async_id);
  }else{
  if(reflect->lo_rank != MPI_PROC_NULL){
    int lo_width = reflect->lo_width;
    size_t lo_buf_size = lo_width * reflect->blocklength * reflect->count;
    gpu_memcpy_async(reflect->lo_recv_buf, reflect->lo_recv_host_buf, lo_buf_size, reflect->lo_async_id);
  }
  if(reflect->hi_rank != MPI_PROC_NULL){
    int hi_width = reflect->hi_width;
    size_t hi_buf_size = hi_width * reflect->blocklength * reflect->count;
    gpu_memcpy_async(reflect->hi_recv_buf, reflect->hi_recv_host_buf, hi_buf_size, reflect->hi_async_id);
  }
  }
}

static void gpu_pack_wait(_XMP_reflect_sched_t *reflect)
{
  if((!useHostBuffer && (reflect->hi_rank != MPI_PROC_NULL || reflect->lo_rank != MPI_PROC_NULL))
     || (reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL)){
    gpu_wait_async(reflect->lo_async_id);
  }else{
    if(reflect->hi_rank != MPI_PROC_NULL){
      gpu_wait_async(reflect->lo_async_id);
    }
    if(reflect->lo_rank != MPI_PROC_NULL){
      gpu_wait_async(reflect->hi_async_id);
    }
  }
}
static void gpu_unpack_wait(_XMP_reflect_sched_t *reflect)
{
  if((!useHostBuffer && (reflect->hi_rank != MPI_PROC_NULL || reflect->lo_rank != MPI_PROC_NULL))
     || (reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL)){
    gpu_wait_async(reflect->lo_async_id);
  }else{
    if(reflect->lo_rank != MPI_PROC_NULL){
      gpu_wait_async(reflect->lo_async_id);
    }
    if(reflect->hi_rank != MPI_PROC_NULL){
      gpu_wait_async(reflect->hi_async_id);
    }
  }
}

static void gpu_unpack(_XMP_reflect_sched_t *reflect, size_t type_size)
{
  char *lo_recv_array, *lo_recv_buf;
  char *hi_recv_array, *hi_recv_buf;
  int lo_width = reflect->lo_width;
  int hi_width = reflect->hi_width;
  long lo_buf_size = lo_width * reflect->blocklength;
  long hi_buf_size = hi_width * reflect->blocklength;

  if(!useHostBuffer || (reflect->lo_rank != MPI_PROC_NULL && reflect->hi_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL)){

    if (reflect->lo_rank != MPI_PROC_NULL){
      lo_recv_array = reflect->lo_recv_array;
      lo_recv_buf = reflect->lo_recv_buf;
    }else{
      lo_recv_array = lo_recv_buf = NULL;
    }
    if (reflect->hi_rank != MPI_PROC_NULL){
      hi_recv_array = reflect->hi_recv_array;
      hi_recv_buf = reflect->hi_recv_buf;
    }else{
      hi_recv_array = hi_recv_buf = NULL;
    }
  
    _XMP_gpu_unpack_vector2_async(lo_recv_array, lo_recv_buf, lo_buf_size, reflect->stride,
				  hi_recv_array, hi_recv_buf, hi_buf_size, reflect->stride,
				  reflect->count, type_size, *(cudaStream_t*)reflect->lo_async_id);
  }else{
    if (lo_width && reflect->lo_rank != MPI_PROC_NULL){
      _XMP_gpu_unpack_vector_async((char *)reflect->lo_recv_array,
				   (char *)reflect->lo_recv_buf,
				   reflect->count, lo_buf_size,
				   reflect->stride, type_size, reflect->lo_async_id);
    }

    if (hi_width && reflect->hi_rank != MPI_PROC_NULL){
      _XMP_gpu_unpack_vector_async((char *)reflect->hi_recv_array,
				   (char *)reflect->hi_recv_buf,
				   reflect->count, hi_buf_size,
				   reflect->stride, type_size, reflect->hi_async_id);

    }
  }

}

static void gpu_pack_vector2(_XMP_reflect_sched_t *reflect, size_t type_size)
{
  char *lo_send_array, *lo_send_buf;
  char *hi_send_array, *hi_send_buf;
  int lo_width = reflect->lo_width;
  int hi_width = reflect->hi_width;
  long lo_buf_size = lo_width * reflect->blocklength;
  long hi_buf_size = hi_width * reflect->blocklength;

  if(!useHostBuffer || (reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL)){
    if (reflect->hi_rank != MPI_PROC_NULL){
      lo_send_array = reflect->lo_send_array;
      lo_send_buf = reflect->lo_send_buf;
    }else{
      lo_send_array = lo_send_buf = NULL;
    }
    if (reflect->lo_rank != MPI_PROC_NULL){
      hi_send_array = reflect->hi_send_array;
      hi_send_buf = reflect->hi_send_buf;
    }else{
      hi_send_array = hi_send_buf = NULL;
    }
    _XMP_gpu_pack_vector2_async(lo_send_buf, lo_send_array, lo_buf_size, reflect->stride,
				hi_send_buf, hi_send_array, hi_buf_size, reflect->stride,
				reflect->count, type_size, *(cudaStream_t*)reflect->lo_async_id);
  }else{
    if (lo_width && reflect->hi_rank != MPI_PROC_NULL){
      _XMP_gpu_pack_vector_async((char *)reflect->lo_send_buf,
				 (char *)reflect->lo_send_array,
				 reflect->count, lo_buf_size,
				 reflect->stride, type_size, reflect->lo_async_id);
    }

    if (hi_width && reflect->lo_rank != MPI_PROC_NULL){
      _XMP_gpu_pack_vector_async((char *)reflect->hi_send_buf,
				 (char *)reflect->hi_send_array,
				 reflect->count, hi_buf_size,
				 reflect->stride, type_size, reflect->hi_async_id);
    }
  }
}

static void _XMP_reflect_start(_XMP_array_t *a, int dummy)
{
  int packSkipDim = 0;
  if (_XMPF_running && !_XMPC_running){ /* for XMP/F */
    packSkipDim = a->dim - 1;
  } else if (!_XMPF_running && _XMPC_running){ /* for XMP/C */
    packSkipDim = 0;
  } else {
    _XMP_fatal("cannot determin the base language.");
  }

  TLOG_LOG(TLOG_EVENT_3_IN);
  for (int i = 0; i < a->dim; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      if(packVector && (i != packSkipDim)){
	gpu_pack_vector2(reflect, a->type_size);
      }
      TLOG_LOG(TLOG_EVENT_9);
      if(useHostBuffer){
	gpu_update_host(reflect);
      }
    }
  }
  TLOG_LOG(TLOG_EVENT_3_OUT);

  TLOG_LOG(TLOG_EVENT_4_IN);
  for (int i = 0; i < a->dim; i++){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      if((packVector && i != packSkipDim) || useHostBuffer){
	gpu_pack_wait(reflect);
	TLOG_LOG(TLOG_EVENT_2);
      }
      MPI_Startall(4, reflect->req);
      TLOG_LOG(TLOG_EVENT_1);
    }
  }
  TLOG_LOG(TLOG_EVENT_4_OUT);
}

static void _XMP_reflect_wait(_XMP_array_t *a)
{
  int packSkipDim = 0;
  if (_XMPF_running && !_XMPC_running){ /* for XMP/F */
    packSkipDim = a->dim - 1;
  } else if (!_XMPF_running && _XMPC_running){ /* for XMP/C */
    packSkipDim = 0;
  } else {
    _XMP_fatal("cannot determin the base language.");
  }

  TLOG_LOG(TLOG_EVENT_6_IN);
  for (int i = 0; i < a->dim; i++){
    //  for (int i = a->dim - 1; i >= 0; i--){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
      TLOG_LOG(TLOG_EVENT_9);
      
      if(useHostBuffer){
	gpu_update_device(reflect);
      }
      if(packVector && (i != packSkipDim)){
	gpu_unpack(reflect, a->type_size);
	TLOG_LOG(TLOG_EVENT_4);
      }
    }
  }
  TLOG_LOG(TLOG_EVENT_6_OUT);

  TLOG_LOG(TLOG_EVENT_7_IN);
  for(int i = 0; i < a->dim; i++){
    //  for (int i = a->dim - 1; i >= 0; i--){
    _XMP_array_info_t *ai = &(a->info[i]);
    if(! ai->is_shadow_comm_member) continue;
    _XMP_reflect_sched_t *reflect = ai->reflect_acc_sched;
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    if (!lo_width && !hi_width) continue;

    if (ai->shadow_type == _XMP_N_SHADOW_NORMAL){
      if((packVector && i != packSkipDim) || useHostBuffer){
	gpu_unpack_wait(reflect);
	TLOG_LOG(TLOG_EVENT_9);
      }
    }
  }
  TLOG_LOG(TLOG_EVENT_7_OUT);
}
