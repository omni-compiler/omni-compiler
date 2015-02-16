#include <stdio.h>
#if 1
#define _XMP_XACC
#include "xmp_internal.h"
#include "xmp_math_function.h"
#include "xmp_data_struct.h"
#include "xacc_internal.h"
#include "xacc_data_struct.h"
#include "include/cuda_runtime.h"
#else
#define _XMP_XACC
#include "../../libxmp/include/xmp_internal.h"
#include "../../libxmp/include/xmp_math_function.h"
#include "../../libxmp/include/xmp_data_struct.h"
#include "../include/xacc_internal.h"
#include "../include/xacc_data_struct.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#endif
#include <unistd.h>
static char packVector = 1;
static char useHostBuffer = 1;
static const char useKernelPacking = 1; //use kernel for packing
static const int useSingleStreamLimit = 16384; //# of elements
void _XACC_gpu_pack_vector_async_test(char * restrict dst, const char * restrict src, size_t typesize, cudaStream_t st);

static void _XACC_reflect_sched_dim(_XACC_arrays_t *a, int target_device, int target_dim);

static cudaStream_t *stream_pool = NULL;

static void enablePeerAccess(_XACC_device_t *device)
{
  //  printf("deviceInfo(%d,%d,%d)\n", device->lb,device->ub,device->step);
  int d;
  cudaError_t cudaError;

  for(d = device->lb; d <= device->ub; d += device->step){
    cudaError = cudaSetDevice(d);
    if(cudaError != cudaSuccess){
      _XMP_fatal("failed to set device");
      return;
    }

    int d2;
    
    d2 = d - device->step;
    if(d2 >= device->lb){
      int canAccess;
      cudaError = cudaDeviceCanAccessPeer(&canAccess, d, d2);
      if(cudaError != cudaSuccess){
	_XMP_fatal("failed to check access peer");
      }
      if(canAccess == 0){
	//	printf("eneblePeerAccess(%d) on %d\n", d2-1, d-1);
	cudaError = cudaDeviceEnablePeerAccess(d2, 0);
	if(cudaError == cudaErrorPeerAccessAlreadyEnabled){
	  //
	}else if(cudaError == cudaErrorInvalidDevice){
	  fprintf(stderr, "failed to enable peer access, invalidDevice\n");
	}else if(cudaError == cudaErrorInvalidValue){
	  fprintf(stderr, "failed to enable peer access, invalid value\n");
	}else if(cudaError != cudaSuccess){
	  fprintf(stderr, "failed to enable peer access, %d, (%d,%d), %s\n", (int)cudaError, d,d2, cudaGetErrorString(cudaError));
	  //	  return;
	}
      }else{
	fprintf(stderr, "already enabled peer access, (%d,%d)\n", d, d2);
      }
    }

    d2 = d + device->step;
    if(d2 <= device->ub){
      int canAccess;
      cudaError = cudaDeviceCanAccessPeer(&canAccess, d, d2);
      if(cudaError != cudaSuccess){
	_XMP_fatal("failed to check access peer");
      }
      if(canAccess == 0){
	//	printf("eneblePeerAccess(%d) on %d\n", d2-1, d-1);
	cudaError = cudaDeviceEnablePeerAccess(d2, 0);
	if(cudaError == cudaErrorPeerAccessAlreadyEnabled){
	  //
	}else if(cudaError == cudaErrorInvalidDevice){
	  fprintf(stderr, "failed to enable peer access, invalidDevice\n");
	}else if(cudaError == cudaErrorInvalidValue){
	  fprintf(stderr, "failed to enable peer access, invalid value\n");
	}else if(cudaError != cudaSuccess){
	  fprintf(stderr, "failed to enable peer access, %d, (%d,%d), %s\n", (int)cudaError, d,d2, cudaGetErrorString(cudaError));
	  //	  return;
	}
      }
    }
  }
}

void _XACC_reflect_init(_XACC_arrays_t *arrays_desc)
{
  _XACC_device_t *device = arrays_desc->device_type;

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


  //enablePeerAccess(device);

  int dim = arrays_desc->dim;
  //他ノードとの通信のセットアップ
  //  int *lwidth = _XMP_alloc(sizeof(int)*dim);
  //  int *uwidth = _XMP_alloc(sizeof(int)*dim);
  for(int i = device->lb; i <= device->ub; i+= device->step){
    _XACC_array_t *array_desc = arrays_desc->array + i;
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
        _XMP_reflect_sched_t *reflect = ai->reflect_sched;

        if(reflect == NULL){
          reflect = _XMP_alloc(sizeof(_XMP_reflect_sched_t));
          reflect->is_periodic = -1; /* not used yet */
          reflect->datatype_lo = MPI_DATATYPE_NULL;
          reflect->datatype_hi = MPI_DATATYPE_NULL;
          for (int k = 0; k < 4; k++) reflect->req[k] = MPI_REQUEST_NULL;
          reflect->lo_send_buf = NULL;
          reflect->lo_recv_buf = NULL;
          reflect->hi_send_buf = NULL;
          reflect->hi_recv_buf = NULL;
          reflect->lo_send_host_buf = NULL;
          reflect->lo_recv_host_buf = NULL;
          reflect->hi_send_host_buf = NULL;
          reflect->hi_recv_host_buf = NULL;
          ai->reflect_sched = reflect;
        }

        reflect->lo_width = ai->shadow_size_lo;
        reflect->hi_width = ai->shadow_size_hi;
        reflect->is_periodic = 0;

        _XACC_reflect_sched_dim(arrays_desc, i, j);
      }
    }
  }

}

static void _XACC_reflect_sched_dim(_XACC_arrays_t *arrays_desc, int target_device, int target_dim){
  //if (lwidth == 0 && uwidth == 0) return;

  _XACC_array_t *array_desc = arrays_desc->array + target_device;
  _XACC_array_info_t *ai = array_desc->info + target_dim;
  _XACC_array_info_t *ainfo = array_desc->info;
  _XMP_reflect_sched_t *reflect = ai->reflect_sched;
  if (reflect->lo_width > ai->shadow_size_lo || reflect->hi_width > ai->shadow_size_hi){
    _XMP_fatal("reflect width is larger than shadow width.");
  }

  int target_tdim = (arrays_desc->xmp_array->info + target_dim)->align_template_index;
  _XMP_nodes_info_t *xmp_ni =   arrays_desc->xmp_array->align_template->chunk[target_tdim].onto_nodes_info;
  int ndims = arrays_desc->dim;

  _XMP_array_t *xmp_adesc = arrays_desc->xmp_array;
  _XMP_array_info_t *xmp_ai = xmp_adesc->info + target_dim;
  
  // 0-origin
  int my_pos = xmp_ni->rank;
  int lb_pos = _XMP_get_owner_pos(xmp_adesc, target_dim, xmp_ai->ser_lower);
  int ub_pos = _XMP_get_owner_pos(xmp_adesc, target_dim, xmp_ai->ser_upper);
  int lo_pos = (my_pos == lb_pos) ? ub_pos : my_pos - 1;
  int hi_pos = (my_pos == ub_pos) ? lb_pos : my_pos + 1;

  MPI_Comm *comm = xmp_adesc->align_template->onto_nodes->comm;
  int my_rank = xmp_adesc->align_template->onto_nodes->comm_rank;

  int lo_rank = my_rank + (lo_pos - my_pos) * xmp_ni->multiplier;
  int hi_rank = my_rank + (hi_pos - my_pos) * xmp_ni->multiplier;

  int type_size = xmp_adesc->type_size;
  void *array_addr = array_desc->deviceptr;/////xmp_adesc->array_addr_p;

  void *lo_send_array = NULL;
  void *lo_recv_array = NULL;
  void *hi_send_array = NULL;
  void *hi_recv_array = NULL;
  void *lo_send_buf = NULL;
  void *lo_recv_buf = NULL;
  void *hi_send_buf = NULL;
  void *hi_recv_buf = NULL;
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

  int count, blocklength;
  long long stride;
  int count_offset = 0;

  if (_XMPF_running && (!_XMPC_running)){ /* for XMP/F */

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

  } else if ((!_XMPF_running) && _XMPC_running){ /* for XMP/C */
    count = 1;
    blocklength = type_size;
    stride = ainfo[ndims-1].alloc_size * type_size;

    if(target_dim > 0){
      //count *= ainfo[0].par_size;
      //count_offset = ainfo[0].shadow_size_lo;
      count *= ainfo[0].alloc_size - ainfo[0].shadow_size_lo - ainfo[0].shadow_size_hi;
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

    /* printf("count =%d, blength=%d, stride=%lld\n", count, blocklength, stride); */
    /* printf("ainfo[0].par_size=%d\n", ainfo[0].par_size); */
    /* printf("count_ofset=%d,\n", count_offset); */
  }
  else {
    _XMP_fatal("cannot determin the base language.");
  }

  int lwidth = reflect->lo_width;
  if (lwidth){
    lo_send_array = lo_recv_array = (void *)((char*)array_addr + count_offset * stride);

    for (int i = 0; i < ndims; i++) {
      int lb_send, lb_recv;
      unsigned long long dim_acc;

      if (i == target_dim) {
	//printf("ainfo[%d].local_upper=%d\n",i,ainfo[i].local_upper);
	lb_send = ainfo[i].local_upper - lwidth + 1;
	//lb_recv = ainfo[i].shadow_size_lo - lwidth;;
	lb_recv = ainfo[i].local_lower - lwidth;
      }else {
	// Note: including shadow area
	//lb_send = 0;
	//lb_recv = 0;
	lb_send = ainfo[i].local_lower - ainfo[i].shadow_size_lo;
	lb_recv = ainfo[i].local_lower - ainfo[i].shadow_size_lo;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_array = (void *)((char *)lo_send_array + lb_send * dim_acc * type_size);
      lo_recv_array = (void *)((char *)lo_recv_array + lb_recv * dim_acc * type_size);
    }

    //    lo_send_buf = lo_send_array;
    //    lo_recv_buf = lo_recv_array;
  }

  int uwidth = reflect->hi_width;
  if (uwidth){
    hi_send_array = hi_recv_array = (void *)((char*)array_addr + count_offset * stride);

    for (int i = 0; i < ndims; i++) {
      int lb_send, lb_recv;
      unsigned long long dim_acc;

      if (i == target_dim) {
	lb_send = ainfo[i].local_lower;
	lb_recv = ainfo[i].local_upper + 1;
      }else {
	// Note: including shadow area
	/* lb_send = 0; */
	/* lb_recv = 0; */
	lb_send = ainfo[i].local_lower - ainfo[i].shadow_size_lo;
	lb_recv = ainfo[i].local_lower - ainfo[i].shadow_size_lo;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_array = (void *)((char *)hi_send_array + lb_send * dim_acc * type_size);
      hi_recv_array = (void *)((char *)hi_recv_array + lb_recv * dim_acc * type_size);
    }

    //    hi_send_buf = hi_send_array;
    //    hi_recv_buf = hi_recv_array;
  }

  // for lower reflect

  if (reflect->datatype_lo != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_lo);
  }

  //printf("lower type_vector(%d, %d, %lld) @ rank=%d,dev=%d\n",count, blocklength * lwidth,stride,my_rank, target_device);
  if(packVector || count == 1){
    MPI_Type_contiguous(blocklength * lwidth * count, MPI_BYTE, &reflect->datatype_lo);
  }else{
    //printf("use type vector\n");
  MPI_Type_vector(count, blocklength * lwidth, stride, //(count!=1)?stride:blocklength*lwidth,
		  MPI_BYTE, &reflect->datatype_lo);
  }
  MPI_Type_commit(&reflect->datatype_lo);

  // for upper reflect

  if (reflect->datatype_hi != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_hi);
  }

  //printf("upper type_vector(%d, %d, %lld) @ rank=%d,dev=%d\n",count, blocklength * uwidth,stride,my_rank,target_device);
  if(packVector || count == 1){
    MPI_Type_contiguous(blocklength * uwidth * count, MPI_BYTE, &reflect->datatype_hi);
  }else{
    //printf("use type vector\n");
  MPI_Type_vector(count, blocklength * uwidth, stride, //(count!=1)?stride:blocklength * uwidth,
		  MPI_BYTE, &reflect->datatype_hi);
  }
  MPI_Type_commit(&reflect->datatype_hi);


  CUDA_SAFE_CALL(cudaSetDevice(target_device));
  
  //alloc buffer
  if(packVector){
    /* _XACC_gpu_host_free(reflect->lo_send_buf); */
    /* _XACC_gpu_host_free(reflect->lo_recv_buf); */
    /* _XACC_gpu_host_free(reflect->hi_send_buf); */
    /* _XACC_gpu_host_free(reflect->hi_recv_buf); */
    //    CUDA_SAFE_CALL(cudaFree(reflect->lo_send_buf));
    //    CUDA_SAFE_CALL(cudaFree(reflect->lo_recv_buf));
    //    CUDA_SAFE_CALL(cudaFree(reflect->hi_send_buf));
    //    CUDA_SAFE_CALL(cudaFree(reflect->hi_recv_buf));
    /* _XACC_gpu_free(reflect->lo_send_buf); */
    /* _XACC_gpu_free(reflect->lo_recv_buf); */
    /* _XACC_gpu_free(reflect->hi_send_buf); */
    /* _XACC_gpu_free(reflect->hi_recv_buf); */
  }
  if(useHostBuffer){
    //    CUDA_SAFE_CALL(cudaFreeHost(reflect->lo_send_host_buf));
    //    CUDA_SAFE_CALL(cudaFreeHost(reflect->lo_recv_host_buf));
    //    CUDA_SAFE_CALL(cudaFreeHost(reflect->hi_send_host_buf));
    //    CUDA_SAFE_CALL(cudaFreeHost(reflect->hi_recv_host_buf));
  }

  // for lower reflect
  if (1){
    lo_buf_size = lwidth * blocklength * count;
    hi_buf_size = uwidth * blocklength * count;
    if (!packVector ||
	(_XMPF_running && target_dim == ndims - 1) ||
	(_XMPC_running && target_dim == 0)){
      lo_send_buf = lo_send_array;
      lo_recv_buf = lo_recv_array;
      hi_send_buf = hi_send_array;
      hi_recv_buf = hi_recv_array;
    } else {
      CUDA_SAFE_CALL(cudaMalloc((void**)&lo_send_buf, lo_buf_size + hi_buf_size));
      //CUDA_SAFE_CALL(cudaMalloc((void**)&hi_send_buf, hi_buf_size));
      hi_send_buf = (char*)lo_send_buf + lo_buf_size;
      CUDA_SAFE_CALL(cudaMalloc((void**)&lo_recv_buf, lo_buf_size + hi_buf_size));
      //CUDA_SAFE_CALL(cudaMalloc((void**)&hi_recv_buf, hi_buf_size));
      hi_recv_buf = (char*)lo_recv_buf + lo_buf_size;
    }

    if(useHostBuffer){
      CUDA_SAFE_CALL(cudaMallocHost((void**)&lo_send_host_buf, lo_buf_size + hi_buf_size));
      //CUDA_SAFE_CALL(cudaMallocHost((void**)&hi_send_host_buf, hi_buf_size));
      hi_send_host_buf = (char*)lo_send_host_buf + lo_buf_size;
      CUDA_SAFE_CALL(cudaMallocHost((void**)&lo_recv_host_buf, lo_buf_size + hi_buf_size));
      //CUDA_SAFE_CALL(cudaMallocHost((void**)&hi_recv_host_buf, hi_buf_size));
      hi_recv_host_buf = (char*)lo_recv_host_buf + lo_buf_size;

      mpi_lo_send_buf = lo_send_host_buf;
      mpi_lo_recv_buf = lo_recv_host_buf;
      mpi_hi_send_buf = hi_send_host_buf;
      mpi_hi_recv_buf = hi_recv_host_buf;
    }else{
      mpi_lo_send_buf = lo_send_buf;
      mpi_lo_recv_buf = lo_recv_buf;
      mpi_hi_send_buf = hi_send_buf;
      mpi_hi_recv_buf = hi_recv_buf;
    }
  }

  /* // for upper reflect */
  /* if (uwidth){ */
  /*   hi_buf_size = uwidth * blocklength * count; */
  /*   if (!packVector || */
  /* 	(_XMPF_running && target_dim == ndims - 1) || */
  /* 	(_XMPC_running && target_dim == 0)){ */
  /*     hi_send_buf = hi_send_array; */
  /*     hi_recv_buf = hi_recv_array; */
  /*   } else { */
  /*     //      hi_send_buf = _XACC_gpu_alloc(hi_buf_size); */
  /*     //      hi_recv_buf = _XACC_gpu_alloc(hi_buf_size); */
  /*     CUDA_SAFE_CALL(cudaMalloc((void**)&hi_send_buf, hi_buf_size)); */
  /*     CUDA_SAFE_CALL(cudaMalloc((void**)&hi_recv_buf, hi_buf_size)); */
  /*   } */

  /*   if(useHostBuffer){ */
  /*     CUDA_SAFE_CALL(cudaMallocHost((void**)&hi_send_host_buf, hi_buf_size)); */
  /*     CUDA_SAFE_CALL(cudaMallocHost((void**)&hi_recv_host_buf, hi_buf_size)); */
  /*     mpi_hi_send_buf = hi_send_host_buf; */
  /*     mpi_hi_recv_buf = hi_recv_host_buf; */
  /*   }else{ */
  /*     mpi_hi_send_buf = hi_send_buf; */
  /*     mpi_hi_recv_buf = hi_recv_buf; */
  /*   } */
  /* } */


  //
  // initialize communication
  //

  int src, dst;
  int is_periodic = reflect->is_periodic;
  //int num_devices = arrays_desc->device_type->size;
  _XACC_device_t *device = arrays_desc->device_type;
  if (!is_periodic && my_pos == lb_pos){ // && (target_dim != 0 || target_device == 0)){ // no periodic
    lo_rank = MPI_PROC_NULL;
  }else if(target_dim == 0 && target_device != device->lb){
    lo_rank = MPI_PROC_NULL; //lo_rank = my_rank;
  }

  if (!is_periodic && my_pos == ub_pos){ // && (target_dim != 0 || target_device == num_devices - 1)){ // no periodic
    hi_rank = MPI_PROC_NULL;
  }else if(target_dim == 0 && target_device != device->ub){
    hi_rank = MPI_PROC_NULL; //hi_rank = my_rank;
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

  //printf("lo_recv pos=%lld, from(%d) @rank=%d,dev=%d\n", (long long )(lo_recv_buf - array_addr), src, my_rank, target_device);
  //printf("lo_send pos=%lld, to(%d) @rank=%d,dev=%d\n", (long long )(lo_send_buf - array_addr), dst, my_rank, target_device);
  int tag_offset;
  if(target_dim == 0){
    tag_offset = 0;
  }else{
    tag_offset = target_device * 1000;
  }
  
  MPI_Recv_init(mpi_lo_recv_buf, 1, reflect->datatype_lo, src,
		_XMP_N_MPI_TAG_REFLECT_LO + tag_offset, *comm, &reflect->req[0]);
  MPI_Send_init(mpi_lo_send_buf, 1, reflect->datatype_lo, dst,
		_XMP_N_MPI_TAG_REFLECT_LO + tag_offset, *comm, &reflect->req[1]);


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

  //printf("hi_recv pos=%lld, from(%d) @rank=%d,dev=%d\n", (long long )(hi_recv_buf - array_addr), src, my_rank, target_device);
  //printf("hi_send pos=%lld, to(%d) @rank=%d,dev=%d\n", (long long )(hi_send_buf - array_addr), dst, my_rank, target_device);
  MPI_Recv_init(mpi_hi_recv_buf, 1, reflect->datatype_hi, src,
		_XMP_N_MPI_TAG_REFLECT_HI + tag_offset, *comm, &reflect->req[2]);
  MPI_Send_init(mpi_hi_send_buf, 1, reflect->datatype_hi, dst,
		_XMP_N_MPI_TAG_REFLECT_HI + tag_offset, *comm, &reflect->req[3]);

  if(target_dim == 1){
    cudaStream_t *lo_stream;
    /* if(stream_pool == NULL){ */
    /*   stream_pool = (cudaStream_t*)_XMP_alloc(sizeof(cudaStream_t)*64); */
    /* } */
    /* lo_stream = stream_pool; */
    /* stream_pool++; */
    lo_stream = (cudaStream_t*)_XMP_alloc(sizeof(cudaStream_t));
  CUDA_SAFE_CALL(cudaStreamCreate(lo_stream));
  fprintf(stderr, "createdStream(%p)\n", lo_stream);
  reflect->lo_async_id = (void*)lo_stream;
  }else{
    reflect->lo_async_id = NULL;
  }
  if(0){//if((src != MPI_PROC_NULL && dst != MPI_PROC_NULL && (lo_buf_size / type_size) <= useSingleStreamLimit) || !useHostBuffer){
    reflect->hi_async_id = NULL;
  }else{
    cudaStream_t *hi_stream = (cudaStream_t*)_XMP_alloc(sizeof(cudaStream_t));
    CUDA_SAFE_CALL(cudaStreamCreate(hi_stream));
    reflect->hi_async_id = (void*)hi_stream;
  }
  //fprintf(stderr, "lo stream p=%p, hi stream p =%p\n", reflect->lo_async_id, reflect->hi_async_id);

  cudaEvent_t *event_packed = (cudaEvent_t*)_XMP_alloc(sizeof(cudaEvent_t));
  cudaEvent_t *event_unpacked = (cudaEvent_t*)_XMP_alloc(sizeof(cudaEvent_t));
  CUDA_SAFE_CALL(cudaEventCreateWithFlags(event_packed, cudaEventDisableTiming));
  CUDA_SAFE_CALL(cudaEventCreateWithFlags(event_unpacked, cudaEventDisableTiming));
  reflect->event_packed = event_packed;
  reflect->event_unpacked = event_unpacked;

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

  if(useHostBuffer){
    reflect->lo_send_host_buf = lo_send_host_buf;
    reflect->lo_recv_host_buf = lo_recv_host_buf;
    reflect->hi_send_host_buf = hi_send_host_buf;
    reflect->hi_recv_host_buf = hi_recv_host_buf;
  }

  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;
}

static void reflect_pack_start(_XMP_reflect_sched_t *reflect, size_t type_size)
{
  cudaStream_t *st_lo = (cudaStream_t*)(reflect->lo_async_id);
  cudaStream_t *st_hi = (cudaStream_t*)(reflect->hi_async_id);

  {
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    char *lo_send_buf,*lo_send_array;
    char *hi_send_buf,*hi_send_array;
    if (reflect->hi_rank != MPI_PROC_NULL){
      lo_send_buf = (char*)reflect->lo_send_buf;
      lo_send_array = (char*)reflect->lo_send_array;
    }else{
      lo_send_buf = lo_send_array = NULL;
    }
    if (reflect->lo_rank != MPI_PROC_NULL){
      hi_send_buf = (char*)reflect->hi_send_buf;
      hi_send_array = (char*)reflect->hi_send_array;
    }else{
      hi_send_buf = hi_send_array = NULL;
    }
    if(lo_send_buf == NULL && hi_send_buf == NULL) return;

  if(lo_send_array)
#if 1
    _XACC_gpu_pack_vector_async(lo_send_buf, lo_send_array, reflect->count, lo_width * reflect->blocklength, reflect->stride,
    				type_size, *st_lo);
#else
      _XACC_gpu_pack_vector_async_test(lo_send_buf, lo_send_array, type_size, *st_lo);
#endif
  if(hi_send_array)
#if 1
    _XACC_gpu_pack_vector_async(hi_send_buf, hi_send_array, reflect->count, hi_width * reflect->blocklength, reflect->stride,
    				type_size, *st_lo);
#else
    _XACC_gpu_pack_vector_async_test(hi_send_buf, hi_send_array, type_size, *st_lo);
#endif

  //  fprintf(stderr,"pack stream =%p\n", st_lo);

  }
  return;
  if(useKernelPacking){
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    char *lo_send_buf,*lo_send_array;
    char *hi_send_buf,*hi_send_array;
    if (reflect->hi_rank != MPI_PROC_NULL){
      lo_send_buf = (char*)reflect->lo_send_buf;
      lo_send_array = (char*)reflect->lo_send_array;
    }else{
      lo_send_buf = lo_send_array = NULL;
    }
    if (reflect->lo_rank != MPI_PROC_NULL){
      hi_send_buf = (char*)reflect->hi_send_buf;
      hi_send_array = (char*)reflect->hi_send_array;
    }else{
      hi_send_buf = hi_send_array = NULL;
    }
    if(lo_send_buf == NULL && hi_send_buf == NULL) return;

    if(reflect->lo_rank != MPI_PROC_NULL && reflect->hi_rank != MPI_PROC_NULL && reflect->hi_async_id == NULL){
      _XACC_gpu_pack_vector2_async(lo_send_buf,
				   lo_send_array,
				   lo_width * reflect->blocklength,
				   reflect->stride,
				   hi_send_buf,
				   hi_send_array,
				   hi_width * reflect->blocklength,
				   reflect->stride,
				   reflect->count, type_size, *st_lo);
      //      cudaEvent_t *event_packed = (cudaEvent_t*)(reflect->event_packed);
      //      CUDA_SAFE_CALL(cudaEventRecord(*event_packed, *st_lo));
      //      CUDA_SAFE_CALL(cudaStreamWaitEvent(*st_hi, *event_packed, 0));
    }else{
      if(lo_send_array)
      _XACC_gpu_pack_vector_async(lo_send_buf, lo_send_array, reflect->count, lo_width * reflect->blocklength, reflect->stride,
				  type_size, *st_lo);
      if(hi_send_array)
      _XACC_gpu_pack_vector_async(hi_send_buf, hi_send_array, reflect->count, hi_width * reflect->blocklength, reflect->stride,
				  type_size, *st_hi);
    }
  }else{
    /* // for lower reflect */
    /* int lo_width = reflect->lo_width; */
    /* if (reflect->hi_rank != MPI_PROC_NULL){ */
    /*   CUDA_SAFE_CALL(cudaMemcpy2DAsync(reflect->lo_send_buf, */
    /* 				       lo_width * reflect->blocklength, */
    /* 				       reflect->lo_send_array, */
    /* 				       reflect->stride, */
    /* 				       lo_width * reflect->blocklength, */
    /* 				       reflect->count, cudaMemcpyDefault, *st_lo)); */
    /* } */

    /* // for upper reflect */
    /* int hi_width = reflect->hi_width; */
    /* if (reflect->lo_rank != MPI_PROC_NULL){ */
    /*   CUDA_SAFE_CALL(cudaMemcpy2DAsync(reflect->hi_send_buf, */
    /* 				       hi_width * reflect->blocklength, */
    /* 				       reflect->hi_send_array, */
    /* 				       reflect->stride, */
    /* 				       hi_width * reflect->blocklength, */
    /* 				       reflect->count, cudaMemcpyDefault, *st_hi)); */
    /* } */
  }
}


static void reflect_pack_wait(_XMP_reflect_sched_t *reflect)
{
  //cudaEvent_t *ev = (cudaEvent_t*)reflect->event_packed;
  //CUDA_SAFE_CALL(cudaEventSynchronize(*ev));
  //  int lo_width = reflect->lo_width;
  //  int hi_width = reflect->hi_width;
/*   if (reflect->lo_rank != MPI_PROC_NULL || */
/*       reflect->hi_rank != MPI_PROC_NULL){ */
/*     cudaStream_t *st = (cudaStream_t*)(reflect->lo_async_id); */
/* #if 1 */
/*     CUDA_SAFE_CALL(cudaStreamSynchronize(*st)); */
/* #else */
/*     cudaEvent_t *ev = (cudaEvent_t*)(reflect->event_packed); */
/*     CUDA_SAFE_CALL(cudaEventRecord(*ev, *st)); */
/*     CUDA_SAFE_CALL(cudaEventSynchronize(*ev)); */
/* #endif */
/*   } */

  if (reflect->hi_rank != MPI_PROC_NULL){
    cudaStream_t *st_lo = (cudaStream_t*)(reflect->lo_async_id);
    if(st_lo !=NULL)
      CUDA_SAFE_CALL(cudaStreamSynchronize(*st_lo));
  }
  if (reflect->lo_rank != MPI_PROC_NULL){
    cudaStream_t *st_lo = (cudaStream_t*)(reflect->lo_async_id);
    if(st_lo != NULL){
      CUDA_SAFE_CALL(cudaStreamSynchronize(*st_lo));
    }
  }
}

static void reflect_unpack_wait(_XMP_reflect_sched_t *reflect)
{
  //cudaEvent_t *ev = (cudaEvent_t*)reflect->event_unpacked;
  //CUDA_SAFE_CALL(cudaEventSynchronize(*ev));
/*   if (reflect->lo_rank != MPI_PROC_NULL || */
/*       reflect->hi_rank != MPI_PROC_NULL){ */
/*     cudaStream_t *st = (cudaStream_t*)(reflect->lo_async_id); */
/* #if 1 */
/*     CUDA_SAFE_CALL(cudaStreamSynchronize(*st)); */
/* #else */
/*     cudaEvent_t *ev = (cudaEvent_t*)(reflect->event_unpacked); */
/*     CUDA_SAFE_CALL(cudaEventRecord(*ev, *st)); */
/*     CUDA_SAFE_CALL(cudaEventSynchronize(*ev)); */
/* #endif */

  if (reflect->lo_rank != MPI_PROC_NULL){
    cudaStream_t *st_lo = (cudaStream_t*)(reflect->lo_async_id);
    //    fprintf("unpackwait st_lo=%p\n", st_lo);
    if(st_lo != NULL){
      CUDA_SAFE_CALL(cudaStreamSynchronize(*st_lo));
    }
  }
  if (reflect->hi_rank != MPI_PROC_NULL){
    cudaStream_t *st_hi = (cudaStream_t*)(reflect->lo_async_id);
    if(st_hi != NULL){
      CUDA_SAFE_CALL(cudaStreamSynchronize(*st_hi));
    }
  }
}
static void reflect_update_host(_XMP_reflect_sched_t *reflect)
{
  cudaStream_t *st_lo = (cudaStream_t*)(reflect->lo_async_id);
  cudaStream_t *st_hi = (cudaStream_t*)(reflect->hi_async_id);
  if(reflect->hi_rank != MPI_PROC_NULL && reflect->lo_rank != MPI_PROC_NULL && st_hi == NULL){
    int lo_width = reflect->lo_width;
    int hi_width = reflect->hi_width;
    size_t buf_size = (lo_width + hi_width) * reflect->blocklength * reflect->count;
    CUDA_SAFE_CALL(cudaMemcpyAsync(reflect->lo_send_host_buf,
				   reflect->lo_send_buf,
				   buf_size,
				   cudaMemcpyDeviceToHost, *st_lo));
    return;
  }
  if(reflect->hi_rank != MPI_PROC_NULL){
    int lo_width = reflect->lo_width;
    size_t lo_buf_size = lo_width * reflect->blocklength * reflect->count;
    CUDA_SAFE_CALL(cudaMemcpyAsync(reflect->lo_send_host_buf,
				   reflect->lo_send_buf,
				   lo_buf_size,
				   cudaMemcpyDeviceToHost, *st_lo));
  }
  if(reflect->lo_rank != MPI_PROC_NULL){
    int hi_width = reflect->hi_width;
    size_t hi_buf_size = hi_width * reflect->blocklength * reflect->count;
    CUDA_SAFE_CALL(cudaMemcpyAsync(reflect->hi_send_host_buf,
				   reflect->hi_send_buf,
				   hi_buf_size,
				   cudaMemcpyDeviceToHost, *st_hi));
  }
}

static void reflect_pack_start_all(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;
  _XACC_device_t *device = arrays_desc->device_type;
  //#pragma omp for
  //  for(int i = device->lb; i <= device->ub; i += device->step){

  {
    int i = device->lb + device->step*omp_get_thread_num();
    _XACC_array_t *array_desc = arrays_desc->array + i;
    if(useKernelPacking){
      //#ifndef _USE_OMP
      CUDA_SAFE_CALL(cudaSetDevice(i));
      //#endif
    }

    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if(j != 0){
	  reflect_pack_start(reflect, arrays_desc->type_size);
	  TLOG_LOG(TLOG_EVENT_9);
	}
	if(useHostBuffer){
	  //reflect_update_host(reflect);
	}
      }
    }
  }
}

static void reflect_pack_wait_all(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;

  _XACC_device_t *device = arrays_desc->device_type;
  //#pragma omp master
  //#pragma omp for
  //  for(int i = device->lb; i <= device->ub; i += device->step){
  {
    int i = device->lb + device->step * omp_get_thread_num();
    _XACC_array_t *array_desc = arrays_desc->array + i;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if( j != 0 || useHostBuffer){
	    TLOG_LOG(TLOG_EVENT_4_IN);
	  reflect_pack_wait(reflect);
	  //	  TLOG_LOG(TLOG_EVENT_1);
	    TLOG_LOG(TLOG_EVENT_4_OUT);
	}
      }
    }
  }
}

static void reflect_unpack_start_all(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;

  _XACC_device_t *device = arrays_desc->device_type;
#pragma omp for
  for(int i = device->lb; i <= device->ub; i += device->step){  
    _XACC_array_t *array_desc = arrays_desc->array + i;
    if(useKernelPacking){
#ifndef _USE_OMP
      CUDA_SAFE_CALL(cudaSetDevice(i));
#endif
    }

    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if(useHostBuffer){
	  //reflect_update_device(reflect);
	}
	if(j != 0){
	  //reflect_unpack_start(reflect, arrays_desc->type_size);
	  TLOG_LOG(TLOG_EVENT_1);
	}
      }
    }
  }
}

static void reflect_unpack_wait_all(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;

  _XACC_device_t *device = arrays_desc->device_type;
#pragma omp master
  for(int i = device->lb; i <= device->ub; i += device->step){

    _XACC_array_t *array_desc = arrays_desc->array + i;
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if( j != 0 || useHostBuffer){
	  reflect_unpack_wait(reflect);
	  TLOG_LOG(TLOG_EVENT_1);
	}
      }
    }
  }
}


static void _XACC_reflect_do_inter_wait_dev(_XACC_arrays_t *arrays_desc, int i)
{
  int dim = arrays_desc->dim;
  _XACC_array_t *array_desc = arrays_desc->array + i;

  for(int j = 0; j < dim; j++){
    if(j==0)continue;
    _XACC_array_info_t *ai = array_desc->info + j;
    if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
      _XMP_reflect_sched_t *reflect = ai->reflect_sched;

      //MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
      TLOG_LOG(TLOG_EVENT_1);
    }
  }
}


static void _XACC_reflect_do_inter_wait_dim0(_XACC_arrays_t *arrays_desc)
{
#pragma omp master
  {
    _XACC_array_t *array_desc = arrays_desc->array + arrays_desc->device_type->lb;
    _XACC_array_info_t *ai = array_desc->info + 0;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    //MPI_Wait(reflect->req + 0, MPI_STATUS_IGNORE); //lo recv                                                                                                                                      
    //MPI_Wait(reflect->req + 3, MPI_STATUS_IGNORE); //hi send                                                                                                                                      
  }
#pragma omp master
  {
    _XACC_array_t *array_desc = arrays_desc->array + arrays_desc->device_type->ub;
    _XACC_array_info_t *ai = array_desc->info + 0;
    _XMP_reflect_sched_t *reflect = ai->reflect_sched;
    //MPI_Wait(reflect->req + 1, MPI_STATUS_IGNORE); //lo send                                                                                                                                      
    //MPI_Wait(reflect->req + 2, MPI_STATUS_IGNORE); //hi recv                                                                                                                                      
  }
  TLOG_LOG(TLOG_EVENT_1);
}


static void _XACC_reflect_do_inter_start(_XACC_arrays_t *arrays_desc)
{
  if(packVector){
    TLOG_LOG(TLOG_EVENT_3_IN);
    reflect_pack_start_all(arrays_desc);
    TLOG_LOG(TLOG_EVENT_3_OUT);
    //    reflect_pack_wait_all(arrays_desc);
  }

/*   TLOG_LOG(TLOG_EVENT_1_IN); */
/* #pragma omp barrier */
/*   TLOG_LOG(TLOG_EVENT_1_OUT); */

//  TLOG_LOG(TLOG_EVENT_4_IN);
  reflect_pack_wait_all(arrays_desc);
  //_XACC_reflect_do_inter_start_dim0(arrays_desc);

  _XACC_device_t *device = arrays_desc->device_type;
/* #pragma omp master */
/*   for(int i = device->lb; i <= device->ub; i += device->step){ */
/*    _XACC_reflect_do_inter_start_dev(arrays_desc, i); */
/*   } */
//  TLOG_LOG(TLOG_EVENT_4_OUT);
}

static void _XACC_reflect_do_inter_wait(_XACC_arrays_t *arrays_desc)
{
  TLOG_LOG(TLOG_EVENT_6_IN);
  _XACC_reflect_do_inter_wait_dim0(arrays_desc);

  _XACC_device_t *device = arrays_desc->device_type;
#pragma omp master
  for(int i = device->lb; i <= device->ub; i += device->step){   
   _XACC_reflect_do_inter_wait_dev(arrays_desc, i);
  }
  TLOG_LOG(TLOG_EVENT_6_OUT);

  if(packVector){
#ifdef _USE_OMP
    TLOG_LOG(TLOG_EVENT_1_IN);
#pragma omp barrier
    TLOG_LOG(TLOG_EVENT_1_OUT);
#endif
    TLOG_LOG(TLOG_EVENT_7_IN);
    reflect_unpack_start_all(arrays_desc);
    TLOG_LOG(TLOG_EVENT_7_OUT);
    TLOG_LOG(TLOG_EVENT_8_IN);
    reflect_unpack_wait_all(arrays_desc);
    TLOG_LOG(TLOG_EVENT_8_OUT);
  }
}

void _XACC_reflect_do(_XACC_arrays_t *arrays_desc){
  int  numDevices = arrays_desc->device_type->size;
  TLOG_LOG(TLOG_EVENT_8);
#pragma omp parallel num_threads(numDevices)
  {
  _XACC_device_t *device = arrays_desc->device_type;
  //#pragma omp for
  //  for(int dev = device->lb; dev <= device->ub; dev += device->step){
  {
    int dev = omp_get_thread_num();
    CUDA_SAFE_CALL(cudaSetDevice(dev));
  }
    
  //start comm
  //  _XACC_reflect_do_inter_start(arrays_desc);
  if(packVector){
    TLOG_LOG(TLOG_EVENT_3_IN);
    //reflect_pack_start_all(arrays_desc);
   {
  int dim = arrays_desc->dim;
  _XACC_device_t *device = arrays_desc->device_type;
#pragma omp for
  for(int i = device->lb; i <= device->ub; i += device->step){
  // {
  //    int i = device->lb + device->step*omp_get_thread_num();
    _XACC_array_t *array_desc = arrays_desc->array + i;
    if(useKernelPacking){
      //#ifndef _USE_OMP
      //      CUDA_SAFE_CALL(cudaSetDevice(i));
      //#endif
    }

    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if(j != 0){
	  reflect_pack_start(reflect, arrays_desc->type_size);
	  TLOG_LOG(TLOG_EVENT_9);
	}
	if(useHostBuffer){
	  //reflect_update_host(reflect);
	}
      }
    }
  }

    }
    TLOG_LOG(TLOG_EVENT_3_OUT);
  }
  //  reflect_pack_wait_all(arrays_desc);
  {
  int dim = arrays_desc->dim;

  _XACC_device_t *device = arrays_desc->device_type;
  //#pragma omp master
 #pragma omp for
  for(int i = device->lb; i <= device->ub; i += device->step){
    //  {
    //    int i = device->lb + device->step * omp_get_thread_num();
    _XACC_array_t *array_desc = arrays_desc->array + i;
    //    CUDA_SAFE_CALL(cudaSetDevice(i));
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
	_XMP_reflect_sched_t *reflect = ai->reflect_sched;
	if( j != 0 || useHostBuffer){
	    TLOG_LOG(TLOG_EVENT_4_IN);
	  reflect_pack_wait(reflect);
	  //	  TLOG_LOG(TLOG_EVENT_1);
	    TLOG_LOG(TLOG_EVENT_4_OUT);
	}
      }
    }
  }

  }

  //_XACC_reflect_do_intra_start(arrays_desc);

  //wait comm
  _XACC_reflect_do_inter_wait(arrays_desc);
  //_XACC_reflect_do_intra_wait(arrays_desc);
  }
  TLOG_LOG(TLOG_EVENT_4);
  usleep(100);
}
