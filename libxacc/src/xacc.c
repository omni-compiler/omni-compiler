#include <stdio.h>
#if 1
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


static _XACC_device_t *_XACC_current_device = NULL;
typedef _XACC_device_t xacc_device_t;

int xacc_device_size(xacc_device_t device);
int xacc_get_num_devices(xacc_device_t device);
void xacc_set_device(xacc_device_t device);
acc_device_t xacc_get_device(); // current
void xacc_set_device_num(int num, xacc_device_t device);
//int xacc_get_device_num(xacc_device_t device); // current


acc_device_t xacc_get_current_device(){
  return _XACC_current_device->acc_device;
}
int xacc_get_device_num(){
  return -1; //tmporary
}

//void acc_set_device_type(acc_device_t device);
//void acc_set_device_num(int num);


//internal functions
int _XACC_get_num_current_devices(){
  return _XACC_current_device->size;
}

acc_device_t _XACC_get_current_device(){
  return _XACC_current_device->acc_device;
}

void _XACC_init_device(_XACC_device_t** desc, acc_device_t device, int lower, int upper, int step)
{
  _XACC_device_t* xacc_device = (_XACC_device_t*)_XMP_alloc(sizeof(_XACC_device_t));
  xacc_device->acc_device = device;
  xacc_device->lb = lower;
  xacc_device->ub = upper;
  xacc_device->step = step;
  xacc_device->size = upper - lower + 1; //must consider step

  *desc = xacc_device;
}

void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step)
{
  _XACC_device_t* xacc_device = (_XACC_device_t*)desc;
  *lower = xacc_device->lb;
  *upper = xacc_device->ub;
  *step  = xacc_device->step;
}
void _XACC_get_current_device_info(int* lower, int* upper, int* step)
{
  _XACC_get_device_info(_XACC_current_device, lower, upper, step);
}




void _XACC_init_layouted_array(_XACC_arrays_t **arrays, _XMP_array_t* alignedArray, _XACC_device_t* device){
  _XACC_arrays_t* layoutedArray = (_XACC_arrays_t*)_XMP_alloc(sizeof(_XACC_arrays_t));

  layoutedArray->device_type = device;
  //alignedArray->device_type = device;
  int dim = alignedArray->dim;
  int num_devices = device->size;
  layoutedArray->device_array = (_XACC_array_t*)_XMP_alloc(sizeof(_XACC_array_t) * num_devices);
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t* d_array_info = (_XACC_array_info_t*)_XMP_alloc(sizeof(_XACC_array_info_t) * dim);
    _XMP_array_info_t *h_array_info = alignedArray->info;
    layoutedArray->device_array[dev].info = d_array_info;
    int i;
    for(i = 0; i <dim; i++){
      d_array_info[i].device_layout_manner = _XMP_N_DIST_DUPLICATION;
      d_array_info[i].par_lower = h_array_info[i].par_lower;
      d_array_info[i].par_upper = h_array_info[i].par_upper;
      d_array_info[i].par_stride = h_array_info[i].par_stride;
      d_array_info[i].par_size = h_array_info[i].par_size;

      d_array_info[i].alloc_size = h_array_info[i].alloc_size;

      d_array_info[i].local_lower = h_array_info[i].local_lower;
      d_array_info[i].local_upper = h_array_info[i].local_upper;
      d_array_info[i].local_stride = h_array_info[i].local_stride;

      d_array_info[i].shadow_size_lo = h_array_info[i].shadow_size_lo;
      d_array_info[i].shadow_size_hi = h_array_info[i].shadow_size_hi;

      d_array_info[i].reflect_sched = NULL;

    printf("dim=%d, host par (%d, %d, %d) local(%d, %d, %d, %d)\n",
           dim,
           h_array_info[i].par_lower,
           h_array_info[i].par_upper,
           h_array_info[i].par_stride,
           h_array_info[i].local_lower,
           h_array_info[i].local_upper,
           h_array_info[i].local_stride,
           h_array_info[i].alloc_size);

    }
  }
  layoutedArray->dim = dim;
  layoutedArray->hostptr = alignedArray->array_addr_p;
  layoutedArray->type_size = alignedArray->type_size;
  layoutedArray->xmp_array = alignedArray;
  *arrays = layoutedArray;
}

void _XACC_split_layouted_array_DUPLICATION(_XACC_arrays_t* array_desc, int dim){
  /* _XMP_array_info_t *h_array_info = &(array_desc->info[dim]); */

  /* int num_devices = array_desc->device_type->size; */
  /* for(int dev = 0; dev < num_devices; dev++){ */
  /*   _XACC_array_info_t *d_array_info = &(array_desc->device_array[dev].info[dim]); */
  /*   d_array_info->device_layout_manner = _XMP_N_DIST_DUPLICATION; */

  /*   d_array_info->par_lower = h_array_info->par_lower; */
  /*   d_array_info->par_upper = h_array_info->par_upper; */
  /*   d_array_info->par_stride = h_array_info->par_stride; */

  /*   d_array_info->alloc_size = h_array_info->alloc_size; */

  /*   d_array_info->local_lower = h_array_info->local_lower; */
  /*   d_array_info->local_upper = h_array_info->local_upper; */
  /*   d_array_info->local_stride = h_array_info->local_stride; */



    /* printf("dim=%d, dup par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
    /*        dim, */
    /*        d_array_info->par_lower, */
    /*        d_array_info->par_upper, */
    /*        d_array_info->par_stride, */
    /*        d_array_info->local_lower, */
    /*        d_array_info->local_upper, */
    /*        d_array_info->local_stride, */
    /*        d_array_info->alloc_size); */

  /* } */
}

void _XACC_split_layouted_array_BLOCK(_XACC_arrays_t* array_desc, int dim){
  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t *d_array_info = &(array_desc->device_array[dev].info[dim]);
    d_array_info->device_layout_manner = _XMP_N_DIST_BLOCK;

    unsigned long long size = _XMP_M_CEILi(d_array_info->par_size, num_devices);
    //d_array_info->par_stride;// = h_array_info->par_stride;    
    d_array_info->par_lower += size * dev;
    
    if(dev != num_devices - 1){
      d_array_info->par_upper = d_array_info->par_lower + size - 1;
    }else{
      //      d_array_info->par_upper = h_array_info->par_upper;
    }
    //d_array_info->local_stride = h_array_info->local_stride;
    d_array_info->local_lower += size * dev;
    if(dev != num_devices - 1){
      d_array_info->local_upper = d_array_info->local_lower + size - 1;
    }else{
      //      d_array_info->local_upper = h_array_info->local_upper;
    }

    d_array_info->alloc_size = size 
      + d_array_info->shadow_size_lo 
      + d_array_info->shadow_size_hi;

/* #if not */
/*     int size = h_array_info->par_size / num_devices; */
/*     d_array_info->lower = h_array_info->par_lower + size * dev; */
/*     d_array_info->upper = h_array_info->par_lower + size * (dev+1); */
/*     d_array_info->stride =h_array_info->par_stride; */
/*     d_array_info->alloc_size = size; */
/* #endif */

    
    printf("dim=%d, block par (%d, %d, %d) local(%d, %d, %d, %d)\n",
           dim,
           d_array_info->par_lower,
           d_array_info->par_upper,
           d_array_info->par_stride,
           d_array_info->local_lower,
           d_array_info->local_upper,
           d_array_info->local_stride,
           d_array_info->alloc_size);

    /* printf("block par (%d, %d, %d, %d)\n",  */
    /*        d_array_info->lower, */
    /*        d_array_info->upper, */
    /*        d_array_info->stride, */
    /*        d_array_info->size); */
  }
}

void _XACC_set_shadow_NORMAL(_XACC_arrays_t* array_desc, int dim , int lo, int hi)
{
  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t *d_array_info = &(array_desc->device_array[dev].info[dim]);

      int d_lo = lo;
      int d_hi = hi;
      int h_lo = d_array_info->shadow_size_lo;
      int h_hi = d_array_info->shadow_size_hi;

      if((d_lo != 0 || d_hi != 0) && (h_lo != 0 || h_hi != 0)){
        if(d_lo != h_lo || d_hi != h_hi){
          //has error
        }
      }else if((d_lo == 0 && d_hi == 0) && (h_lo == 0 && h_hi == 0)){
        //no error
      }else if((d_lo == 0 && d_hi == 0) && (h_lo != 0 || h_hi != 0)){
        //no
      }else{
        //only d has shadow
        if(dev != 0){
          d_array_info->shadow_size_lo = lo;
          d_array_info->local_lower += lo;
          d_array_info->local_upper += lo;
        }
        if(dev != num_devices -1){
          d_array_info->shadow_size_hi = hi;
          d_array_info->local_upper += hi;
        }
      }

    /* d_array_info->shadow_size_lo = lo; */
    /* d_array_info->shadow_size_hi = hi; */

    printf("shadow dim=%d, lo=%d,hi=%d\n", dim, lo, hi);
    /* printf("dim=%d, par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
    /*        dim, */
    /*        d_array_info->par_lower, */
    /*        d_array_info->par_upper, */
    /*        d_array_info->par_stride, */
    /*        d_array_info->local_lower, */
    /*        d_array_info->local_upper, */
    /*        d_array_info->local_stride, */
    /*        d_array_info->alloc_size); */
  }    


}

void _XACC_calc_size(_XACC_arrays_t* array_desc){
  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    unsigned long long device_acc = 1;
    unsigned long long device_offset = 0;
    int dim = array_desc->dim;
    _XACC_array_t *d_array_desc = &(array_desc->device_array[dev]);
    for(int i = dim - 1; i > 0; i--){
      _XACC_array_info_t* info = &(d_array_desc->info[i]);
      info->dim_acc = device_acc;
      device_offset += device_acc * (info->local_lower - info->shadow_size_lo);
      device_acc *= info->alloc_size;
    }
    {
      _XACC_array_info_t* info = &(d_array_desc->info[0]);
      info->dim_acc = device_acc;
      //old
      //device_offset += device_acc * info->local_lower;
      //device_acc *= (info->alloc_size - info->shadow_size_lo - info->shadow_size_hi);
      //new
      //device_offset += device_acc * (info->local_lower - info->shadow_size_lo);
      //device_acc *= (info->alloc_size);

      
      d_array_desc->alloc_size = device_acc * info->alloc_size;
      d_array_desc->alloc_offset = device_offset + device_acc * (info->local_lower - info->shadow_size_lo);
      d_array_desc->copy_size = device_acc * (info->alloc_size - info->shadow_size_lo - info->shadow_size_hi);
      d_array_desc->copy_offset = device_offset + device_acc * info->local_lower;
    }

    printf("alloc(%llu, %llu), copy(%llu, %llu)@dev\n",
           d_array_desc->alloc_offset,
           d_array_desc->alloc_size,
           d_array_desc->copy_offset,
           d_array_desc->copy_size);
  }
}

static _XACC_array_t* get_device_array(_XACC_arrays_t* array_desc, int deviceNum)
{
  _XACC_device_t* device = array_desc->device_type;
  int lower = device->lb;
  int step = device->step;
  
  int n = (deviceNum - lower) / step;
  return &(array_desc->device_array[n]);
}

void _XACC_get_size(_XACC_arrays_t* array_desc, unsigned long long* offset,
                    unsigned long long* size, int deviceNum)
{
  _XACC_array_t* device_array = get_device_array(array_desc, deviceNum);
  *size = device_array->alloc_size;
  *offset = device_array->alloc_offset;
}

void _XACC_get_copy_size(_XACC_arrays_t* array_desc, unsigned long long* offset,
                         unsigned long long* size, int deviceNum)
{
  _XACC_array_t* device_array = get_device_array(array_desc, deviceNum);
  *size = device_array->copy_size;
  *offset = device_array->copy_offset;
}

void _XACC_sched_loop_layout_BLOCK(int init,
                              int cond,
                              int step,
                              int* sched_init,
                              int* sched_cond,
                              int* sched_step,
                              _XACC_arrays_t* array_desc,
                              int dim,
                              int deviceNum)
{
  _XACC_array_t* device_array = get_device_array(array_desc, deviceNum);
  _XACC_array_info_t* info = &device_array->info[dim];

  int lb = info->local_lower - info->shadow_size_lo;
  int ub = info->local_upper - info->shadow_size_lo;;

  int r_init, r_cond;
  
  if(init < lb){
    r_init = lb;
  }else if(init > ub){
    r_init = ub + 1;
  }else{
    r_init = init;
  }

  if(cond <= lb){
    r_cond = lb;
  }else if(cond > ub){
    r_cond = ub + 1;
  }else{
    r_cond = cond;
  }

  //printf("lb=%d, ub =%d, rinit=%d,rcond=%d\n", lb, ub,r_init, r_cond);

  *sched_init = r_init;// - lb;//info->local_lower - info->shadow_size_lo;
  *sched_cond = r_cond;// - lb;//info->local_upper - info->shadow_size_lo + 1;
  *sched_step = info->local_stride;

  //printf("loop org(%d, %d, %d), mod(%d, %d, %d)@%d\n",
  //       init, cond,step, *sched_init, *sched_cond, *sched_step, deviceNum);
}

void _XACC_set_deviceptr(_XACC_arrays_t *arrays_desc, void *deviceptr, int deviceNum)
{
  _XACC_array_t* arrayOnDevice = get_device_array(arrays_desc, deviceNum);
  arrayOnDevice->deviceptr = deviceptr;
  printf("deviceptr=%p@%d\n", deviceptr, deviceNum);
}

static void _XACC_reflect_sched_dim(_XACC_arrays_t *a, int target_device, int target_dim);

void _XACC_reflect_init(_XACC_arrays_t *arrays_desc)
{
  _XACC_device_t *device = arrays_desc->device_type;
  int d;
  cudaError_t cudaError;
  /*
  printf("deviceInfo(%d,%d,%d)\n", device->lb,device->ub,device->step);
  for(d = device->lb; d <= device->ub; d += device->step){
    cudaError = cudaSetDevice(d-1);
    if(cudaError != cudaSuccess){
      _XMP_fatal("failed to set device");
      return;
    }

    int d2;
    
    d2 = d - device->step;
    if(d2 >= device->lb){
      printf("eneblePeerAccess(%d) on %d\n", d2-1, d-1);
      cudaError = cudaDeviceEnablePeerAccess(d2-1, 0);
      if(cudaError != cudaSuccess){
        _XMP_fatal("failed to enable peer access");
        return;
      }
    }

    d2 = d + device->step;
    if(d2 <= device->ub){
      printf("eneblePeerAccess(%d) on %d\n", d2-1, d-1);
      cudaError = cudaDeviceEnablePeerAccess(d2-1, 0);
      if(cudaError != cudaSuccess){
        _XMP_fatal("failed to enable peer access");
        return;
      }
    }
  }
*/

  int dim = arrays_desc->dim;
  //他ノードとの通信のセットアップ
  //  int *lwidth = _XMP_alloc(sizeof(int)*dim);
  //  int *uwidth = _XMP_alloc(sizeof(int)*dim);
  for(int i = 0; i < arrays_desc->device_type->size; i++){
    _XACC_array_t *array_desc = arrays_desc->device_array + i;
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

  _XACC_array_t *array_desc = arrays_desc->device_array + target_device;
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

    printf("count =%d, blength=%d, stride=%lld\n", count, blocklength, stride);
    printf("ainfo[0].par_size=%d\n", ainfo[0].par_size);
    printf("count_ofset=%d,\n", count_offset);
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
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      lo_send_array = (void *)((char *)lo_send_array + lb_send * dim_acc * type_size);
      lo_recv_array = (void *)((char *)lo_recv_array + lb_recv * dim_acc * type_size);
    }

    lo_send_buf = lo_send_array;
    lo_recv_buf = lo_recv_array;
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
	lb_send = 0;
	lb_recv = 0;
      }

      dim_acc = ainfo[i].dim_acc;

      hi_send_array = (void *)((char *)hi_send_array + lb_send * dim_acc * type_size);
      hi_recv_array = (void *)((char *)hi_recv_array + lb_recv * dim_acc * type_size);
    }

    hi_send_buf = hi_send_array;
    hi_recv_buf = hi_recv_array;
  }

  // for lower reflect

  if (reflect->datatype_lo != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_lo);
  }

  printf("lower type_vector(%d, %d, %lld) @ rank=%d,dev=%d\n",count, blocklength * lwidth,stride,my_rank, target_device);
  MPI_Type_vector(count, blocklength * lwidth, stride,
		  MPI_BYTE, &reflect->datatype_lo);
  MPI_Type_commit(&reflect->datatype_lo);

  // for upper reflect

  if (reflect->datatype_hi != MPI_DATATYPE_NULL){
    MPI_Type_free(&reflect->datatype_hi);
  }

  printf("upper type_vector(%d, %d, %lld) @ rank=%d,dev=%d\n",count, blocklength * uwidth,stride,my_rank,target_device);
  MPI_Type_vector(count, blocklength * uwidth, stride,
		  MPI_BYTE, &reflect->datatype_hi);
  MPI_Type_commit(&reflect->datatype_hi);

  
  /*
  //alloc buffer
  if ((_XMPF_running && target_dim != ndims - 1) ||
      (_XMPC_running && target_dim != 0)){
    _XACC_gpu_host_free(reflect->lo_send_buf);
    _XACC_gpu_host_free(reflect->lo_recv_buf);
    _XACC_gpu_host_free(reflect->hi_send_buf);
    _XACC_gpu_host_free(reflect->hi_recv_buf);
  }
  if ((_XMPF_running && target_dim == ndims - 1) ||
      (_XMPC_running && target_dim == 0)){
    _XACC_gpu_host_free(reflect->lo_send_buf);
    _XACC_gpu_host_free(reflect->lo_recv_buf);
    _XACC_gpu_host_free(reflect->hi_send_buf);
    _XACC_gpu_host_free(reflect->hi_recv_buf);
  }

  if (lwidth){

    lo_buf_size = lwidth * blocklength * count;

    if ((_XMPF_running && target_dim == ndims - 1) ||
	(_XMPC_running && target_dim == 0)){
      lo_send_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_recv_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_send_dev_buf = lo_send_dev_array;
      lo_recv_dev_buf = lo_recv_dev_array;
    } else {
      _XMP_TSTART(t0);
      lo_send_buf = _XMP_gpu_host_alloc(lo_buf_size);
      lo_recv_buf = _XMP_gpu_host_alloc(lo_buf_size);

      _XMP_gpu_alloc((void **)&lo_send_dev_buf, lo_buf_size); //lo_send_dev_buf = _XMP_gpu_alloc(lo_buf_size);
      _XMP_gpu_alloc((void **)&lo_recv_dev_buf, lo_buf_size); //lo_recv_dev_buf = _XMP_gpu_alloc(lo_buf_size);
      _XMP_TEND2(xmptiming_.t_mem, xmptiming_.tdim_mem[target_dim], t0);
    }

  }
  */


  //
  // initialize communication
  //

  int src, dst;
  int is_periodic = reflect->is_periodic;
  int num_devices = arrays_desc->device_type->size;
  if (!is_periodic && my_pos == lb_pos && (target_dim != 0 || target_device == 0)){ // no periodic
    lo_rank = MPI_PROC_NULL;
  }else if(target_device != 0){
    lo_rank = my_rank;
  }

  if (!is_periodic && my_pos == ub_pos && (target_dim != 0 || target_device == num_devices - 1)){ // no periodic
    hi_rank = MPI_PROC_NULL;
  }else if(target_device != num_devices -1){
    hi_rank = my_rank;
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

  printf("lo_recv pos=%lld, from(%d) @rank=%d,dev=%d\n", (long long )(lo_recv_buf - array_addr), src, my_rank, target_device);
  printf("lo_send pos=%lld, to(%d) @rank=%d,dev=%d\n", (long long )(lo_send_buf - array_addr), dst, my_rank, target_device);
  MPI_Recv_init(lo_recv_buf, 1, reflect->datatype_lo, src,
		_XMP_N_MPI_TAG_REFLECT_LO, *comm, &reflect->req[0]);
  MPI_Send_init(lo_send_buf, 1, reflect->datatype_lo, dst,
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

  printf("hi_recv pos=%lld, from(%d) @rank=%d,dev=%d\n", (long long )(hi_recv_buf - array_addr), src, my_rank, target_device);
  printf("hi_send pos=%lld, to(%d) @rank=%d,dev=%d\n", (long long )(hi_send_buf - array_addr), dst, my_rank, target_device);
  MPI_Recv_init(hi_recv_buf, 1, reflect->datatype_hi, src,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[2]);
  MPI_Send_init(hi_send_buf, 1, reflect->datatype_hi, dst,
		_XMP_N_MPI_TAG_REFLECT_HI, *comm, &reflect->req[3]);

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

  reflect->lo_rank = lo_rank;
  reflect->hi_rank = hi_rank;
}


void _XACC_reflect_do_inter_start(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;
  for(int i = 0; i < arrays_desc->device_type->size; i++){
    _XACC_array_t *array_desc = arrays_desc->device_array + i;
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
        _XMP_reflect_sched_t *reflect = ai->reflect_sched;
        MPI_Startall(4, reflect->req);
      }
    }
  }
}

void _XACC_reflect_do_inter_wait(_XACC_arrays_t *arrays_desc)
{
  int dim = arrays_desc->dim;
  for(int i = 0; i < arrays_desc->device_type->size; i++){
    _XACC_array_t *array_desc = arrays_desc->device_array + i;
    for(int j = 0; j < dim; j++){
      _XACC_array_info_t *ai = array_desc->info + j;
      if(ai->shadow_size_lo != 0 || ai->shadow_size_hi != 0){
        _XMP_reflect_sched_t *reflect = ai->reflect_sched;
        MPI_Waitall(4, reflect->req, MPI_STATUSES_IGNORE);
      }
    }
  }
}


void _XACC_reflect_do(_XACC_arrays_t *arrays_desc){
  int numDevices = arrays_desc->device_type->size;
  int dev;

  //他ノードとの通信の開始
  _XACC_reflect_do_inter_start(arrays_desc);

  /*
  for(dev=0; dev < numDevices; dev++){
    _XACC_array_t* device_array = &(arrays_desc->device_array[dev]);
    _XACC_array_info_t* info0 = &device_array->info[0];

    if(info0->device_layout_manner != _XMP_N_DIST_BLOCK){
      return;
    }

    size_t type_size = arrays_desc->type_size;
    if(dev > 0){
      _XACC_array_t* lower_device_array = &(arrays_desc->device_array[dev-1]);
      //      _XACC_array_info_t* lower_info0 = &lower_device_array->info[0];

      unsigned long long loSendOffset;
      unsigned long long loSendElements;

      loSendOffset = info0->dim_acc * info0->local_lower;
      loSendElements = info0->dim_acc * info0->shadow_size_lo;
      //printf("loSendOffset=%lld, size=%lld, recvOffset=%lld\n", loSendOffset, loSendSize, loRecvOffset);

      size_t loSendSize = type_size * loSendElements;
      char* sendPtr= (char*)device_array->deviceptr + loSendOffset * type_size;
      char* recvPtr= (char*)lower_device_array->deviceptr + loSendOffset * type_size;
      //printf("sendP=%p, recvP=%p, size=%zd\n", sendPtr,recvPtr,loSendSize);
      cudaMemcpy(recvPtr, sendPtr, loSendSize, cudaMemcpyDefault);
    }

    if(dev < numDevices - 1){
      _XACC_array_t* upper_device_array = &(arrays_desc->device_array[dev+1]);
      //      _XACC_array_info_t* upper_info0 = &upper_device_array->info[0];

      unsigned long long hiSendOffset;
      unsigned long long hiSendElements;

      hiSendOffset = info0->dim_acc * info0->local_upper;
      hiSendElements = info0->dim_acc * info0->shadow_size_hi;
      printf("hiSendOffset=%lld, elements=%lld\n", hiSendOffset, hiSendElements);
      
      size_t hiSendSize = hiSendElements * type_size;
      char* sendPtr= (char*)device_array->deviceptr + hiSendOffset * type_size;
      char* recvPtr= (char*)upper_device_array->deviceptr + hiSendOffset * type_size;
      cudaMemcpy(recvPtr, sendPtr, hiSendSize, cudaMemcpyDefault);
    }

    //cudaMemcpy(, cudaMemcpyDefault);
  }
  */

  //他ノードとの通信の待機
  _XACC_reflect_do_inter_wait(arrays_desc);

}

