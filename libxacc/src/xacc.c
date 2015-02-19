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
  xacc_device->lb = lower - 1;
  xacc_device->ub = upper - 1;
  xacc_device->step = step;
  xacc_device->size = (upper - lower) / step + 1;

  //  fprintf(stderr,"device(%d,%d,%d, size=%d)\n", xacc_device->lb, xacc_device->ub, xacc_device->step, xacc_device->size);
  *desc = xacc_device;
}

void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step)
{
  _XACC_device_t* xacc_device = (_XACC_device_t*)desc;
  *lower = xacc_device->lb + 1;
  *upper = xacc_device->ub + 1;
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
  layoutedArray->array = (_XACC_array_t*)_XMP_alloc(sizeof(_XACC_array_t) * (device->ub + 1));
  for(int dev = device->lb; dev <= device->ub; dev += device->step){
    _XACC_array_info_t* d_array_info = (_XACC_array_info_t*)_XMP_alloc(sizeof(_XACC_array_info_t) * dim);
    _XMP_array_info_t *h_array_info = alignedArray->info;
    layoutedArray->array[dev].info = d_array_info;
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

      /* fprintf(stderr,"dim=%d, host par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
      /*      i, */
      /*      h_array_info[i].par_lower, */
      /*      h_array_info[i].par_upper, */
      /*      h_array_info[i].par_stride, */
      /*      h_array_info[i].local_lower, */
      /*      h_array_info[i].local_upper, */
      /*      h_array_info[i].local_stride, */
      /*      h_array_info[i].alloc_size); */

    }
  }
  layoutedArray->dim = dim;
  layoutedArray->hostptr = alignedArray->array_addr_p;
  layoutedArray->type_size = alignedArray->type_size;
  layoutedArray->xmp_array = alignedArray;
  *arrays = layoutedArray;
}

void _XACC_init_layouted_array_normal(_XACC_arrays_t **arrays, void* host_ptr, size_t element_size, int dim, unsigned long long * dim_size, _XACC_device_t* device)
{
  _XACC_arrays_t* layoutedArray = (_XACC_arrays_t*)_XMP_alloc(sizeof(_XACC_arrays_t));

  layoutedArray->device_type = device;
  layoutedArray->array = (_XACC_array_t*)_XMP_alloc(sizeof(_XACC_array_t) * (device->ub + 1));
  
  for(int dev = device->lb; dev <= device->ub; dev += device->step){
    _XACC_array_info_t* d_array_info = (_XACC_array_info_t*)_XMP_alloc(sizeof(_XACC_array_info_t) * dim);
    //    _XMP_array_info_t *h_array_info = (_XMP_array_info_t*)_XMP_alloc(sizeof(_XMP_array_info_t) * dim);//alignedArray->info;
    layoutedArray->array[dev].info = d_array_info;
    int i;
    for(i = 0; i <dim; i++){
      unsigned long long dim_size_i = dim_size[i];

      d_array_info[i].device_layout_manner = _XMP_N_DIST_DUPLICATION;
      d_array_info[i].par_lower = 0; //h_array_info[i].par_lower;
      d_array_info[i].par_upper = dim_size_i; //h_array_info[i].par_upper;
      d_array_info[i].par_stride = 1; //h_array_info[i].par_stride;
      d_array_info[i].par_size = dim_size_i; //h_array_info[i].par_size;

      d_array_info[i].alloc_size = dim_size_i; //h_array_info[i].alloc_size;

      d_array_info[i].local_lower = 0; //h_array_info[i].local_lower;
      d_array_info[i].local_upper = dim_size_i; //h_array_info[i].local_upper;
      d_array_info[i].local_stride = 1; //h_array_info[i].local_stride;

      d_array_info[i].shadow_size_lo = 0; //h_array_info[i].shadow_size_lo;
      d_array_info[i].shadow_size_hi = 0; //h_array_info[i].shadow_size_hi;

      d_array_info[i].reflect_sched = NULL;

      /* fprintf(stderr,"dim=%d, host par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
      /*      i, */
      /*      h_array_info[i].par_lower, */
      /*      h_array_info[i].par_upper, */
      /*      h_array_info[i].par_stride, */
      /*      h_array_info[i].local_lower, */
      /*      h_array_info[i].local_upper, */
      /*      h_array_info[i].local_stride, */
      /*      h_array_info[i].alloc_size); */

    }
  }
  layoutedArray->dim = dim;
  layoutedArray->hostptr = host_ptr; //alignedArray->array_addr_p;
  layoutedArray->type_size = element_size; //alignedArray->type_size;
  layoutedArray->xmp_array = NULL; //alignedArray;
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
  //int num_devices = array_desc->device_type->size;
  _XACC_device_t* device = array_desc->device_type;
  for(int dev = device->lb; dev <= device->ub; dev += device->step){
    _XACC_array_info_t *d_array_info = &(array_desc->array[dev].info[dim]);
    d_array_info->device_layout_manner = _XMP_N_DIST_BLOCK;

    unsigned long long size = _XMP_M_CEILi(d_array_info->par_size, device->size);
    //d_array_info->par_stride;// = h_array_info->par_stride;    
    d_array_info->par_lower += size * (dev - device->lb);
    
    if(dev != device->ub){
      d_array_info->par_upper = d_array_info->par_lower + size - 1;
    }else{
      //      d_array_info->par_upper = h_array_info->par_upper;
    }
    d_array_info->par_size = d_array_info->par_upper - d_array_info->par_lower + 1;
    //d_array_info->local_stride = h_array_info->local_stride;
    d_array_info->local_lower += size * (dev - device->lb);
    if(dev != device->ub){
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

    
    /* fprintf(stderr, "dim=%d, block par (%d, %d, %d, %d) local(%d, %d, %d) alloc(%d)\n", */
    /*        dim, */
    /*        d_array_info->par_lower, */
    /*        d_array_info->par_upper, */
    /*        d_array_info->par_stride, */
    /*        d_array_info->par_size,	     */
    /*        d_array_info->local_lower, */
    /*        d_array_info->local_upper, */
    /*        d_array_info->local_stride, */
    /*        d_array_info->alloc_size); */

    /* Printf("block par (%d, %d, %d, %d)\n",  */
    /*        d_array_info->lower, */
    /*        d_array_info->upper, */
    /*        d_array_info->stride, */
    /*        d_array_info->size); */
  }
}

void _XACC_set_shadow_NORMAL(_XACC_arrays_t* array_desc, int dim , int lo, int hi)
{
  //int num_devices = array_desc->device_type->size;
  _XACC_device_t* device = array_desc->device_type;
  for(int dev = device->lb; dev <= device->ub; dev += device->step){
    _XACC_array_info_t *d_array_info = &(array_desc->array[dev].info[dim]);

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
        if(dev != device->lb){
          d_array_info->shadow_size_lo = lo;
          d_array_info->local_lower += lo;
          d_array_info->local_upper += lo;
        }
        if(dev != device->ub){
          d_array_info->shadow_size_hi = hi;
          d_array_info->local_upper += hi;
        }
      }

    /* d_array_info->shadow_size_lo = lo; */
    /* d_array_info->shadow_size_hi = hi; */

      //    printf("shadow dim=%d, lo=%d,hi=%d\n", dim, lo, hi);
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
  //int num_devices = array_desc->device_type->size;
  _XACC_device_t * device = array_desc->device_type;
  for(int dev = device->lb; dev <= device->ub; dev += device->step){
    unsigned long long device_acc = 1;
    unsigned long long device_offset = 0;
    int dim = array_desc->dim;
    _XACC_array_t *d_array_desc = &(array_desc->array[dev]);
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
      d_array_desc->copy_size = device_acc * (info->par_upper - info->par_lower + 1); //device_acc * (info->alloc_size - info->shadow_size_lo - info->shadow_size_hi);
      d_array_desc->copy_offset = device_offset + device_acc * info->local_lower;
    }

    /* printf("alloc(%llu, %llu), copy(%llu, %llu)@dev\n", */
    /*        d_array_desc->alloc_offset, */
    /*        d_array_desc->alloc_size, */
    /*        d_array_desc->copy_offset, */
    /*        d_array_desc->copy_size); */
  }
}

static _XACC_array_t* get_array(_XACC_arrays_t* array_desc, int deviceNum)
{
  /* deviceNum is 1-based */
  //_XACC_device_t* device = array_desc->device_type;
  //int n = ((deviceNum - 1) - device->lb) / device->step;
  int n = deviceNum - 1;
  return &(array_desc->array[n]);
}

void _XACC_get_size(_XACC_arrays_t* array_desc, unsigned long long* offset,
                    unsigned long long* size, int deviceNum)
{
  _XACC_array_t* device_array = get_array(array_desc, deviceNum);
  *size = device_array->alloc_size;
  *offset = device_array->alloc_offset;
}

void _XACC_get_copy_size(_XACC_arrays_t* array_desc, unsigned long long* offset,
                         unsigned long long* size, int deviceNum)
{
  _XACC_array_t* device_array = get_array(array_desc, deviceNum);
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
  _XACC_array_t* device_array = get_array(array_desc, deviceNum);
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
  _XACC_array_t* arrayOnDevice = get_array(arrays_desc, deviceNum);
  arrayOnDevice->deviceptr = deviceptr;
  //  printf("deviceptr=%p@%d\n", deviceptr, deviceNum);
}

