#include "xmp_internal.h"
#include "xmp_data_struct.h"
#include "xmp_math_function.h"

/* void _XACC_init_device_array(_XMP_array_t* array, _XACC_device_t* device); */
/* void _XACC_split_device_array_BLOCK(_XMP_array_t* array, int dim); */
/* void _XACC_calc_size(_XMP_array_t* array); */

/* void _XACC_get_size(_XMP_array_t* array, unsigned long long* offset, */
/*                unsigned long long* size, int deviceNum); */
/* void _XACC_sched_loop_layout_BLOCK(int init, */
/*                                    int cond, */
/*                                    int step, */
/*                                    int* sched_init, */
/*                                    int* sched_cond, */
/*                                    int* sched_step, */
/*                                    _XMP_array_t* array_desc, */
/*                                    int dim, */
/*                                    int deviceNum); */


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




void _XACC_init_device_array(_XMP_array_t* array, _XACC_device_t* device){
  array->device_type = device;
  int dim = array->dim;
  int num_devices = device->size;
  array->device_array = (_XACC_array_t*)_XMP_alloc(sizeof(_XACC_array_t) * num_devices);
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t* d_info = (_XACC_array_info_t*)_XMP_alloc(sizeof(_XACC_array_info_t) * dim);
    array->device_array[dev].info = d_info;
  }
}

void _XACC_split_device_array_DUPLICATION(_XMP_array_t* array_desc, int dim){
  _XMP_array_info_t *h_array_info = &(array_desc->info[dim]);

  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t *d_array_info = &(array_desc->device_array[dev].info[dim]);
    d_array_info->device_layout_manner = _XMP_N_DIST_DUPLICATION;

    d_array_info->par_lower = h_array_info->par_lower;
    d_array_info->par_upper = h_array_info->par_upper;
    d_array_info->par_stride = h_array_info->par_stride;

    d_array_info->alloc_size = h_array_info->alloc_size;

    d_array_info->local_lower = h_array_info->local_lower;
    d_array_info->local_upper = h_array_info->local_upper;
    d_array_info->local_stride = h_array_info->local_stride;

    /* printf("dim=%d, dup par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
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

void _XACC_split_device_array_BLOCK(_XMP_array_t* array_desc, int dim){
  _XMP_array_info_t *h_array_info = &(array_desc->info[dim]);

  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    _XACC_array_info_t *d_array_info = &(array_desc->device_array[dev].info[dim]);
    d_array_info->device_layout_manner = _XMP_N_DIST_BLOCK;


    unsigned long long size = _XMP_M_CEILi(h_array_info->par_size, num_devices);
    d_array_info->par_stride =h_array_info->par_stride;    
    d_array_info->par_lower = h_array_info->par_lower + size * dev;
    
    if(dev != num_devices - 1){
      d_array_info->par_upper = h_array_info->par_lower + size * (dev+1);
    }else{
      d_array_info->par_upper = h_array_info->par_upper;
    }
    d_array_info->local_stride = h_array_info->local_stride;
    d_array_info->local_lower = h_array_info->local_lower + size * dev;
    if(dev != num_devices - 1){
      d_array_info->local_upper = h_array_info->local_lower + size * (dev + 1);
    }else{
      d_array_info->local_upper = h_array_info->local_upper;
    }

    d_array_info->alloc_size = size;

/* #if not */
/*     int size = h_array_info->par_size / num_devices; */
/*     d_array_info->lower = h_array_info->par_lower + size * dev; */
/*     d_array_info->upper = h_array_info->par_lower + size * (dev+1); */
/*     d_array_info->stride =h_array_info->par_stride; */
/*     d_array_info->alloc_size = size; */
/* #endif */

    
    /* printf("dim=%d, block par (%d, %d, %d) local(%d, %d, %d, %d)\n", */
    /*        dim, */
    /*        d_array_info->par_lower, */
    /*        d_array_info->par_upper, */
    /*        d_array_info->par_stride, */
    /*        d_array_info->local_lower, */
    /*        d_array_info->local_upper, */
    /*        d_array_info->local_stride, */
    /*        d_array_info->alloc_size); */

    /* printf("block par (%d, %d, %d, %d)\n",  */
    /*        d_array_info->lower, */
    /*        d_array_info->upper, */
    /*        d_array_info->stride, */
    /*        d_array_info->size); */
  }
}
void _XACC_calc_size(_XMP_array_t* array_desc){
  int num_devices = array_desc->device_type->size;
  for(int dev = 0; dev < num_devices; dev++){
    unsigned long long device_acc = 1;
    unsigned long long device_offset = 0;
    int dim = array_desc->dim;
    _XACC_array_t *d_array_desc = &(array_desc->device_array[dev]);
    for(int i = dim - 1; i >= 0; i--){
      _XACC_array_info_t* info = &(d_array_desc->info[i]);
      info->device_dim_acc = device_acc;
      device_offset += device_acc * info->local_lower;
      device_acc *= info->alloc_size;
    }

    d_array_desc->alloc_size = device_acc; //num elements
    d_array_desc->alloc_offset = device_offset; //num elements
    /* printf("alloc (%llu, %llu)@dev\n",  device_offset, device_acc); */
  }
}

static _XACC_array_t* get_device_array(_XMP_array_t* array_desc, int deviceNum)
{
  _XACC_device_t* device = array_desc->device_type;
  int lower = device->lb;
  int step = device->step;
  
  int n = (deviceNum - lower) / step;
  return &(array_desc->device_array[n]);
}

void _XACC_get_size(_XMP_array_t* array_desc, unsigned long long* offset,
                    unsigned long long* size, int deviceNum)
{
  _XACC_array_t* device_array = get_device_array(array_desc, deviceNum);
  *size = device_array->alloc_size;
  *offset = device_array->alloc_offset;
}

void _XACC_sched_loop_layout_BLOCK(int init,
                              int cond,
                              int step,
                              int* sched_init,
                              int* sched_cond,
                              int* sched_step,
                              _XMP_array_t* array_desc,
                              int dim,
                              int deviceNum)
{
  _XACC_array_t* device_array = get_device_array(array_desc, deviceNum);
  _XACC_array_info_t* info = &device_array->info[dim];
  *sched_init = info->local_lower;
  *sched_cond = info->local_upper + 1;
  *sched_step = info->local_stride;
  /* printf("orginal loop (%d, %d, %d)\n", init, cond,step); */
  /* printf("loop(%d, %d, %d)@%d\n", *sched_init, *sched_cond, *sched_step, deviceNum) */;
}
