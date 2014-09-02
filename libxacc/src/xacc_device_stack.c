#include <stdlib.h>
#include "xacc_internal.h"

/* typedef struct _XACC_device_dish_type { */
/*   _XACC_device_t device; */
/*   int num; */
/*   struct _XACC_device_dish_type *prev; */
/* } _XACC_device_dish_t; */

static _XACC_device_t *_XACC_current_device = NULL;

/* _XACC_device_t _XACC_current_device = NULL; */
/* int _XACC_current_device_num = 0; */

/* void _XACC_push_device(_XACC_device_t device, int num) { */
/*   _XACC_device_dish_t *new_dish = _XACC_alloc(sizeof(_XACC_device_dish_t)); */
/*   new_dish->device = device; */
/*   new_dish->num = num; */
/*   new_dish->prev = _XACC_device_stack_top; */
/*   _XACC_device_stack_top = new_dish; */

/*   _XACC_current_device = device; */
/*   _XACC_current_device_num = num; */
/* } */

/* void _XACC_pop_device(void) { */
/*   _XACC_device_dish_t *freed_dish = _XACC_device_stack_top; */
/*   _XACC_device_stack_top = freed_dish->prev; */
/*   _XACC_free(freed_dish); */

/*   _XACC_current_device = _XACC_device_stack_top->device; */
/*   _XACC_current_device_num = _XACC_device_stack_top->num; */
/* } */

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


void acc_set_device_type(acc_device_t device);
void acc_set_device_num(int num);


//internal functions
int _XACC_get_num_current_devices(){
  return _XACC_current_device->size;
}

acc_device_t _XACC_get_current_device(){
  return _XACC_current_device->acc_device;
}

void _XACC_init_device(void* desc, acc_device_t device, int lower, int upper, int step)
{
  _XACC_device_t* xacc_device = (_XACC_device_t*)malloc(sizeof(_XACC_device_t));//(_XACC_device_t*)_XMP_alloc(sizeof(_XACC_device_t));
  xacc_device->acc_device = device;
  xacc_device->lb = lower;
  xacc_device->ub = upper;
  xacc_device->step = step;
  xacc_device->size = upper - lower + 1; //must consider step

  desc = xacc_device;
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


void _XMP_init_device(void* desc, acc_device_t device, int lower, int upper, int step)
{
  _XACC_init_device(desc, device, lower, upper, step);
}
void _XMP_get_device_info(void *desc, int* lower, int* upper, int* step)
{
  _XACC_get_device_info(desc, lower, upper, step);
}
