//#include<openacc.h>
typedef  int acc_device_t;



//_XACC_device_t *current_device;

typedef struct _XACC_device_type {
  acc_device_t acc_device;
  int lb;
  int ub;
  int step;
  int size;
} _XACC_device_t;

void _XACC_init_device(void* desc, acc_device_t device, int lower, int upper, int step);
int _XACC_get_num_current_devices();
acc_device_t _XACC_get_current_device();
void _XACC_get_current_device_info(int* lower, int* upper, int* step);
void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step);

void _XMP_init_device(void* desc, acc_device_t device, int lower, int upper, int step);
void _XMP_get_device_info(void *desc, int* lower, int* upper, int* step);
