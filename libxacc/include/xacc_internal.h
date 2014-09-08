_XACC_device_t *current_device;

typedef struct _XACC_device_type {
  acc_device_t acc_device;
  int lb;
  int ub;
  int size;
} _XACC_device_t;
