// xacc.c
extern void _XACC_init_device(void* desc, acc_device_t device, int lower, int upper, int step);
extern int _XACC_get_num_current_devices();
extern acc_device_t _XACC_get_current_device();
extern void _XACC_get_current_device_info(int* lower, int* upper, int* step);
extern void _XACC_get_device_info(void *desc, int* lower, int* upper, int* step);


extern void _XACC_init_layouted_array(void **layoutedArray, void* alignedArray, void* device);
extern void _XACC_split_layouted_array_BLOCK(void* array, int dim);
extern void _XACC_split_layouted_array_DUPLICATION(void* array, int dim);
extern void _XACC_calc_size(void* array);

extern void _XACC_get_size(void* array, unsigned long long* offset,
               unsigned long long* size, int deviceNum);
extern void _XACC_sched_loop_layout_BLOCK(int init,
                                   int cond,
                                   int step,
                                   int* sched_init,
                                   int* sched_cond,
                                   int* sched_step,
                                   void* array_desc,
                                   int dim,
                                   int deviceNum);
extern void _XACC_set_shadow_NORMAL(void* array_desc, int dim , int lo, int hi);
