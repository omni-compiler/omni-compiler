#ifndef _ACC_INTERNAL_CL_HEADER
#define _ACC_INTERNAL_CL_HEADER

#include <stdio.h>
#include <stdbool.h>

#ifdef PEZY
#include "pzcl/pzcl_ocl_wrapper.h"
#include "PZSDKHelper.h"
#else
#include <CL/cl.h>
#endif

struct _ACC_memory_type{
  void *host_addr;
  cl_mem memory_object;
  size_t size;
  bool is_alloced;
  
  bool is_pagelocked;
  bool is_registered;
  unsigned int ref_count;
};

#define CL_CHECK(ret)					\
  do{							\
    cl_int _internal_ret = ret;				\
    if(_internal_ret != CL_SUCCESS){			\
      printf("%s(%d) OpenCL error : %s (code %d)\n", __FILE__, __LINE__, _ACC_cl_get_error_string(_internal_ret), (int)_internal_ret); \
      exit(1);						\
    }							\
  }while(0)

/* #ifdef _DEBUG */
/* #define _ACC_DEBUG(...) do{printf("%s(%d)[%s]: ", __FILE__, __LINE__, __func__); printf(__VA_ARGS__);}while(0) */
/* #else */
/* #define _ACC_DEBUG(...) do{}while(0) */
/* #endif */

#define _ACC_FATAL(...) do{printf("%s(%d)[%s] fatal: ", __FILE__, __LINE__, __func__); printf(__VA_ARGS__); exit(1); }while(0)

#define _ACC_COPY_HOST_TO_DEVICE 400
#define _ACC_COPY_DEVICE_TO_HOST 401

#ifdef PEZY
#define _ACC_CL_MAX_NUM_DEVICES 4
#else
#define _ACC_CL_MAX_NUM_DEVICES 4
#endif

//global variables
extern cl_context _ACC_cl_current_context;
extern cl_uint _ACC_cl_num_devices;
extern cl_device_id _ACC_cl_device_ids[_ACC_CL_MAX_NUM_DEVICES];
extern cl_uint _ACC_cl_device_num;

typedef struct _ACC_program_type _ACC_program_t;
typedef struct _ACC_kernel_type _ACC_kernel_t;

void _ACC_queue_set_last_event(_ACC_queue_t* queue, cl_event event);
cl_command_queue _ACC_queue_get_command_queue(_ACC_queue_t *queue);

void _ACC_cl_copy(void *host_addr, cl_mem memory_object, size_t mem_offset, size_t size, int direction, int asyncId);
const char* _ACC_cl_get_error_string(cl_int);

#endif  //end _ACC_INTERNAL_CL_HEADER
