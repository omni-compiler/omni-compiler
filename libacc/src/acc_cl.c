#include "acc_internal.h"
#include "acc_internal_cl.h"

//global variables
cl_uint _ACC_cl_num_devices; //=1
cl_device_id _ACC_cl_device_ids[_ACC_CL_MAX_NUM_DEVICES];
cl_context _ACC_cl_current_context;
cl_uint _ACC_cl_device_num; //now fixed

static bool is_working = false;

#define PRINT_PLATFORM_INFO(PLATFORM_ID, PARAM)				\
  do{									\
    size_t param_val_size;						\
    CL_CHECK(clGetPlatformInfo((PLATFORM_ID), (PARAM), 0, NULL, &param_val_size)); \
    char param_val[param_val_size];					\
    CL_CHECK(clGetPlatformInfo((PLATFORM_ID), (PARAM), param_val_size, param_val, NULL)); \
    _ACC_DEBUG("%s: %s\n", #PARAM, param_val);				\
  }while(0);

#define PRINT_DEVICE_INFO(DEVICE_ID, PARAM)				\
  do{									\
    size_t param_val_size;						\
    CL_CHECK(clGetDeviceInfo((DEVICE_ID), (PARAM), 0, NULL, &param_val_size)); \
    char param_val[param_val_size];					\
    CL_CHECK(clGetDeviceInfo((DEVICE_ID), (PARAM), param_val_size, param_val, NULL)); \
    _ACC_DEBUG("%s: %s\n", #PARAM, param_val);				\
  }while(0);

void _ACC_platform_init()
{
  if(is_working == true){
    _ACC_fatal("_ACC_platform_init was called more than once");
  }

  cl_platform_id platform_id = NULL;
  cl_uint ret_num_platforms;
  cl_int ret;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, &ret_num_platforms));
  if(ret_num_platforms == 0){
    _ACC_fatal("no available cl_platform");
  }

  cl_uint ret_num_devices;
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, _ACC_CL_MAX_NUM_DEVICES, _ACC_cl_device_ids, &ret_num_devices));
  if(ret_num_devices == 0){
    _ACC_fatal("no available cl_device");
  }
  
  _ACC_DEBUG("req =%d, available devices=%d\n", _ACC_CL_MAX_NUM_DEVICES, ret_num_devices);

  if(ret_num_devices <= _ACC_CL_MAX_NUM_DEVICES){
    _ACC_cl_num_devices = ret_num_devices;
  }else{
    fprintf(stderr, "available devices are limited by _ACC_CL_MAX_NUM_DEVICES=%d\n", _ACC_CL_MAX_NUM_DEVICES);
    _ACC_cl_num_devices = _ACC_CL_MAX_NUM_DEVICES;
  }
  
  _ACC_cl_current_context = clCreateContext(NULL, _ACC_cl_num_devices, _ACC_cl_device_ids, NULL, NULL, &ret);
  CL_CHECK(ret);

  _ACC_DEBUG("Platform info\n");
  PRINT_PLATFORM_INFO(platform_id, CL_PLATFORM_PROFILE);
  PRINT_PLATFORM_INFO(platform_id, CL_PLATFORM_VERSION);
  PRINT_PLATFORM_INFO(platform_id, CL_PLATFORM_NAME);
  PRINT_PLATFORM_INFO(platform_id, CL_PLATFORM_VENDOR);
  PRINT_PLATFORM_INFO(platform_id, CL_PLATFORM_EXTENSIONS);

  for(int i = 0; i < _ACC_cl_num_devices; i++){
    _ACC_DEBUG("Device %d info\n", i);
    PRINT_DEVICE_INFO(_ACC_cl_device_ids[i], CL_DEVICE_VENDOR);
    PRINT_DEVICE_INFO(_ACC_cl_device_ids[i], CL_DEVICE_NAME);
    PRINT_DEVICE_INFO(_ACC_cl_device_ids[i], CL_DEVICE_VERSION);
    PRINT_DEVICE_INFO(_ACC_cl_device_ids[i], CL_DRIVER_VERSION);
    _ACC_DEBUG("\n");
  }

  is_working = true;
}
void _ACC_platform_finalize()
{
  if(is_working == false){
    _ACC_fatal("_ACC_platform_finalize was called before _ACC_platform_init");    
  }
  CL_CHECK(clReleaseContext(_ACC_cl_current_context));
  is_working = false;
}

int _ACC_platform_get_num_devices()
{
  return (int)_ACC_cl_num_devices;
}

bool _ACC_platform_allocate_device(int device_num)
{
  if(device_num < _ACC_cl_num_devices){
    return true;
  }else{
    return false;
  }
}

void _ACC_platform_set_device_num(int device_num /*0-based*/)
{
  if(device_num < 0){
    _ACC_fatal("device_num < 0 is not allowed");
  }else{
    _ACC_cl_device_num = device_num;
  }
}

void _ACC_platform_init_device(int device_num /*0-based*/)
{
  if(device_num < 0){
    _ACC_fatal("device_num < 0 is not allowed");
  }else{
    _ACC_cl_device_num = device_num;
  }

  char buf[1024];
  clGetDeviceInfo(_ACC_cl_device_ids[_ACC_cl_device_num], CL_DEVICE_NAME, sizeof(buf), &buf, NULL);
  _ACC_DEBUG("CL_DEVICE_NAME=%s\n", buf);
}

void _ACC_copy(void *host_addr, void *device_addr, size_t size, int direction){
  _ACC_cl_copy(host_addr, (cl_mem)device_addr, 0, size, direction, ACC_ASYNC_SYNC);
}

void _ACC_copy_async(void *host_addr, void *device_addr, size_t size, int direction, int async){
  _ACC_cl_copy(host_addr, (cl_mem)device_addr, 0, size, direction, async);
}

void _ACC_gpu_alloc(void** addr, size_t size)
{
  cl_int ret;
  cl_mem mem = clCreateBuffer(_ACC_cl_current_context, CL_MEM_READ_WRITE, size, NULL, &ret);
  CL_CHECK(ret);
  *addr = mem;
}

void _ACC_gpu_free(void *addr)
{
  CL_CHECK(clReleaseMemObject((cl_mem)addr));
}



/// OpenACC API for OpenCL
void* acc_get_current_opencl_device()
{
  return _ACC_cl_device_ids[_ACC_cl_device_num];
}
void* acc_get_current_opencl_context()
{
  return _ACC_cl_current_context;
}
