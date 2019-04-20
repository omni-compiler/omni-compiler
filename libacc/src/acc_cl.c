#include "acc_internal.h"
#include "acc_internal_cl.h"

#define INITIAL_DEVICE_NUM 1000

//global variables
cl_uint _ACC_cl_num_devices; //=1
cl_device_id _ACC_cl_device_ids[_ACC_CL_MAX_NUM_DEVICES];
cl_context _ACC_cl_current_context;
cl_uint _ACC_cl_device_num = INITIAL_DEVICE_NUM; //now fixed

static bool is_working = false;

static cl_platform_id _ACC_cl_platform_id = NULL;

void _ACC_platform_init()
{
  if(is_working == true){
    _ACC_fatal("_ACC_platform_init was called more than once");
  }

  cl_uint ret_num_platforms;
  cl_int ret;

  CL_CHECK(clGetPlatformIDs(1, &_ACC_cl_platform_id, &ret_num_platforms));

  if(ret_num_platforms == 0){
    _ACC_fatal("no available cl_platform");
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

static
cl_device_type to_cl_device_type(acc_device_t device_type)
{
  _ACC_DEBUG("to_cl_device_type (acc_device_t device_type = %d)\n",device_type);
  switch(device_type){
  case acc_device_none:
    _ACC_fatal("no cl_device_type for acc_device_none");
    return CL_DEVICE_TYPE_DEFAULT; // this is dummy
  case acc_device_host:
    return CL_DEVICE_TYPE_CPU;
  case acc_device_nvidia:
    return CL_DEVICE_TYPE_DEFAULT; // this should be CL_DEVICE_TYPE_GPU
  default:
    return CL_DEVICE_TYPE_DEFAULT;
  }
}

int _ACC_platform_get_num_devices(acc_device_t device_type)
{
  _ACC_DEBUG("_ACC_platform_get_num_devices(device_type=%d)\n", device_type);
  cl_uint ret_num_devices;
  cl_device_type d_type = to_cl_device_type(device_type);

  CL_CHECK(clGetDeviceIDs(_ACC_cl_platform_id, d_type, 0, NULL, &ret_num_devices));
  _ACC_DEBUG("Number of devices = %d\n", ret_num_devices);

  return (int)ret_num_devices; //(int)_ACC_cl_num_devices;
}

bool _ACC_platform_allocate_device(int device_num)
{
  _ACC_DEBUG("_ACC_platform_allocate_device(device_num=%d)\n", device_num);

  cl_int ret;
  _ACC_cl_current_context = clCreateContext(NULL, 1, &_ACC_cl_device_ids[device_num], NULL, NULL, &ret);
  return (ret == CL_SUCCESS);
}

void _ACC_platform_set_device_type(acc_device_t device_type)
{
  _ACC_DEBUG("_ACC_platform_set_device_type(device_type=%d)\n", device_type);
  cl_device_type d_type = to_cl_device_type(device_type);
  cl_uint ret_num_devices;

  _ACC_cl_num_devices = _ACC_platform_get_num_devices(device_type);
  CL_CHECK(clGetDeviceIDs(_ACC_cl_platform_id, d_type, _ACC_cl_num_devices, _ACC_cl_device_ids, &ret_num_devices));
  if(_ACC_cl_num_devices != ret_num_devices){
    _ACC_fatal("the number of returned device_ids is not equal to requested");
  }
}

void _ACC_platform_set_device_num(int device_num /*0-based*/)
{
  _ACC_DEBUG("_ACC_platform_set_device_num(device_num=%d)\n", device_num);
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
  _ACC_init_current_device_if_not_inited();
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

const char* _ACC_cl_get_error_string(cl_int err_code)
{
  switch(err_code){
  case CL_SUCCESS                                  : return "Success";
  case CL_DEVICE_NOT_FOUND                         : return "Device not found";
  case CL_DEVICE_NOT_AVAILABLE                     : return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE                   : return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE            : return "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES                         : return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY                       : return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE             : return "Profiling information not available";
  case CL_MEM_COPY_OVERLAP                         : return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH                    : return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED               : return "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE                    : return "Build program failure";
  case CL_MAP_FAILURE                              : return "Map failure";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET             : return "Misaligned sub-buffer offset";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "Exec status error for events in wait list";
  case CL_INVALID_VALUE                            : return "Invalid value";
  case CL_INVALID_DEVICE_TYPE                      : return "Invalid device type";
  case CL_INVALID_PLATFORM                         : return "Invalid platform";
  case CL_INVALID_DEVICE                           : return "Invalid device";
  case CL_INVALID_CONTEXT                          : return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES                 : return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE                    : return "Invalid command queue";
  case CL_INVALID_HOST_PTR                         : return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT                       : return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          : return "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE                       : return "Invalid image size";
  case CL_INVALID_SAMPLER                          : return "Invalid sampler";
  case CL_INVALID_BINARY                           : return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS                    : return "Invalid build options";
  case CL_INVALID_PROGRAM                          : return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE               : return "Invalid program executable";
  case CL_INVALID_KERNEL_NAME                      : return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION                : return "Invalid kernel definition";
  case CL_INVALID_KERNEL                           : return "Invalid kernel";
  case CL_INVALID_ARG_INDEX                        : return "Invalid argument index";
  case CL_INVALID_ARG_VALUE                        : return "Invalid argument value";
  case CL_INVALID_ARG_SIZE                         : return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS                      : return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION                   : return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE                  : return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE                   : return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET                    : return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST                  : return "Invalid event wait list";
  case CL_INVALID_EVENT                            : return "Invalid event";
  case CL_INVALID_OPERATION                        : return "Invalid operation";
  case CL_INVALID_GL_OBJECT                        : return "Invalid GL object";
  case CL_INVALID_BUFFER_SIZE                      : return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL                        : return "Invalid MIP level";
  case CL_INVALID_GLOBAL_WORK_SIZE                 : return "Invalid global work size";
  }
  return "Unknown OpenCL error code";
}
