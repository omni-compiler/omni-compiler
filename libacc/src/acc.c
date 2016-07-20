#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "acc_internal.h"

const static char NVIDIA[] = "NVIDIA";
const static char HOST[] = "HOST";
const static char NONE[] = "NONE";
static bool _ACC_runtime_working = false;
static bool _ACC_device_working = false;

const static acc_device_t default_device = acc_device_nvidia;
const static acc_device_t default_not_host_device = acc_device_nvidia;

static void init_device(int dev_num);
static int _num_devices;
static int _default_device_num = 0;
static int _device_num = 0; //normalized

typedef struct acc_context{
  bool isInitialized;
  _ACC_queue_map_t *queue_map;
  _ACC_mpool_t *mpool;
}acc_context;

acc_context *contexts;
static void _ACC_init_device_if_not_inited(int num/*0-based*/);

int _ACC_num_gangs_limit = -1; //negative num means not-set, and 0 means no limit

void _ACC_init(int argc, char** argv)
{
  _ACC_DEBUG("begin _ACC_init\n")
  if (_ACC_runtime_working) {
    _ACC_fatal("_ACC_init was called more than once");
  }
  _ACC_runtime_working = true;

  _ACC_platform_init();
  
  //get device type
  acc_device_t device_t = acc_device_default; //set default type
  char *acc_device_type_str = getenv("ACC_DEVICE_TYPE");
  if(acc_device_type_str != NULL){ // if not defined
    if (strncmp(acc_device_type_str, NVIDIA, sizeof(NVIDIA)) == 0){
      device_t = acc_device_nvidia;
    }else if(strncmp(acc_device_type_str, HOST, sizeof(HOST)) == 0){
      device_t = acc_device_host;
    }else if(strncmp(acc_device_type_str, NONE, sizeof(NONE)) == 0){
      device_t = acc_device_none;
    }else{
      _ACC_fatal("ACC_DEVICE_TYPE is invalid");
    }
  }
  device_t = _ACC_normalize_device_type(device_t); //normalize

  //init runtime for device_t   //_ACC_initialize(device_t);
  /* switch(device_t){ */
  /* case acc_device_none: */
  /*   _ACC_fatal("device_type = none is unsupported"); */
  /*   break; */
  /* case acc_device_host: */
  /*   _ACC_fatal("device_type = host is unsupported"); */
  /*   break; */
  /* case acc_device_nvidia: */
  /*   _ACC_gpu_init(); */
  /*   break; */
  /* default: */
  /*   _ACC_fatal("unknown device_type\n"); */
  /* } */

  _ACC_init_type(device_t);

  //get device number
  int device_num = 0; //default device_num is 0
  char *acc_device_num_str = getenv("ACC_DEVICE_NUM");
  if(acc_device_num_str != NULL){
    device_num = atoi(acc_device_num_str);
  }

  acc_set_device_num(device_num, device_t);

  //get num_gangs limit
  char *omni_acc_num_gangs_limit = getenv("OMNI_ACC_NUM_GANGS_LIMIT");
  if(omni_acc_num_gangs_limit != NULL){
    _ACC_num_gangs_limit = atoi(omni_acc_num_gangs_limit);
    if(_ACC_num_gangs_limit < 0){
      _ACC_fatal("invalid value for OMNI_ACC_NUM_GANGS_LIMIT");
    }
  }

  _ACC_DEBUG("end _ACC_init\n")
}

void _ACC_finalize(void)
{
  _ACC_DEBUG("begin\n");

  if (!_ACC_runtime_working) {
    _ACC_fatal("_ACC_finalize was called before _ACC_init");
  }

  //finalize runtime for current_device_type
  /* switch(acc_get_device_type()){ */
  /* case acc_device_none: */
  /*   _ACC_fatal("device_type = none is unsupported"); */
  /*   break; */
  /* case acc_device_host: */
  /*   _ACC_fatal("device_type = host is unsupported"); */
  /*   break; */
  /* case acc_device_nvidia: */
  /*   _ACC_gpu_finalize(); */
  /*   break; */
  /* default: */
  /*   _ACC_fatal("unknown device type"); */
  /* } */
  _ACC_finalize_type(acc_get_device_type());

  _ACC_platform_finalize();
  
  _ACC_runtime_working = false;
  _ACC_DEBUG("end\n");
}

void _ACC_init_type(acc_device_t device_type)
{ 
  switch(device_type){
  case acc_device_none:
    _ACC_fatal("device_type = none is unsupported");
    break;
  case acc_device_host:
    _ACC_fatal("device_type = host is unsupported");
    break;
  case acc_device_nvidia:
    break;
  default:
    _ACC_fatal("unknown device_type\n");
  }

  if(_ACC_device_working){
    if(acc_get_device_type() == device_type){
      return;
    }else{
      _ACC_fatal("try to init device although another type device is running");
    }
  }

  _ACC_DEBUG("begin _ACC_init_type\n")

  _num_devices = _ACC_platform_get_num_devices();

  _ACC_DEBUG("Total number of devices = %d\n", _num_devices)

  contexts = (acc_context*)_ACC_alloc(sizeof(acc_context) * _num_devices);
  for(int i = 0; i< _num_devices; i++){
    contexts[i].isInitialized = false;
  }

  acc_set_device_type(device_type);
  _ACC_set_device_num(-10); //set device to default

  _ACC_device_working = true;

  _ACC_DEBUG("end _ACC_init_type\n")
}

void _ACC_init_api(void)
{
  _ACC_init_current_device_if_not_inited();
}

void _ACC_finalize_type(acc_device_t device_type)
{
  switch(device_type){
  case acc_device_none:
    return;
  case acc_device_host:
    _ACC_fatal("device_type = host is unsupported");
    break;
  case acc_device_nvidia:
    break;
  default:
    _ACC_fatal("unknown device type");
  }

  if(! _ACC_device_working){
    _ACC_fatal("device is not initialized\n");
  }

  if(acc_get_device_type() != device_type){
    _ACC_fatal("device type is not matched");
  }

  //finalize each device
  for(int device_num = 0; device_num < _num_devices; device_num++){
    if(! contexts[device_num].isInitialized) continue;

    _ACC_set_device_num(device_num); //0-based

    _ACC_gpu_finalize_stream_map(contexts[device_num].queue_map);
    _ACC_mpool_destroy(contexts[device_num].mpool);
    contexts[device_num].isInitialized = false;

    //XXX add _ACC_platform_free_device()
  }

  acc_set_device_type(acc_device_none);
  _ACC_free(contexts);
  contexts = NULL;
  _ACC_device_working = false;
}

void _ACC_set_device_num(int device_num)
{
  /* device_num is 0-origin */
  /* negative value means default devcie num */
  _ACC_DEBUG("set_device_num(%d)\n",device_num)

  if(device_num >= _num_devices){
    _ACC_fatal("invalid device num in _ACC_gpu_set_device_num");
  }

  int actual_device_num = _ACC_normalize_device_num(device_num);
  _device_num = actual_device_num;
  _ACC_platform_set_device_num(actual_device_num);
}

int _ACC_get_device_num(){ //returns 0-based num
  return _device_num;
}

int _ACC_normalize_device_num(int n)
{
  if(n < 0){
    return _default_device_num;
  }else{
    return n;
  }
}

acc_device_t _ACC_normalize_device_type(acc_device_t device_type)
{
  switch(device_type){
  case acc_device_none:
  case acc_device_host:
  case acc_device_nvidia:
    return device_type;

  case acc_device_default:
    return default_device;
  case acc_device_not_host:
    return default_not_host_device;

  default:
    _ACC_fatal("acc_set_device_type : unknown device type");
    return acc_device_none;
  }
}

void _ACC_init_current_device_if_not_inited()
{
  _ACC_init_device_if_not_inited(_device_num);
}

_ACC_queue_map_t* _ACC_get_queue_map()
{
  _ACC_DEBUG("get_current_queue_map\n")
  
  _ACC_init_current_device_if_not_inited();
  return contexts[_device_num].queue_map;
}

_ACC_mpool_t* _ACC_get_mpool()
{
  _ACC_DEBUG("get_mpool\n")

  _ACC_init_current_device_if_not_inited();
  return (_ACC_mpool_t*)(contexts[_device_num].mpool);
}


//Fortran interfaces
void acc_init_(int *argc, char** argv)
{
  _ACC_init(*argc, argv);
}
void acc_finalize_()
{
  _ACC_finalize();
}


//internal functions
static void _ACC_init_device_if_not_inited(int num)
{
  if(! _ACC_runtime_working){
    _ACC_init(0, NULL);
  }
  if(! _ACC_device_working){
    _ACC_fatal("current device type is not initialized");
  }
  if(! contexts[_ACC_normalize_device_num(num)].isInitialized){
    init_device(num);
  }
}

static void init_device(int dev_num){ //0-based, notnormalized
  _ACC_DEBUG("initializing GPU %d\n",dev_num)

  if(dev_num < 0){ //default device num
    int i;
    for(i = 0; i < _num_devices; i++){
      if( _ACC_platform_allocate_device(i) == true){
	_default_device_num = i;
	break;
      }
    }
    if(i == _num_devices){
      _ACC_fatal("failed to alloc GPU device");
    }
  }else{
    if(! _ACC_platform_allocate_device(dev_num)){
      _ACC_fatal("failed to alloc GPU device");
    }
  }

  int normalized_device_num = _ACC_normalize_device_num(dev_num);
  _ACC_platform_init_device(normalized_device_num);

  //init mpool
  contexts[normalized_device_num].isInitialized = true;
  contexts[normalized_device_num].mpool = _ACC_mpool_create();
  //init stream hashmap
  contexts[normalized_device_num].queue_map = _ACC_gpu_init_stream_map(16);
}
