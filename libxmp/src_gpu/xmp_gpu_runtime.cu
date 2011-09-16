#include <stdlib.h>
#include <unistd.h>
#include "xmp_gpu_internal.h"
#define BUF_LEN 256

int _XMP_gpu_device_count;

int _XMP_gpu_max_thread;

int _XMP_gpu_max_block_dim_x;
int _XMP_gpu_max_block_dim_y;
int _XMP_gpu_max_block_dim_z;

static void *_XMP_gpu_dummy;

static int _XMP_gpu_select_device(void) {
  char host_name[BUF_LEN];
  if (gethostname(host_name, BUF_LEN) < 0) {
    _XMP_fatal("fail to init GPU device");
  }

  char color_str[BUF_LEN];
  int index = 0;
  for (int i = 0; i < BUF_LEN; i++) {
    char c = host_name[i];
    if ((c == '.') || (c == '\0')) {
      color_str[index] = '\0';
      return (_XMP_split_world_by_color(atoi(color_str)) % _XMP_gpu_device_count);
    } else if ((c >= '0') && (c <= '9')) {
      color_str[index] = c;
      index++;
    }
  }

  _XMP_fatal("fail to init GPU device");
  // dummy
  return 0;
}

extern "C" void _XMP_gpu_init(void) {
  cudaGetDeviceCount(&_XMP_gpu_device_count);

  if (_XMP_gpu_device_count == 0) {
    _XMP_fatal("no GPU device found");
  }

  int dev_num = _XMP_gpu_select_device();
  if (cudaSetDevice(dev_num) != cudaSuccess) {
    _XMP_fatal("fail to init GPU device");
  }

  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties(&dev_prop, dev_num);

  _XMP_gpu_max_thread = dev_prop.maxThreadsPerBlock;

  _XMP_gpu_max_block_dim_x = dev_prop.maxGridSize[0];
  _XMP_gpu_max_block_dim_y = dev_prop.maxGridSize[1];
  _XMP_gpu_max_block_dim_z = dev_prop.maxGridSize[2];

  _XMP_gpu_alloc(&_XMP_gpu_dummy, sizeof(int));
  _XMP_gpu_free(_XMP_gpu_dummy);
}

extern "C" void _XMP_gpu_finalize(void) {
  return;
}
