#include <stdio.h>
#include <stdarg.h>
#include "acc_internal.h"
#include "acc_gpu_internal.h"
#include "acc_gpu_data_struct.h"

static void register_memory(void *host_addr, size_t size);
static void unregister_memory(void *host_addr);

#define INIT_DEFAULT 0
#define INIT_PRESENT 1
#define INIT_PRESENTOR 2

static void init_data(int mode, _ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]);
void _ACC_init_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_DEFAULT, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}
void _ACC_pinit_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENTOR, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}
void _ACC_find_data(_ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  init_data(INIT_PRESENT, host_data_desc, device_addr, addr, type_size, dim, lower, length);
}

static void init_data(int mode, _ACC_gpu_data_t **host_data_desc, void **device_addr, void *addr, size_t type_size, int dim, unsigned long long lower[], unsigned long long length[]){
  //  va_list args;
  _ACC_gpu_data_t *host_data_d = NULL;

  // set array info
  _ACC_gpu_array_t *array_info = (_ACC_gpu_array_t *)_ACC_alloc(dim * sizeof(_ACC_gpu_array_t));
  //  va_start(args, dim);
  //printf("array");
  for(int i=0;i<dim;i++){
    array_info[i].dim_offset = lower[i];//va_arg(args, int);
    if(i != 0 && array_info[i].dim_offset != 0){
      _ACC_fatal("Non-zero lower is allowed only top dimension");
    }
    array_info[i].dim_elmnts = length[i];//va_arg(args, int);
	//printf("[%llu:%llu]", array_info[i].dim_offset, array_info[i].dim_elmnts );
  }
  //printf("\n");
  //va_end(args);
  unsigned long long accumulation = 1;
  for(int i=dim-1; i >= 0; i--){
    array_info[i].dim_acc = accumulation;
    accumulation *= array_info[i].dim_elmnts;
  }
  size_t size = accumulation * type_size;
  size_t offset = (dim > 0)? array_info[0].dim_offset * array_info[0].dim_acc * type_size : 0;

  _ACC_gpu_data_t *present_host_data_d = NULL;
  void *present_dev_addr;
  //find host_data_d
  if(mode == INIT_PRESENT || mode == INIT_PRESENTOR){
	_ACC_gpu_get_data_sub(&present_host_data_d, &present_dev_addr, addr, offset, size);
  }

  if(mode == INIT_PRESENT){
	if(present_host_data_d == NULL){
	  _ACC_fatal("data not found");
	}
  }

  // alloc & init host descriptor
  host_data_d = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
  host_data_d->host_addr = addr;
  ////device_addr
  host_data_d->offset = offset;
  host_data_d->size = size;
  /////host_data_d->is_original = true;
  /////is_pagelocked;
  /////is_original
  host_data_d->type_size = type_size;
  host_data_d->dim = dim;
  host_data_d->array_info = array_info;


  if(present_host_data_d == NULL){
	//device memory alloc
	_ACC_gpu_alloc(&(host_data_d->device_addr), size);
	host_data_d->is_original = true;

	//about pagelock
	unsigned int flags;
	cudaHostGetFlags(&flags, addr);
	cudaError_t error = cudaGetLastError();
	if(error == cudaSuccess){
	  //printf("memory is pagelocked\n");
	  host_data_d->is_pagelocked = true;
	}else{
	  //printf("memory is not pagelocked\n");
	  host_data_d->is_pagelocked = false;
	}
	host_data_d->is_registered = false;

	_ACC_gpu_add_data(host_data_d);
  }else{
	host_data_d->device_addr = (void *)((char*)(present_dev_addr) + offset);
	host_data_d->is_original = false;
    host_data_d->is_pagelocked = present_host_data_d->is_pagelocked;
    host_data_d->is_registered = present_host_data_d->is_registered;
  }

  //printf("hostaddr=%p, size=%zu, offset=%zu\n", addr, size, offset);

  // init params
  *host_data_desc = host_data_d;
  *device_addr = (void *)((char*)(host_data_d->device_addr) - offset);
}

void _ACC_finalize_data(_ACC_gpu_data_t *desc, int type) {
  //  printf("finalize\n");
  //type 0:data, 1:enter data, 2:exit data
  if((type == 0 && desc->is_original == true) || type == 2){
	//	printf("desc=%p, %d\n", desc,type);
    if(desc->is_registered == true){
      unregister_memory(desc->host_addr);
    }

    if(_ACC_gpu_remove_data(desc) == false){
      _ACC_fatal("can't remove data from data table\n");
    }
    _ACC_gpu_free(desc->device_addr);
    //desc->device_addr = NULL;
  }

  //  printf("free desc\n");
  _ACC_free(desc->array_info);
  _ACC_free(desc);
}

void _ACC_copy_data(_ACC_gpu_data_t *desc, int direction, int asyncId){
  void *host_addr = (void*)((char*)(desc->host_addr) + desc->offset);
  void *dev_addr = (void*)((char *)(desc->device_addr));
  size_t size = desc->size;

  switch(asyncId){
  case ACC_ASYNC_SYNC:
    _ACC_gpu_copy(host_addr, dev_addr, size, direction);
    break;
  case ACC_ASYNC_NOVAL:
  default:
    {
      //pagelock if data is not pagelocked
      if(desc->is_pagelocked == false && desc->is_registered == false){
	register_memory(desc->host_addr, desc->size);
	desc->is_registered = true;
      }
      _ACC_gpu_copy_async(host_addr, dev_addr, size, direction, asyncId);
    }
  }
}

void _ACC_pcopy_data(_ACC_gpu_data_t *desc, int direction, int asyncId){
  if(desc->is_original == true){
    _ACC_copy_data(desc, direction, asyncId);
  }
}

static void copy_subdata_using_pack_vector(_ACC_gpu_data_t *desc, int direction, int asyncId, unsigned long long offset, unsigned long long count, unsigned long long blocklength, unsigned long long stride)
{
  void *dev_buf;
  void *host_buf;
  int type_size = desc->type_size;
  size_t buf_size = count * blocklength * type_size;
  size_t offset_size = offset * type_size;

  //alloc buffer
  host_buf = (void *)_ACC_alloc(buf_size);
  void *mpool;
  long long mpool_pos = 0;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);

  ////
  void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset + offset_size);
  void *host_data = (void*)((char*)(desc->host_addr) + offset_size);

  if(direction == 400){
    //host to device
    _ACC_pack_vector(host_buf, host_data, count, blocklength, stride, type_size);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_HOST_TO_DEVICE, asyncId);
    _ACC_gpu_unpack_vector(dev_data, dev_buf, count, blocklength, stride, type_size, asyncId);
    cudaThreadSynchronize();
  }else{
    //device to host
    _ACC_gpu_pack_vector(dev_buf, dev_data, count, blocklength, stride, type_size, asyncId);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_DEVICE_TO_HOST, asyncId);
    cudaThreadSynchronize();
    _ACC_unpack_vector(host_data, host_buf, count, blocklength, stride, type_size);
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);

  _ACC_free(host_buf);
}

static void copy_subdata_using_pack(_ACC_gpu_data_t *desc, int direction, int isAsync, unsigned long long *trans_info){
  int i;
  int dim = desc->dim;
  void *dev_buf;
  void *host_buf = NULL;
  unsigned long long *info_length = trans_info + dim;
  const char useAsync = 0;

  unsigned long long total_elmnts = 1;
  for(i=0;i<dim;i++){
    total_elmnts *= info_length[i];
  }

  size_t buf_size = total_elmnts * desc->type_size;
  //alloc buffer
  if(useAsync){
    if(host_buf == NULL){
      cudaMallocHost((void**)&host_buf, sizeof(double)*1024*1024);
    }
  }else{
    host_buf = (void *)_ACC_alloc( buf_size);
  }

  void *mpool;
  long long mpool_pos = 0;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);
  //alloc and copy of trans_info
  unsigned long long *dev_trans_info;
  size_t trans_info_size = desc->dim * 3 * sizeof(unsigned long long);
  _ACC_gpu_mpool_alloc((void**)&dev_trans_info, trans_info_size, mpool, &mpool_pos);
  _ACC_gpu_copy(trans_info, dev_trans_info, trans_info_size, 400);


  if(direction == 400){
    //host to device
    _ACC_gpu_pack_data_host(host_buf, desc->host_addr, desc->dim, total_elmnts, desc->type_size, trans_info);
    if(useAsync){
      cudaMemcpyAsync(dev_buf, host_buf, buf_size, cudaMemcpyHostToDevice);
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, 400);
    }
    
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_unpack_data(dev_data, dev_buf, desc->dim, total_elmnts, desc->type_size, dev_trans_info);

    cudaThreadSynchronize();
  }else{
    //device to host
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_pack_data(dev_buf, dev_data, desc->dim, total_elmnts, desc->type_size, dev_trans_info);
    if(useAsync){
      cudaMemcpyAsync(host_buf, dev_buf, buf_size, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, 401);
    }
    _ACC_gpu_unpack_data_host(desc->host_addr, host_buf, desc->dim, total_elmnts, desc->type_size, trans_info);
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);
  _ACC_gpu_mpool_free(dev_trans_info, mpool);

  if(! useAsync){
    _ACC_free(host_buf);
  }
}

/*
static void find_contiguous(int dim, _ACC_gpu_array_t *array_info, int *trans_info, int start_dim, int *offset, int *blockLength, int *next_dim)
{
  int *info_lower = trans_info;
  int *info_length = trans_info + dim;
  int *info_dim_acc = trans_info + dim*2;

  //skip all full-range dim
  int i;
  for(i = start_dim; i >= 0; i--){
    if(info_lower[i] != 0 || info_length[i] != array_info[i].dim_elmnts) break;
  }
      
  if(i < 0){
    //seq
    *blockLength = array_info[0].dim_acc * array_info[0].dim_elmnts;
    *offset = 0;
    *next_dim = -1;
    return;
  }

  *blockLength = array_info[i].dim_acc * info_length[i];
  *offset = array_info[i].dim_acc * info_lower[i];
  i--; //skip sub-range dim

  //skip all range=1 dim
  for(; i >= 0; i--){
    if(info_length[i] != 1) break;
    *offset += array_info[i].dim_acc * info_lower[i];
  }
}
*/

void _ACC_copy_subdata(_ACC_gpu_data_t *desc, int direction, int asyncId, unsigned long long lower[], unsigned long long length[]){
  int dim = desc->dim;
  unsigned long long *trans_info = (unsigned long long *)_ACC_alloc(dim * 3 * sizeof(unsigned long long));
  unsigned long long *info_lower = trans_info;
  unsigned long long *info_length = trans_info + dim;
  unsigned long long *info_dim_acc = trans_info + dim*2;
  _ACC_gpu_array_t *array_info = desc->array_info;
  int i;

  // va_list args;
  // va_start(args, asyncId);
  for(i=0;i<dim;i++){
    info_lower[i] = lower[i];//va_arg(args, int);
    info_length[i] = length[i];//va_arg(args, int);
    info_dim_acc[i] = desc->array_info[i].dim_acc;
  }
  // va_end(args);

  //int next_dim;
  
  //skip all full-range dim
  for(i = dim - 1; i >= 0; i--){
    if(info_lower[i] != 0 || info_length[i] != array_info[i].dim_elmnts) break;
  }
      
  if(i < 0){
    //all data copy
	// printf("sequencial\n");
    _ACC_gpu_copy(desc->host_addr, (void*)((char *)(desc->device_addr) - desc->offset),desc-> size, direction);
    return;
  }

  unsigned long long offset, blockLength;
  blockLength = array_info[i].dim_acc * info_length[i];
  offset = array_info[i].dim_acc * info_lower[i];
  i--; //skip sub-range dim

  //skip all range=1 dim
  for(; i >= 0; i--){
    if(info_length[i] != 1) break;
    offset += array_info[i].dim_acc * info_lower[i];
  }

  if(i < 0){
    size_t offset_size = offset * desc->type_size;
    size_t size = blockLength * desc->type_size;
    //   printf("sequencial\n");
    _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset_size), (void*)((char *)(desc->device_addr) + offset_size - desc->offset), size, direction);
    return;
  }
  
  unsigned long long stride = array_info[i].dim_acc;
  unsigned long long count = 1;
  //skip all full-range dim
  for(; i >= 0; i--){
    count *= info_length[i];
    if(info_lower[i] != 0 || info_length[i] != array_info[i].dim_elmnts) break;
  }
   
  if(i >= 0){
    offset += array_info[i].dim_acc * info_lower[i];
    i--; //skip sub-range dim
  }

  //skip all range=1 dim
  for(; i >= 0; i--){
    if(info_length[i] != 1) break;
    offset += array_info[i].dim_acc * info_lower[i];
  }

  if(i < 0){
    // block stride
	//printf("block stride(%llu,%llu,%llu)\n", count, blockLength, stride);
    copy_subdata_using_pack_vector(desc, direction, asyncId, offset, count, blockLength, stride);
    return;
  }

  //printf("unknown\n");
  copy_subdata_using_pack(desc, direction, asyncId, trans_info);
}

static void register_memory(void *host_addr, size_t size){
  //  printf("register_memory\n");
  cudaError_t cuda_err = cudaHostRegister(host_addr, size, cudaHostRegisterPortable);
  if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }
}

static void unregister_memory(void *host_addr){
  //  printf("unregister_memory\n");
  cudaError_t cuda_err = cudaHostUnregister(host_addr);
  if(cuda_err != cudaSuccess){
    _ACC_gpu_fatal(cuda_err);
  }
}



