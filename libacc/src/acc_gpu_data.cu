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

  _ACC_gpu_data_list_t *present_data = NULL;
  //find host_data_d
  if(mode == INIT_PRESENT || mode == INIT_PRESENTOR){
    present_data = _ACC_gpu_find_data(addr, offset, size);
  }

  if(mode == INIT_PRESENT){
	if(present_data == NULL){
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


  if(present_data == NULL){
	//device memory alloc
	_ACC_gpu_alloc(&(host_data_d->device_addr), size);
	host_data_d->is_original = true;

	//about pagelock
	/*
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
	*/
	host_data_d->is_pagelocked = _ACC_gpu_is_pagelocked(addr);
	host_data_d->is_registered = false;

	_ACC_gpu_add_data(host_data_d);
  }else{
	/* host_data_d->device_addr corresponds to (host_data_d->host_addr + host_data_d->offset) */
        host_data_d->device_addr = (char*)present_data->device_addr
	  + ((char*)addr - (char*)present_data->host_addr) + (offset - present_data->offset);
	host_data_d->is_original = false;
	host_data_d->is_pagelocked = present_data->is_pagelocked;
	host_data_d->is_registered = present_data->is_registered;
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

    if(_ACC_gpu_remove_data(desc->device_addr, desc->size) == false){
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
  size_t type_size = desc->type_size;
  size_t buf_size = count * blocklength * type_size;
  size_t offset_size = offset * type_size;

  //alloc buffer
  void *host_buf = (void *)_ACC_alloc(buf_size);
  void *mpool;
  long long mpool_pos = 0;
  void *dev_buf;
  _ACC_gpu_mpool_get(&mpool);
  _ACC_gpu_mpool_alloc((void**)&dev_buf, buf_size, mpool, &mpool_pos);

  ////
  void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset + offset_size);
  void *host_data = (void*)((char*)(desc->host_addr) + offset_size);

  if(direction == _ACC_GPU_COPY_HOST_TO_DEVICE){
    //host to device
    _ACC_pack_vector(host_buf, host_data, count, blocklength, stride, type_size);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_HOST_TO_DEVICE, asyncId);
    _ACC_gpu_unpack_vector(dev_data, dev_buf, count, blocklength, stride, type_size, asyncId);
    cudaThreadSynchronize();
  }else if(direction == _ACC_GPU_COPY_DEVICE_TO_HOST){
    //device to host
    _ACC_gpu_pack_vector(dev_buf, dev_data, count, blocklength, stride, type_size, asyncId);
    _ACC_gpu_copy_async(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_DEVICE_TO_HOST, asyncId);
    cudaThreadSynchronize();
    _ACC_unpack_vector(host_data, host_buf, count, blocklength, stride, type_size);
  }else{
    _ACC_fatal("bad direction");
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);

  _ACC_free(host_buf);
}

static void copy_subdata_using_pack(_ACC_gpu_data_t *desc, int direction, int isAsync, unsigned long long lowers[], unsigned long long lengths[]){
  int i;
  int dim = desc->dim;
  void *dev_buf;
  void *host_buf = NULL;
  const char useAsync = 0;

  unsigned long long total_elmnts = 1;
  for(i=0;i<dim;i++){
    total_elmnts *= lengths[i];
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
  unsigned long long host_trans_info[desc->dim * 3];
  for(int i = 0; i < dim; i++){
    host_trans_info[i + dim * 0] = lowers[i];
    host_trans_info[i + dim * 1] = lengths[i];
    host_trans_info[i + dim * 2] = desc->array_info[i].dim_acc;;
  }
  size_t trans_info_size = desc->dim * 3 * sizeof(unsigned long long);
  _ACC_gpu_mpool_alloc((void**)&dev_trans_info, trans_info_size, mpool, &mpool_pos);
  _ACC_gpu_copy(host_trans_info, dev_trans_info, trans_info_size, _ACC_GPU_COPY_HOST_TO_DEVICE);


  if(direction == _ACC_GPU_COPY_HOST_TO_DEVICE){
    //host to device
    _ACC_gpu_pack_data_host(host_buf, desc->host_addr, desc->dim, total_elmnts, desc->type_size, host_trans_info);
    if(useAsync){
      cudaMemcpyAsync(dev_buf, host_buf, buf_size, cudaMemcpyHostToDevice);
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_HOST_TO_DEVICE);
    }
    
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_unpack_data(dev_data, dev_buf, desc->dim, total_elmnts, desc->type_size, dev_trans_info);

    cudaThreadSynchronize();
  }else if(direction == _ACC_GPU_COPY_DEVICE_TO_HOST){
    //device to host
    void *dev_data = (void*)((char*)(desc->device_addr) - desc->offset);
    _ACC_gpu_pack_data(dev_buf, dev_data, desc->dim, total_elmnts, desc->type_size, dev_trans_info);
    if(useAsync){
      cudaMemcpyAsync(host_buf, dev_buf, buf_size, cudaMemcpyDeviceToHost);
      cudaThreadSynchronize();
    }else{
      _ACC_gpu_copy(host_buf, dev_buf, buf_size, _ACC_GPU_COPY_DEVICE_TO_HOST);
    }
    _ACC_gpu_unpack_data_host(desc->host_addr, host_buf, desc->dim, total_elmnts, desc->type_size, host_trans_info);
  }else{
    _ACC_fatal("bad direction");
  }

  //free buffer
  _ACC_gpu_mpool_free(dev_buf, mpool);
  _ACC_gpu_mpool_free(dev_trans_info, mpool);

  if(! useAsync){
    _ACC_free(host_buf);
  }
}

void _ACC_copy_subdata(_ACC_gpu_data_t *desc, int direction, int asyncId, unsigned long long lowers[], unsigned long long lengths[]){
  int dim = desc->dim;
  _ACC_gpu_array_t *array_info = desc->array_info;

  unsigned long long count = 1;
  unsigned long long blocklength = 1;
  unsigned long long stride = 1;
  unsigned long long offset = 0;
  char isblockstride = 1;

  for(int i = 0 ; i < dim; i++){
    unsigned long long dim_elmnts = array_info[i].dim_elmnts;
    if(blocklength == 1 || (lowers[i] == 0 && lengths[i] == dim_elmnts)){
      blocklength *= lengths[i];
      stride *= dim_elmnts;
      offset = offset * dim_elmnts + lowers[i];
    }else if(count == 1){
      count = blocklength;
      blocklength = lengths[i];
      stride = dim_elmnts;
      offset = offset * dim_elmnts + lowers[i];
    }else{
      isblockstride = 0;
      break;
    }
  }
  
  if(!isblockstride){
    copy_subdata_using_pack(desc, direction, asyncId, lowers, lengths);
  }else if(count == 1){
    size_t offset_size = offset * desc->type_size;
    size_t size = blocklength * desc->type_size;
    _ACC_gpu_copy((void*)((char*)(desc->host_addr) + offset_size), (void*)((char *)(desc->device_addr) + offset_size - desc->offset), size, direction);
  }else{
    copy_subdata_using_pack_vector(desc, direction, asyncId, offset, count, blocklength, stride);
  }
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

void _ACC_gpu_map_data(void *host_addr, void* device_addr, size_t size)
{
  if(_ACC_gpu_find_data(host_addr, 0, size) != NULL){
    _ACC_fatal("map_data: already mapped\n");
  }

  // alloc & init host descriptor
  _ACC_gpu_data_t *host_data_d = (_ACC_gpu_data_t *)_ACC_alloc(sizeof(_ACC_gpu_data_t));
  host_data_d->host_addr = host_addr;
  ////device_addr
  host_data_d->offset = 0;
  host_data_d->size = size;
  host_data_d->type_size = 1; //type_size;
  host_data_d->dim = 1;
  {
    _ACC_gpu_array_t *array_info = (_ACC_gpu_array_t *)_ACC_alloc(sizeof(_ACC_gpu_array_t));
    array_info->dim_offset = 0;
    array_info->dim_elmnts = size;
    array_info->dim_acc = 1;
    host_data_d->array_info = array_info;
  }

  host_data_d->device_addr = device_addr;
  host_data_d->is_original = false;

  //about pagelock
  host_data_d->is_pagelocked = _ACC_gpu_is_pagelocked(host_addr);
  host_data_d->is_registered = false;

  _ACC_gpu_add_data(host_data_d);
  //printf("hostaddr=%p, size=%zu, offset=%zu\n", addr, size, offset);
}

void _ACC_gpu_unmap_data(void *host_addr)
{
  _ACC_gpu_data_list_t* data = _ACC_gpu_find_data(host_addr, 0/*offset*/, 1/*size*/);
  if(_ACC_gpu_remove_data(data->device_addr, data->size) == false){
    _ACC_fatal("can't remove data from data table\n");
  }
}


