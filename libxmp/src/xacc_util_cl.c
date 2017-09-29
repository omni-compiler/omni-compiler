#include "xmp_internal.h"
#include "xacc_internal.h"
#include <sys/stat.h>

#if 1 // if Omni OpenACC is enabled
#include "../../libacc/include/openacc.h"
#endif

static unsigned char _kernel_src[] = {
#include "xacc_util_cl_kernel.hex"
};

void _XACC_queue_create(_XACC_queue_t *queue)
{
    cl_int ret;
    cl_context context = (cl_context)acc_get_current_opencl_context();
    cl_device_id device_id = (cl_device_id)acc_get_current_opencl_device();

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0 /*prop*/, &ret);
    CL_CHECK(ret);

    XACC_DEBUG("queue create %p", command_queue);
    *queue = command_queue;
}

void _XACC_queue_destroy(_XACC_queue_t *queue)
{
    XACC_DEBUG("queue destroy %p", *queue);
    CL_CHECK(clFinish(*queue));
    CL_CHECK(clReleaseCommandQueue(*queue));
    *queue = NULL;
}

void _XACC_queue_wait(_XACC_queue_t queue)
{
    CL_CHECK(clFinish(queue));
}



void _XACC_memory_alloc(_XACC_memory_t *memory, size_t size)
{
    cl_int ret;
    cl_context context = (cl_context)acc_get_current_opencl_context();

    cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
    CL_CHECK(ret);

    XACC_DEBUG("alloc %p, %zd", mem, size);
    *memory = mem;
}

void _XACC_memory_free(_XACC_memory_t *memory)
{
    if(*memory == NULL) return;

    XACC_DEBUG("free %p\n", *memory);
    CL_CHECK(clReleaseMemObject(*memory));
    *memory = NULL;
}

void _XACC_memory_read(void *addr, _XACC_memory_t memory, size_t memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    cl_bool is_blocking_cl = is_blocking? CL_TRUE : CL_FALSE;

    XACC_DEBUG("clEnqueueReadBuffer(cq=%p, mem=%p, blocking=%d, offset=%zd, size=%zd, addr=%p)", queue, memory, is_blocking_cl, memory_offset, size, addr);
    CL_CHECK(clEnqueueReadBuffer(queue, memory, is_blocking_cl, memory_offset, size, addr,
				 0 /*num_wait_ev*/, NULL /*wait_ev_list*/, NULL /*ev*/));
}
void _XACC_memory_write(_XACC_memory_t memory, size_t memory_offset, void *addr, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    cl_bool is_blocking_cl = is_blocking? CL_TRUE : CL_FALSE;

    XACC_DEBUG("clEnqueueWriteBuffer(cq=%p, mem=%p, blocking=%d, offset=%zd, size=%zd, addr=%p)", queue, memory, is_blocking_cl, memory_offset, size, addr);
    CL_CHECK(clEnqueueWriteBuffer(queue, memory, is_blocking_cl, memory_offset, size, addr,
				  0 /*num_wait_ev*/, NULL /*wait_ev_list*/, NULL /*ev*/));
}

void _XACC_memory_copy(_XACC_memory_t dst_memory, size_t dst_memory_offset, _XACC_memory_t src_memory, size_t src_memory_offset, size_t size, _XACC_queue_t queue, bool is_blocking)
{
    CL_CHECK(clEnqueueCopyBuffer(queue, src_memory, dst_memory, src_memory_offset, dst_memory_offset, size, 0 /*num_wait_ev*/, NULL /*wait_ev_list*/, NULL /*ev*/));

    if(is_blocking){
	_XACC_queue_wait(queue);
    }
}
void* _XACC_memory_get_address(_XACC_memory_t memory)
{
    _XACC_fatal("cannot get raw address in OpenCL");
    return NULL; //dummy
}

void _XACC_host_malloc(void **ptr, size_t size)
{
    *ptr = _XMP_alloc(size);
}
void _XACC_host_free(void **ptr)
{
    _XMP_free(*ptr);
    *ptr = NULL;
}



static
cl_program create_and_build_program(const char * kernel_src, const size_t kernel_src_size)
{
    cl_int ret;
    cl_program program;

    cl_context context = (cl_context)acc_get_current_opencl_context();
    const char *srcs[] = {kernel_src};
    const size_t src_sizes[] = {kernel_src_size};
    program = clCreateProgramWithSource(context, 1, srcs, src_sizes, &ret);
    CL_CHECK(ret);

    //build program
    cl_device_id device_id = (cl_device_id)acc_get_current_opencl_device();
    const char build_option[] = "";
    ret = clBuildProgram(program, 1, &device_id, build_option, NULL, NULL);
    if(ret != CL_SUCCESS){
	//print build error
	const int max_error_length = 1024*1024;
	size_t returned_size;
	char *error_log = _XMP_alloc(sizeof(char) * max_error_length + 1);
	CL_CHECK(clGetProgramBuildInfo(program,
				       device_id,
				       CL_PROGRAM_BUILD_LOG,
				       max_error_length, error_log, &returned_size));
	fprintf(stderr, "build log:\n%s\n", error_log);
	_XMP_free(error_log);
	exit(1);
    }

    return program;
}

#if 0
static
cl_program create_and_build_program_from_file(const char * kernel_src_filename)
{
    //open kernel file
    FILE *fp = fopen(kernel_src_filename, "r");
    if(fp == NULL){
	fprintf(stderr, "Failed to open kernel file %s.\n", kernel_src_filename);
	exit(1);
    }

    //get kernel file size
    struct stat filestat;
    stat(kernel_src_filename, &filestat);

    //read kernel file
    size_t kernel_src_size = filestat.st_size;
    char *kernel_src = (char*)_XMP_alloc(sizeof(char)*(kernel_src_size));
    size_t read_byte = fread(kernel_src, sizeof(char), kernel_src_size, fp);

    if(read_byte < kernel_src_size){
	_XACC_fatal("faild to read kernel_file");
    }

    //close kernel source file
    fclose(fp);

    XACC_DEBUG("filesize = %ld\n", kernel_src_size);
    XACC_DEBUG("read bytes %zd\n", read_byte);

    //create program
    XACC_DEBUG("create program \"%s\"\n", kernel_src_filename);

    cl_program program = create_and_build_program(kernel_src, kernel_src_size);

    _XMP_free(kernel_src);

    return program;
}
#endif

void create_kernels(cl_kernel kernels[], cl_program program, int num_kernels, const char *kernel_names[])
{
    cl_int ret;
    //create kernels
    for(int i = 0; i < num_kernels; i++){
	XACC_DEBUG("create kernel \"%s\"\n", kernel_names[i]);
	kernels[i] = clCreateKernel(program, kernel_names[i], &ret);
	CL_CHECK(ret);
    }
}

#define KERNEL_FUNCTIONS \
    NAME(_XACC_pack_vector),\
    NAME(_XACC_pack_vector_8),\
    NAME(_XACC_pack_vector_16),\
    NAME(_XACC_pack_vector_32),\
    NAME(_XACC_pack_vector_64),\
    NAME(_XACC_unpack_vector),\
    NAME(_XACC_unpack_vector_8),\
    NAME(_XACC_unpack_vector_16),\
    NAME(_XACC_unpack_vector_32),\
    NAME(_XACC_unpack_vector_64)

#define DEF_ENUM(...) enum {__VA_ARGS__, _XACC_num_kernels};
#define DEF_STRARRAY(...) const static char *_kernel_names[] = {__VA_ARGS__};

#define NAME(s) s
DEF_ENUM(KERNEL_FUNCTIONS)
#undef NAME

#define NAME(s) #s
DEF_STRARRAY(KERNEL_FUNCTIONS)
#undef NAME

static cl_program _program;
static cl_kernel _kernels[_XACC_num_kernels];

void _XACC_util_init()
{
    // _program = create_and_build_program_from_file("util.cl");
    _program = create_and_build_program((char*)_kernel_src, strlen((char*)_kernel_src));

    create_kernels(_kernels, _program, _XACC_num_kernels, _kernel_names);
}


static
void enqueue_kernel(cl_command_queue command_queue, cl_kernel kernel, int num_args, void *args[], size_t arg_sizes[], cl_uint work_dim, const size_t *global_work_size, const size_t *local_work_size)
{
  int i;
  for(i=0;i<num_args; i++){
    CL_CHECK(clSetKernelArg(kernel, i, arg_sizes[i], args[i]));
  }

  CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, /*num_ev_wait*/ 0, /*ev_wait_list*/ NULL, /*ev*/ NULL));
}

#define CHECK_MULTIPLE(size, base) if(size % base != 0){_XACC_fatal(#size " is not a multiple of " #base);}

void _XACC_memory_pack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
			      _XACC_memory_t src_mem, size_t src_offset,
			      size_t blocklength, size_t stride, size_t count,
			      size_t typesize,
			      _XACC_queue_t queue, bool is_blocking)
{
    XACC_DEBUG("pack_vector, dst=%p, dst_off=%zd, src=%p, src_off=%zd, blklen=%zd, stride=%zd, count=%zd, typesize=%zd, queue=%p, is_blocking=%d\n",
	       dst_mem, dst_offset, src_mem, src_offset, blocklength, stride, count, typesize, queue, is_blocking);

    const int numThreads = 128; //must be 2^n

    CHECK_MULTIPLE(blocklength, typesize);
    CHECK_MULTIPLE(stride, typesize);
    CHECK_MULTIPLE(dst_offset, typesize);
    CHECK_MULTIPLE(src_offset, typesize);
    size_t blocklength_e = blocklength / typesize;
    size_t stride_e = stride / typesize;
    size_t dst_offset_e = dst_offset / typesize;
    size_t src_offset_e = src_offset / typesize;

    int bx = 1, by;
    int tx = 1, ty;
    if(blocklength_e >= numThreads){
	tx = numThreads;
    }else{
	while(tx < blocklength_e){
	    tx <<= 1;
	}
    }
    ty = numThreads / tx;
    by = (count-1)/ty + 1;

    size_t global_work_size[2] = {bx*tx, by*ty};
    size_t local_work_size[2] = {tx, ty};

    void *args[] = {&dst_mem, &dst_offset_e, &src_mem, &src_offset_e, &blocklength_e, &stride_e, &count};
    size_t arg_sizes[] = {sizeof(dst_mem), sizeof(dst_offset_e), sizeof(src_mem), sizeof(src_offset_e), sizeof(blocklength_e), sizeof(stride_e), sizeof(count)};

    switch(typesize){
    case 1:
	XACC_DEBUG("pack_vector_8\n");
	enqueue_kernel(queue, _kernels[_XACC_pack_vector_8], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 2:
	XACC_DEBUG("pack_vector_16\n");
	enqueue_kernel(queue, _kernels[_XACC_pack_vector_16], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 4:
	XACC_DEBUG("pack_vector_32\n");
	enqueue_kernel(queue, _kernels[_XACC_pack_vector_32], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 8:
	XACC_DEBUG("pack_vector_64\n");
	enqueue_kernel(queue, _kernels[_XACC_pack_vector_64], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    default:
	{
	    void *args_default[] = {&dst_mem, &dst_offset, &src_mem, &src_offset, &blocklength, &stride, &count};
	    size_t arg_sizes_default[] = {sizeof(dst_mem), sizeof(dst_offset), sizeof(src_mem), sizeof(src_offset), sizeof(blocklength), sizeof(stride), sizeof(count)};

	    XACC_DEBUG("pack_vector_default\n");
	    enqueue_kernel(queue, _kernels[_XACC_pack_vector], 7, args_default, arg_sizes_default, 2, global_work_size, local_work_size);
	}
    }
}
void _XACC_memory_unpack_vector(_XACC_memory_t dst_mem, size_t dst_offset,
				_XACC_memory_t src_mem, size_t src_offset,
				size_t blocklength, size_t stride, size_t count,
				size_t typesize,
				_XACC_queue_t queue, bool is_blocking)
{
    XACC_DEBUG("unpack_vector, dst=%p, dst_off=%zd, src=%p, src_off=%zd, blklen=%zd, stride=%zd, count=%zd, typesize=%zd, queue=%p, is_blocking=%d\n",
	       dst_mem, dst_offset, src_mem, src_offset, blocklength, stride, count, typesize, queue, is_blocking);

    const int numThreads = 128; //must be 2^n

    CHECK_MULTIPLE(blocklength, typesize);
    CHECK_MULTIPLE(stride, typesize);
    CHECK_MULTIPLE(dst_offset, typesize);
    CHECK_MULTIPLE(src_offset, typesize);
    size_t blocklength_e = blocklength / typesize;
    size_t stride_e = stride / typesize;
    size_t dst_offset_e = dst_offset / typesize;
    size_t src_offset_e = src_offset / typesize;

    int bx = 1, by;
    int tx = 1, ty;

    if(blocklength_e >= numThreads){
	tx = numThreads;
    }else{
	while(tx < blocklength_e){
	    tx <<= 1;
	}
    }
    ty = numThreads / tx;
    by = (count-1)/ty + 1;

    size_t global_work_size[2] = {bx*tx, by*ty};
    size_t local_work_size[2] = {tx, ty};

    void *args[] = {&dst_mem, &dst_offset_e, &src_mem, &src_offset_e, &blocklength_e, &stride_e, &count};
    size_t arg_sizes[] = {sizeof(dst_mem), sizeof(dst_offset_e), sizeof(src_mem), sizeof(src_offset_e), sizeof(blocklength_e), sizeof(stride_e), sizeof(count)};

    switch(typesize){
    case 1:
	XACC_DEBUG("unpack_vector_8\n");
	enqueue_kernel(queue, _kernels[_XACC_unpack_vector_8], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 2:
	XACC_DEBUG("unpack_vector_16\n");
	enqueue_kernel(queue, _kernels[_XACC_unpack_vector_16], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 4:
	XACC_DEBUG("unpack_vector_32\n");
	enqueue_kernel(queue, _kernels[_XACC_unpack_vector_32], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    case 8:
	XACC_DEBUG("unpack_vector_64\n");
	enqueue_kernel(queue, _kernels[_XACC_unpack_vector_64], 7, args, arg_sizes, 2, global_work_size, local_work_size);
	break;
    default:
	{
	    void *args_default[] = {&dst_mem, &dst_offset, &src_mem, &src_offset, &blocklength, &stride, &count};
	    size_t arg_sizes_default[] = {sizeof(dst_mem), sizeof(dst_offset), sizeof(src_mem), sizeof(src_offset), sizeof(blocklength), sizeof(stride), sizeof(count)};

	    XACC_DEBUG("unpack_vector_default\n");
	    enqueue_kernel(queue, _kernels[_XACC_unpack_vector], 7, args_default, arg_sizes_default, 2, global_work_size, local_work_size);
	}
    }
}
void _XACC_memory_pack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
			       _XACC_memory_t src0_mem, size_t src0_offset,
			       size_t blocklength0, size_t stride0, size_t count0,
			       _XACC_memory_t dst1_mem, size_t dst1_offset,
			       _XACC_memory_t src1_mem, size_t src1_offset,
			       size_t blocklength1, size_t stride1, size_t count1,
			       size_t typesize,
			       _XACC_queue_t queue, bool is_blocking)
{
}
void _XACC_memory_unpack_vector2(_XACC_memory_t dst0_mem, size_t dst0_offset,
				 _XACC_memory_t src0_mem, size_t src0_offset,
				 size_t blocklength0, size_t stride0, size_t count0,
				 _XACC_memory_t dst1_mem, size_t dst1_offset,
				 _XACC_memory_t src1_mem, size_t src1_offset,
				 size_t blocklength1, size_t stride1, size_t count1,
				 size_t typesize,
				 _XACC_queue_t queue, bool is_blocking)
{
}
