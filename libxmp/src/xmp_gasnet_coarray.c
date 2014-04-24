#include <stdarg.h>
#include "mpi.h"
#include "xmp_internal.h"
#include "xmp_atomic.h"
static unsigned long long _xmp_heap_size, _xmp_stride_size;
static int *_xmp_gasnet_stride_queue;
static int _xmp_gasnet_stride_wait_size = 0;
static int _xmp_gasnet_stride_queue_size = _XMP_GASNET_STRIDE_INIT_SIZE;
static unsigned long long _xmp_coarray_shift = 0;
static char **_xmp_gasnet_buf;
volatile static int done_get_flag;
#define UNROLLING (4)

gasnet_handlerentry_t htable[] = {
  { _XMP_GASNET_LOCK_REQUEST,               _xmp_gasnet_lock_request },
  { _XMP_GASNET_SETLOCKSTATE,               _xmp_gasnet_setlockstate },
  { _XMP_GASNET_UNLOCK_REQUEST,             _xmp_gasnet_unlock_request },
  { _XMP_GASNET_LOCKHANDOFF,                _xmp_gasnet_lockhandoff },
  { _XMP_GASNET_POST_REQUEST,               _xmp_gasnet_post_request },
  { _XMP_GASNET_UNPACK,                     _xmp_gasnet_unpack },
  { _XMP_GASNET_UNPACK_USING_BUF,           _xmp_gasnet_unpack_using_buf },
  { _XMP_GASNET_UNPACK_REPLY,               _xmp_gasnet_unpack_reply },
  { _XMP_GASNET_PACK,                       _xmp_gasnet_pack },
  { _XMP_GASNET_UNPACK_GET_REPLY,           _xmp_gasnet_unpack_get_reply},
  { _XMP_GASNET_PACK_USGIN_BUF,             _xmp_gasnet_pack_using_buf},
  { _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF, _xmp_gasnet_unpack_get_reply_using_buf},
  { _XMP_GASNET_PACK_GET_HANDLER,           _xmp_gasnet_pack_get },
  { _XMP_GASNET_UNPACK_GET_REPLY_NONC,      _xmp_gasnet_unpack_get_reply_nonc }
};

void _XMP_gasnet_malloc_do(_XMP_coarray_t *coarray, void **addr, unsigned long long coarray_size)
{
  int numprocs;
  char **each_addr;  // head address of a local array on each node

  numprocs = gasnet_nodes();
  each_addr = (char **)_XMP_alloc(sizeof(char *) * numprocs);

  for(int i=0;i<numprocs;i++)
    each_addr[i] = (char *)(_xmp_gasnet_buf[i]) + _xmp_coarray_shift;

  if(coarray_size % _XMP_GASNET_ALIGNMENT == 0)
    _xmp_coarray_shift += coarray_size;
  else{
    _xmp_coarray_shift += ((coarray_size / _XMP_GASNET_ALIGNMENT) + 1) * _XMP_GASNET_ALIGNMENT;
  }
    
  if(_xmp_coarray_shift > _xmp_heap_size){
    if(gasnet_mynode() == 0){
      fprintf(stderr, "Cannot allocate coarray. Now HEAP SIZE is %d MB\n", (int)(_xmp_heap_size/1024/1024));
      fprintf(stderr, "But %d MB is needed\n", (int)(_xmp_coarray_shift/1024/1024));
    }
    _XMP_fatal("Please set XMP_COARRAY_HEAP_SIZE=<number> (MB)\n");
  }

  coarray->addr = each_addr;
  *addr = each_addr[gasnet_mynode()];
}

void _XMP_gasnet_initialize(int argc, char **argv, unsigned long long xmp_heap_size, unsigned long long xmp_stride_size){
  int numprocs;

  if(argc != 0)
    gasnet_init(&argc, &argv);
  else{ 
    // In XMP/Fortran, this function is called with "argc == 0" & "**argv == NULL".
    // But if the second argument of gasnet_init() is NULL, gasnet_init() returns error.
    // So dummy argument is created and used.
    char **s;
    s = malloc(sizeof(char *));
    s[0] = malloc(sizeof(char));
    gasnet_init(&argc, &s);
  }

  if(xmp_heap_size % GASNET_PAGESIZE != 0)
    _xmp_heap_size = (xmp_heap_size/GASNET_PAGESIZE -1) * GASNET_PAGESIZE;
  else
    _xmp_heap_size = xmp_heap_size;

  _xmp_stride_size = xmp_stride_size;

  gasnet_attach(htable, sizeof(htable)/sizeof(gasnet_handlerentry_t), _xmp_heap_size, 0); 
  numprocs = gasnet_nodes();

  _xmp_gasnet_buf = (char **)malloc(sizeof(char*) * numprocs);

  gasnet_node_t i;
  gasnet_seginfo_t *s = (gasnet_seginfo_t *)malloc(gasnet_nodes()*sizeof(gasnet_seginfo_t)); 
  gasnet_getSegmentInfo(s, gasnet_nodes());
  for(i=0;i<numprocs;i++)
    _xmp_gasnet_buf[i] =  (char*)s[i].addr;

  _xmp_coarray_shift = xmp_stride_size;
  _xmp_gasnet_stride_queue = malloc(sizeof(int) * _XMP_GASNET_STRIDE_INIT_SIZE);
}

void _XMP_gasnet_finalize(int val){
  _XMP_gasnet_sync_all();
  gasnet_exit(val);
}

void _XMP_gasnet_sync_memory(){
  int i;

  for(i=0;i<_xmp_gasnet_stride_wait_size;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[i] == 1);

  _xmp_gasnet_stride_wait_size = 0;

  gasnet_wait_syncnbi_puts();
}

void _XMP_gasnet_sync_all(){
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

static void XMP_gasnet_from_c_to_c_put(const int target_image, const long long dst_point, 
				       const long long src_point, const _XMP_coarray_t *dst, 
				       const void *src, const long long transfer_size){

  gasnet_put_nbi_bulk(target_image, dst->addr[target_image]+dst_point, ((char *)src)+src_point, 
		      transfer_size);

}

static int is_all_elmt(const _XMP_array_section_t* array_info, const int dim){
  if(array_info[dim].start == 0 && array_info[dim].length == array_info[dim].elmts){
    return _XMP_N_INT_TRUE;
  }
  else{
    return _XMP_N_INT_FALSE;
  }
}

// How depth is memory continuity ?
// when depth is 0, all dimension is not continuous.
// ushiro no jigen kara kazoete "depth" banme made rennzokuka ?
// eg. a[:][2:2:1]    -> depth is 1. The last diemnsion is continuous.
//     a[:][2:2:2]    -> depth is 0.
//     a[:][:]        -> depth is 2. But, this function is called when array is not continuous. 
//                       So depth does not become 2.
//     b[:][:][1:2:2]   -> depth is 0.
//     b[:][:][1]       -> depth is 1.
//     b[:][2:2:2][1]   -> depth is 1.
//     b[:][2:2:2][:]   -> depth is 1.
//     b[2:2:2][:][:]   -> depth is 2.
//     b[2:2][2:2][2:2] -> depth is 1.
//     c[1:2][1:2][1:2][1:2] -> depth is 1.
//     c[1:2:2][:][:][:]     -> depth is 3. 
//     c[1:2:2][::2][:][:]   -> depth is 2.
static int get_depth(const int dims, const _XMP_array_section_t* array_info)  // 7 >= dims >= 2
{
  if(dims == 2){
    if(array_info[1].stride == 1)
      return 1;
    else
      return 0;
  }
  else if(dims == 3){
    if(is_all_elmt(array_info, 1) && is_all_elmt(array_info, 2)){
      return 2;
    }
    else if(array_info[2].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 4){
    if(is_all_elmt(array_info, 1) && is_all_elmt(array_info, 2) && 
       is_all_elmt(array_info, 3)){
      return 3;
    }
    else if(is_all_elmt(array_info, 2) && is_all_elmt(array_info, 3)){
      return 2;
    }
    else if(array_info[3].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 5){
    if(is_all_elmt(array_info, 1) && is_all_elmt(array_info, 2) &&
       is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4)){
      return 4;
    }
    else if(is_all_elmt(array_info, 2) && is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4)){
      return 3;
    }
    else if(is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4)){
      return 2;
    }
    else if(array_info[4].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 6){
    if(is_all_elmt(array_info, 1) && is_all_elmt(array_info, 2) &&
       is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4) &&
       is_all_elmt(array_info, 5)){
      return 5;
    }
    else if(is_all_elmt(array_info, 2) && is_all_elmt(array_info, 3) && 
	    is_all_elmt(array_info, 4) && is_all_elmt(array_info, 5)){
      return 4;
    }
    else if(is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4) &&
	    is_all_elmt(array_info, 5)){
      return 3;
    }
    else if(is_all_elmt(array_info, 4) && is_all_elmt(array_info, 5)){
      return 2;
    }
    else if(array_info[5].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else if(dims == 7){
    if(is_all_elmt(array_info, 1) && is_all_elmt(array_info, 2) &&
       is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4) &&
       is_all_elmt(array_info, 5) && is_all_elmt(array_info, 6)){
      return 6;
    }
    else if(is_all_elmt(array_info, 2) && is_all_elmt(array_info, 3) &&
            is_all_elmt(array_info, 4) && is_all_elmt(array_info, 5) &&
	    is_all_elmt(array_info, 6)){
      return 5;
    }
    else if(is_all_elmt(array_info, 3) && is_all_elmt(array_info, 4) &&
            is_all_elmt(array_info, 5) && is_all_elmt(array_info, 6)){
      return 4;
    }
    else if(is_all_elmt(array_info, 4) && is_all_elmt(array_info, 5) &&
	    is_all_elmt(array_info, 6)){
      return 3;
    }
    else if(is_all_elmt(array_info, 5) && is_all_elmt(array_info, 6)){
      return 2;
    }
    else if(array_info[6].stride == 1){
      return 1;
    }
    else{
      return 0;
    }
  }
  else{
    _XMP_fatal("Dimensions of Coarray is too big.");
    return -1;
  }
#ifdef _NOT_USED
  if(array_info[dims-1].stride != 1){
    return 0;
  }
  else if(is_all_elmt(array_info, dims-1) || array_info[dims-1].length == 1){
    return 1;
  }

  int i, j, flag;
  for(j=dims-1;j>=1;j--){
    flag = _XMP_N_INT_TRUE;
    for(i=j;i>=1;i--){
      if(!is_all_elmt(array_info, dims-i)){
	flag = _XMP_N_INT_FALSE;
	break;
      }
    }

    if(flag)
      return j;
  }
  
  if(array_info[dims-1].stride == 1)
    return 1;
  else
    return 0;
#endif
}

static void pack_for_7_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
                                 const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = src[6].distance;
  long long start_offset = 0, archive_offset = 0, src_offset;
  int tmp[7];
  long long stride_offset[7], length;

  for(int i=0;i<7;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 6){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 5){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 4){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 3){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
            archive_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[5].distance * src[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
              archive_offset += length;
            }
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[6].distance * src[6].length;
    for(int i=0;i<6;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<src[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		archive_offset += length;
	      }
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<7;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<src[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		for(int q=0;q<src[6].length;q++){
		  tmp[6] = stride_offset[6] * q;
		  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]);
		  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		  archive_offset += length;
		}
              }
            }
          }
        }
      }
    }
  }
}

static void pack_for_6_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
                                 const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = src[5].distance;
  long long start_offset = 0, archive_offset = 0, src_offset;
  int tmp[6];
  long long stride_offset[6], length;

  for(int i=0;i<6;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 5){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 4){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 3){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
            archive_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[5].distance * src[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	      archive_offset += length;
	    }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<6;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<src[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<src[5].length;p++){
		tmp[5] = stride_offset[5] * p;
		src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
		archive_offset += length;
	      }
            }
          }
        }
      }
    }
  }
}

static void pack_for_5_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
                                 const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = src[4].distance;
  long long start_offset = 0, archive_offset = 0, src_offset;
  int tmp[5];
  long long stride_offset[5], length;

  for(int i=0;i<5;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 4){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 3){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        src_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
        archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 2){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
          archive_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[4].distance * src[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	    archive_offset += length;
	  }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<5;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<src[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<src[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<src[4].length;n++){
	      tmp[4] = stride_offset[4] * n;
	      src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	      archive_offset += length;
	    }
          }
        }
      }
    }
  }
}

static void pack_for_4_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
				 const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = src[3].distance;
  long long start_offset = 0, archive_offset = 0, src_offset;
  int tmp[4];
  long long stride_offset[4], length;

  for(int i=0;i<4;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 3){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 2){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	src_offset = start_offset + (tmp[0] + tmp[1]);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	archive_offset += length;
      }
    }
  }
  else if(continuous_dim == 1){
    length = src[3].distance * src[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;
    
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	  archive_offset += length;
	}
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<4;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  for(int m=0;m<src[3].length;m++){
	    tmp[3] = stride_offset[3] * m;
	    src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	    archive_offset += length;
	  }
	}
      }
    }
  }
}

static void pack_for_3_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
				 const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = src[2].distance;
  long long start_offset = 0, archive_offset = 0, src_offset;
  int tmp[3];
  long long stride_offset[3], length;

  for(int i=0;i<3;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 2){
    length = src[1].length * src[1].distance;
    stride_offset[0] = src[0].stride * src[0].distance;
    
    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      src_offset = start_offset + tmp[0];
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else if(continuous_dim == 1){
    length = src[2].distance * src[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	src_offset = start_offset + (tmp[0] + tmp[1]);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	archive_offset += length;
      }
    }
  }
  else{ // continuous_dim == 0
    length = element_size;
    for(int i=0;i<3;i++)
      stride_offset[i] = src[i].stride * src[i].distance;

    for(int i=0;i<src[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	tmp[1] = stride_offset[1] * j;
	for(int k=0;k<src[2].length;k++){
	  tmp[2] = stride_offset[2] * k;
	  src_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
	  archive_offset += length;
	}
      }
    }
  }
}

static void pack_for_2_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr, 
				 const int continuous_dim){  // continuous_dim is 0 or 1

  size_t element_size = src[1].distance;
  long long start_offset = 0;
  long long archive_offset = 0, src_offset;
  for(int i=0;i<2;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 1){
    int length = element_size * src[1].length;
    long long stride_offset = (src[0].stride * src[1].elmts) * element_size;
    for(int i=0;i<src[0].length;i++){
      src_offset = start_offset + stride_offset * i;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else{ // continuous_dim == 0
    long long stride_offset[2];
    stride_offset[0] = src[0].stride * src[1].elmts * element_size;
    stride_offset[1] = src[1].stride * element_size;
    for(int i=0;i<src[0].length;i++){
      long long tmp = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	src_offset = start_offset + (tmp + stride_offset[1] * j);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
	archive_offset += element_size;
      }
    }
  }
}

static void pack_for_1_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr){
  // for(i=0;i<src[0].length;i++){
  //   src_offset = start_offset + (stride_offset * i);
  //   memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
  //   archive_offset += element_size;
  // }
  size_t element_size = src[0].distance;
  int repeat = src[0].length / UNROLLING;
  int left   = src[0].length % UNROLLING;
  long long start_offset  = src[0].start  * element_size;
  long long stride_offset = src[0].stride * element_size;
  long long archive_offset = 0, src_offset;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      src_offset = start_offset + (stride_offset * i);
      archive_offset = i * element_size;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      src_offset += stride_offset;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;

      i += UNROLLING;
    }

    switch (left) {
    case 3 :
      src_offset = start_offset + (stride_offset * (i+2));
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;
    case 2 :
      src_offset = start_offset + (stride_offset * (i+1));
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
      archive_offset += element_size;
    case 1 :
      src_offset = start_offset + (stride_offset * i);
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
    }
  }
}

static void XMP_pack(char* archive_ptr, const char* src_ptr, const int src_dims, 
		     const _XMP_array_section_t* src)
{
  if(src_dims == 1){ 
    pack_for_1_dim_array(src, archive_ptr, src_ptr);
    return;
  }

  // How depth is memory continuity ?
  int continuous_dim = get_depth(src_dims, src);

  if(src_dims == 2){
    pack_for_2_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 3){
    pack_for_3_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 4){
    pack_for_4_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 5){
    pack_for_5_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 6){
    pack_for_6_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 7){
    pack_for_7_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else{
    _XMP_fatal("Dimension of coarray is too big");
    return;
  }

#ifdef _NOT_USED
  size_t element_size = src[src_dims-1].distance;
  int index[src_dims+1], d = 1;                  // d is a position of nested loop
  for(int i=0;i<src_dims+1;i++) index[i] = 0;    // Initialize index
  long long cnt[src_dims], src_offset;
  long long archive_offset = 0;
  cnt[0] = 0;

  if(src[src_dims-1].stride != 1 || continuous_dim+1 == src_dims){
    while(index[0]==0){
      if(index[d]>=src[d-1].length){    // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < src_dims){                 // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*src[d-1].stride+src[d-1].start) * src[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == src_dims){                // the innermost loop
        src_offset = cnt[d-1] + (index[d]*src[d-1].stride+src[d-1].start) * src[d-1].distance;
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
        archive_offset += element_size;
        index[d]++;
      }
    }
  }
  else{
    while(index[0]==0){
      if(index[d]>=src[d-1].length){    // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < continuous_dim+1){         // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*src[d-1].stride+src[d-1].start) * src[d-1].distance;
        index[d+1] = 0;
        d++;
      }

      else if(d == continuous_dim+1){        // the innermost loop
        src_offset = cnt[d-1] + (index[d]*src[d-1].stride+src[d-1].start) * src[d-1].distance;
        memcpy(archive_ptr + archive_offset, src_ptr + src_offset + (src[d].start * element_size),
               src[d].length * src[d+1].distance);
        archive_offset += src[d].length * src[d].distance;
        index[d]++;
      }
    }
  }
#endif
}

static void XMP_gasnet_from_nonc_to_c_put(int target_image, long long dst_point, int src_dims, 
					  _XMP_array_section_t *src, _XMP_coarray_t *dst, void *src_ptr, 
					  long long transfer_size){
  char archive[transfer_size];
  XMP_pack(archive, src_ptr, src_dims, src);
  XMP_gasnet_from_c_to_c_put(target_image, dst_point, (long long)0, dst, archive, transfer_size);
}

void _xmp_gasnet_unpack_reply(gasnet_token_t t, const int ith){
  _xmp_gasnet_stride_queue[ith] = 1;
}

static void extend_stride_queue(){
  if(_xmp_gasnet_stride_wait_size >= _xmp_gasnet_stride_queue_size){
    int old_size = _xmp_gasnet_stride_wait_size;
    int new_size = old_size + _XMP_GASNET_STRIDE_BLK;
    int *new_list = malloc(sizeof(int) * new_size);
    int *old_list = _xmp_gasnet_stride_queue;
    memcpy(new_list, old_list, sizeof(int) * old_size);
    _xmp_gasnet_stride_queue = new_list;
    _xmp_gasnet_stride_queue_size = new_size;
    free(old_list);
  }
}

static void unpack_for_7_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                   char* dst_ptr, const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = dst[6].distance;
  long long start_offset = 0, src_offset = 0, dst_offset;
  int tmp[7];
  long long stride_offset[7], length;
  for(int i=0;i<7;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 6){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 5){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 4){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
              memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
              src_offset += length;
            }
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[6].distance * dst[6].length;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		src_offset += length;
	      }
            }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<7;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              for(int p=0;p<dst[5].length;p++){
                tmp[5] = stride_offset[5] * p;
		for(int q=0;q<dst[6].length;q++){
		  tmp[6] = stride_offset[6] * q;
		  dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]);
		  memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		  src_offset += length;
		}
              }
            }
          }
        }
      }
    }
  }
}

static void unpack_for_6_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                   char* dst_ptr, const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = dst[5].distance;
  long long start_offset = 0, src_offset = 0, dst_offset;
  int tmp[6];
  long long stride_offset[6], length;
  for(int i=0;i<6;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 5){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 4){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 3){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
            memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[5].distance * dst[5].length;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
              dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	      src_offset += length;
	    }
          }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<6;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            for(int n=0;n<dst[4].length;n++){
              tmp[4] = stride_offset[4] * n;
	      for(int p=0;p<dst[5].length;p++){
		tmp[5] = stride_offset[5] * p;
		dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5]);
		memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
		src_offset += length;
	      }
            }
          }
        }
      }
    }
  }
}

static void unpack_for_5_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                   char* dst_ptr, const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = dst[4].distance;
  long long start_offset = 0, src_offset = 0, dst_offset;
  int tmp[5];
  long long stride_offset[5], length;
  for(int i=0;i<5;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 4){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 3){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 2){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[4].distance * dst[4].length;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	    src_offset += length;
	  }
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<5;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
	    for(int n=0;n<dst[4].length;n++){
	      tmp[4] = stride_offset[4] * n;
	      dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4]);
	      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
	      src_offset += length;
	    }
          }
        }
      }
    }
  }
}

static void unpack_for_4_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, 
				   char* dst_ptr, const int continuous_dim)  // continuous_dim is from 0 to 3
{
  size_t element_size = dst[3].distance;
  long long start_offset = 0, src_offset = 0, dst_offset;
  int tmp[4];
  long long stride_offset[4], length;
  for(int i=0;i<4;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 3){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 2){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else if(continuous_dim == 1){
    length = dst[3].distance * dst[3].length;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
	  memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
  else{  // continuous_dim == 0
    length = element_size;
    for(int i=0;i<4;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          for(int m=0;m<dst[3].length;m++){
            tmp[3] = stride_offset[3] * m;
            dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
	    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
            src_offset += length;
          }
        }
      }
    }
  }
}

static void unpack_for_3_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				   char* dst_ptr, const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = dst[2].distance;
  long long start_offset = 0, src_offset = 0, dst_offset;
  int tmp[3];
  long long stride_offset[3], length;
  for(int i=0;i<3;i++)
    start_offset += dst[i].start * dst[i].distance;

  if(continuous_dim == 2){
    length = dst[1].length * dst[1].distance;
    stride_offset[0] = dst[0].stride * dst[0].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      dst_offset = start_offset + tmp[0];
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else if(continuous_dim == 1){
    length = dst[2].distance * dst[2].length;
    for(int i=0;i<2;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        dst_offset = start_offset + (tmp[0] + tmp[1]);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
        src_offset += length;
      }
    }
  }
  else{ // continuous_dim == 0
    length = element_size;
    for(int i=0;i<3;i++)
      stride_offset[i] = dst[i].stride * dst[i].distance;
    for(int i=0;i<dst[0].length;i++){
      tmp[0] = stride_offset[0] * i;
      for(int j=0;j<dst[1].length;j++){
        tmp[1] = stride_offset[1] * j;
        for(int k=0;k<dst[2].length;k++){
          tmp[2] = stride_offset[2] * k;
          dst_offset = start_offset + (tmp[0] + tmp[1] + tmp[2]);
          memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
          src_offset += length;
        }
      }
    }
  }
}

static void unpack_for_2_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
                                   char* dst_ptr, const int continuous_dim){
  // continuous_dim is 0 or 1
  size_t element_size = dst[1].distance;
  long long start_offset  = (dst[0].start * dst[1].elmts + dst[1].start) * element_size;
  long long dst_offset, src_offset = 0;
  int i;

  if(continuous_dim == 1){
    int length = element_size * dst[1].length;
    long long stride_offset = (dst[0].stride * dst[1].elmts) * element_size;
    for(i=0;i<dst[0].length;i++){
      dst_offset = start_offset + stride_offset * i;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else{ // continuous_dim == 0
    int j;
    long long stride_offset[2];
    stride_offset[0] = dst[0].stride * dst[1].elmts * element_size;
    stride_offset[1] = dst[1].stride * element_size;
    for(i=0;i<dst[0].length;i++){
      long long tmp = stride_offset[0] * i;
      for(j=0;j<dst[1].length;j++){
        dst_offset = start_offset + (tmp + stride_offset[1] * j);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
        src_offset += element_size;
      }
    }
  }
}

static void unpack_for_1_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, char* dst_ptr){
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //    src_offset += element_size;
  //  }
  size_t element_size = dst[0].distance;
  int repeat = dst[0].length / UNROLLING;
  int left   = dst[0].length % UNROLLING;
  long long start_offset  = dst[0].start  * element_size;
  long long stride_offset = dst[0].stride * element_size;
  long long dst_offset, src_offset = 0;
  int i = 0;

  if(repeat == 0){
    for(i=0;i<left;i++){
      dst_offset = start_offset + (i * stride_offset);
      src_offset = i * element_size;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
    }
  }
  else{
    while( repeat-- > 0 ) {
      dst_offset = start_offset + (i * stride_offset);
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      dst_offset += stride_offset;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;

      i += UNROLLING;
    }

    switch (left) {
    case 3 :
      dst_offset = start_offset + (stride_offset * (i+2));
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;
    case 2 :
      dst_offset = start_offset + (stride_offset * (i+1));
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
      src_offset += element_size;
    case 1:
      dst_offset = start_offset + (stride_offset * i);
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
    }
  }
}

static void XMP_unpack(char *dst_ptr, const int dst_dims, const char* src_ptr, 
		       _XMP_array_section_t* dst){
  if(dst_dims == 1){
    unpack_for_1_dim_array(dst, src_ptr, dst_ptr);
    return;
  }

  int continuous_dim = get_depth(dst_dims, dst);

  if(dst_dims == 2){
    unpack_for_2_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 3){
    unpack_for_3_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 4){
    unpack_for_4_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 5){
    unpack_for_5_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 6){
    unpack_for_6_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 7){
    unpack_for_7_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else{
    _XMP_fatal("Dimension of coarray is too big.");
    return;
  }

#ifdef _NOT_USED
  unsigned long long dst_offset = 0, src_offset = 0;
  size_t element_size = dst[dst_dims-1].distance;
  int index[dst_dims+1], d = 1, i;                // d is a position of nested loop
  for(i=0;i<dst_dims+1;i++)   index[i] = 0;       // Initialize index
  unsigned long long cnt[dst_dims];
  cnt[0] = 0;

  if(dst[dst_dims-1].stride != 1 || continuous_dim+1 == dst_dims){
    while(index[0]==0){
      if(index[d]>=dst[d-1].length){          // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < dst_dims){                  // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == dst_dims){                 // the innermost loop
        dst_offset = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
        src_offset += element_size;
        index[d]++;
      }
    }
  }
  else{
    while(index[0]==0){
      if(index[d]>=dst[d-1].length){           // Move to outer loop
        d--;
        index[d]++;
      }
      else if(d < continuous_dim+1){           // Move to inner loop
        cnt[d] = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        index[d+1] = 0;
        d++;
      }
      else if(d == continuous_dim+1){          // the innermost loop
        dst_offset = cnt[d-1] + (index[d]*dst[d-1].stride+dst[d-1].start) * dst[d-1].distance;
        memcpy(dst_ptr + dst_offset + (dst[d].start * element_size), src_ptr + src_offset,
               dst[d].length * dst[d].distance);
        src_offset += dst[d].length * dst[d].distance;
        index[d]++;
      }
    }
  }
#endif
}

void _xmp_gasnet_unpack_using_buf(gasnet_token_t t, const int addr_hi, const int addr_lo, 
				  const int dst_dims, const int ith){

  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  char* src_addr = _xmp_gasnet_buf[gasnet_mynode()];
  memcpy(dst, src_addr, dst_info_size);
  XMP_unpack((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

void _xmp_gasnet_unpack(gasnet_token_t t, const char* src_addr, const size_t nbytes, 
			const int addr_hi, const int addr_lo, const int dst_dims, const int ith){

  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  memcpy(dst, src_addr, dst_info_size);
  XMP_unpack((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

static void coarray_stride_size_error(){
  fprintf(stderr, "Corray stride transfer size is too big\n");
  fprintf(stderr, "Reconfigure environmental variant BUFFER_FOR_STRIDE_SIZE > %lld\n", _xmp_stride_size);
  _XMP_fatal("");
}

static void XMP_gasnet_from_c_to_nonc_put(int target_image, long long src_point, int dst_dims, 
					  _XMP_array_section_t *dst_info, 
                                          _XMP_coarray_t *dst, void *src, long long transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += (long long)dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, (char *)src+src_point, transfer_size - dst_info_size);

  extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK, archive, (size_t)transfer_size,
			    HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims, 
			    _xmp_gasnet_stride_wait_size);
  }
  else if(transfer_size < _xmp_stride_size){
    gasnet_put(target_image, _xmp_gasnet_buf[target_image], archive, (size_t)transfer_size);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target_image]), 
			   LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    coarray_stride_size_error();
  }
  _xmp_gasnet_stride_wait_size++;
}

static void XMP_gasnet_from_nonc_to_nonc_put(int target_image, int dst_dims, int src_dims,
					     _XMP_array_section_t *dst_info, _XMP_array_section_t *src,
					     _XMP_coarray_t *dst, void *src_ptr, long long transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  size_t tsize = (size_t)transfer_size + dst_info_size;
  char archive[tsize];
  memcpy(archive, dst_info, dst_info_size);
  XMP_pack(archive + dst_info_size, src_ptr, src_dims, src);
  extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;

  if(tsize < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK, archive, tsize,
    			    HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims,
    			    _xmp_gasnet_stride_wait_size);
  }
  else if(tsize < _xmp_stride_size){
    gasnet_put_bulk(target_image, _xmp_gasnet_buf[target_image], archive, tsize);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target_image]),
                           LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    coarray_stride_size_error();
  }

  GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] == 1);
}

void _XMP_gasnet_put(int dst_continuous, int src_continuous, int target_image, int dst_dims, 
		     int src_dims, _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
		     _XMP_coarray_t *dst, void *src, long long length){

  long long transfer_size = dst->elmt_size*length;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    long long dst_point = get_offset(dst_info, dst_dims);
    long long src_point = get_offset(src_info, src_dims);
    XMP_gasnet_from_c_to_c_put(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    long long dst_point = get_offset(dst_info, dst_dims);
    XMP_gasnet_from_nonc_to_c_put(target_image, dst_point, src_dims, src_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    long long src_point = get_offset(src_info, src_dims);
    XMP_gasnet_from_c_to_nonc_put(target_image, src_point, dst_dims, dst_info, 
				  dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
    XMP_gasnet_from_nonc_to_nonc_put(target_image, dst_dims, src_dims, dst_info,
    				     src_info, dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Unkown shape of coarray");
  }
}

static void XMP_gasnet_from_c_to_c_get(const int target_image, const long long dst_point, 
				       const long long src_point, const void *dst, 
				       const _XMP_coarray_t *src, const long long transfer_size){

  gasnet_get_bulk(((char *)dst)+dst_point, target_image, ((char *)src->addr[target_image])+src_point,
		  transfer_size);

}

static void XMP_gasnet_from_c_to_nonc_get(int target_image, long long src_point, int dst_dims, 
					  _XMP_array_section_t *dst_info, 
					  void *dst, _XMP_coarray_t *src, long long transfer_size){
  if(transfer_size < _xmp_stride_size){
    char* src_addr = (char *)_xmp_gasnet_buf[gasnet_mynode()];
    gasnet_get_bulk(src_addr, target_image, ((char *)src->addr[target_image])+src_point, (size_t)transfer_size);
    XMP_unpack(((char *)dst), dst_dims, src_addr, dst_info);
  }
  else{
    coarray_stride_size_error();
  }
}

void _xmp_gasnet_pack(gasnet_token_t t, const char* info, const size_t am_request_size, 
		      const int src_addr_hi, const int src_addr_lo, const int src_dims, 
		      const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo){
  
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[gasnet_mynode()];
  XMP_pack(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyMedium2(t, _XMP_GASNET_UNPACK_GET_REPLY, archive, tansfer_size,
      			dst_addr_hi, dst_addr_lo);
}

void _xmp_gasnet_pack_get(gasnet_token_t t, const char* info, const size_t am_request_size,
			  const int src_addr_hi, const int src_addr_lo, const int src_dims, const int dst_dims,
			  const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo){

  size_t src_size = sizeof(_XMP_array_section_t) * src_dims;
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *src_info = malloc(src_size);
  memcpy(src_info, info, src_size);
  char archive[tansfer_size + dst_size];
  memcpy(archive, info + src_size, dst_size);
  XMP_pack(archive+dst_size, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  free(src_info);
  gasnet_AMReplyMedium3(t, _XMP_GASNET_UNPACK_GET_REPLY_NONC, archive, tansfer_size + dst_size,
                        dst_addr_hi, dst_addr_lo, dst_dims);
}

void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t t, char *archive, size_t transfer_size,
				       const int dst_addr_hi, const int dst_addr_lo, const int dst_dims){
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst_info = malloc(dst_size);
  memcpy(dst_info, archive, dst_size);

  XMP_unpack((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), dst_dims, archive+dst_size, dst_info);
  done_get_flag = _XMP_N_INT_TRUE;
}


void _xmp_gasnet_unpack_get_reply(gasnet_token_t t, char *archive, size_t transfer_size, 
				  const int dst_addr_hi, const int dst_addr_lo){
  memcpy((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), archive, transfer_size);
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t t){
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_pack_using_buf(gasnet_token_t t, const char* info, const size_t am_request_size,
				const int src_addr_hi, const int src_addr_lo, const int src_dims,
				const int target_image){

  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[gasnet_mynode()];
  XMP_pack(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyShort0(t, _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF);
}

static void XMP_gasnet_from_nonc_to_c_get(int target_image, int src_dims, _XMP_array_section_t *src_info, 
					  void *dst, _XMP_coarray_t *src, 
					  long long transfer_size, long long dst_point){
  size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
  char archive[am_request_size];  // Note: Info. of transfer_size may have better in "archive".
  memcpy(archive, src_info, am_request_size);

  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // mikansei
    gasnet_AMRequestMedium6(target_image, _XMP_GASNET_PACK, archive, am_request_size,
			    HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims,
    			    (size_t)transfer_size, HIWORD((char *)dst+dst_point), LOWORD((char *)dst+dst_point));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
  }
  else if(transfer_size < _xmp_stride_size){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims,
                            gasnet_mynode());
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[gasnet_mynode()], target_image, _xmp_gasnet_buf[target_image], transfer_size);
    memcpy(((char *)dst)+dst_point, _xmp_gasnet_buf[gasnet_mynode()], transfer_size);
  }
  else{
    coarray_stride_size_error();
  }
}

static void XMP_gasnet_from_nonc_to_nonc_get(int target_image, int dst_dims, int src_dims, 
					     _XMP_array_section_t *dst_info, _XMP_array_section_t *src_info, 
					     void *dst, _XMP_coarray_t *src, long long transfer_size)
{
  done_get_flag = _XMP_N_INT_FALSE;
  //  if(transfer_size < gasnet_AMMaxMedium()){
  if(transfer_size < 0){  // mikansei
    size_t am_request_src_size = sizeof(_XMP_array_section_t) * src_dims;
    size_t am_request_dst_size = sizeof(_XMP_array_section_t) * dst_dims;
    char *archive = malloc(am_request_src_size + am_request_dst_size);
    memcpy(archive, src_info, am_request_src_size);
    memcpy(archive + am_request_src_size, dst_info, am_request_dst_size);
    gasnet_AMRequestMedium7(target_image, _XMP_GASNET_PACK_GET_HANDLER, archive, 
			    am_request_src_size+am_request_dst_size,
                            HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims, dst_dims,
                            (size_t)transfer_size, HIWORD((char *)dst), LOWORD((char *)dst));
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    free(archive);
  }
  else if(transfer_size < _xmp_stride_size){
    size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
    char *archive = malloc(am_request_size);
    memcpy(archive, src_info, am_request_size);
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims,
                            gasnet_mynode());
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[gasnet_mynode()], target_image, _xmp_gasnet_buf[target_image], 
		    transfer_size);
    XMP_unpack((char *)dst, dst_dims, _xmp_gasnet_buf[gasnet_mynode()], dst_info);
    free(archive);
  }
  else{
    coarray_stride_size_error();
  }
}

void _XMP_gasnet_get(int src_continuous, int dst_continuous, int target_image, int src_dims, 
		     int dst_dims, _XMP_array_section_t *src_info, _XMP_array_section_t *dst_info, 
		     _XMP_coarray_t *src, void *dst, long long length){

  long long transfer_size = src->elmt_size*length;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    long long dst_point = get_offset(dst_info, dst_dims);
    long long src_point = get_offset(src_info, src_dims);
    XMP_gasnet_from_c_to_c_get(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    long long dst_point = get_offset(dst_info, dst_dims);
    XMP_gasnet_from_nonc_to_c_get(target_image, src_dims, src_info, 
				  dst, src, transfer_size, dst_point);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    long long src_point = get_offset(src_info, src_dims);
    XMP_gasnet_from_c_to_nonc_get(target_image, src_point, dst_dims, dst_info, 
				  dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
    XMP_gasnet_from_nonc_to_nonc_get(target_image, dst_dims, src_dims, dst_info, src_info, 
    				     dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Unkown shape of coarray");
  }
}

