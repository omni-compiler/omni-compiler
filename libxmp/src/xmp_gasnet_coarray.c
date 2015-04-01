#include "xmp_internal.h"
static int _xmp_gasnet_stride_wait_size = 0;
static int _xmp_gasnet_stride_queue_size = _XMP_GASNET_STRIDE_INIT_SIZE;
volatile static int done_get_flag;
extern char ** _xmp_gasnet_buf;
extern int *_xmp_gasnet_stride_queue;
extern size_t _xmp_gasnet_coarray_shift, _xmp_gasnet_stride_size, _xmp_gasnet_heap_size;
#define UNROLLING (4)
struct _shift_queue_t{
  unsigned int max_size;   /**< Max size of queue */
  unsigned int      num;   /**< How many shifts are in this queue */
  size_t        *shifts;   /**< shifts array */
};
static struct _shift_queue_t _shift_queue;

void _XMP_gasnet_build_shift_queue()
{
  _shift_queue.max_size = _XMP_GASNET_COARRAY_SHIFT_QUEUE_INITIAL_SIZE;
  _shift_queue.num      = 0;
  _shift_queue.shifts   = malloc(sizeof(size_t*) * _shift_queue.max_size);
}

static void _rebuild_shift_queue()
{
  _shift_queue.max_size *= _XMP_GASNET_COARRAY_SHIFT_QUEUE_INCREMENT_RAITO;
  size_t *tmp;
  size_t next_size = _shift_queue.max_size * sizeof(size_t*);
  if((tmp = realloc(_shift_queue.shifts, next_size)) == NULL)
    _XMP_fatal("cannot allocate memory");
  else
    _shift_queue.shifts = tmp;
}

void _XMP_push_shift_queue(size_t s)
{
  if(_shift_queue.num >= _shift_queue.max_size)
    _rebuild_shift_queue();

  _shift_queue.shifts[_shift_queue.num++] = s;
}

static size_t _pop_shift_queue()
{
  if(_shift_queue.num == 0)  return 0;

  _shift_queue.num--;
  return _shift_queue.shifts[_shift_queue.num];
}

void _XMP_gasnet_coarray_lastly_deallocate(){
  _xmp_gasnet_coarray_shift -= _pop_shift_queue();
}

void _XMP_gasnet_malloc_do(_XMP_coarray_t *coarray, void **addr, const size_t coarray_size)
{
  char **each_addr;  // head address of a local array on each node
  size_t tmp_shift;

  each_addr = (char **)_XMP_alloc(sizeof(char *) * _XMP_world_size);

  for(int i=0;i<_XMP_world_size;i++) {
    each_addr[i] = (char *)(_xmp_gasnet_buf[i]) + _xmp_gasnet_coarray_shift;
  }

  if(coarray_size % _XMP_GASNET_ALIGNMENT == 0)
    tmp_shift = coarray_size;
  else{
    tmp_shift = ((coarray_size / _XMP_GASNET_ALIGNMENT) + 1) * _XMP_GASNET_ALIGNMENT;
  }
  _xmp_gasnet_coarray_shift += tmp_shift;
  _XMP_push_shift_queue(tmp_shift);

  if(_xmp_gasnet_coarray_shift > _xmp_gasnet_heap_size){
    if(_XMP_world_rank == 0){
      fprintf(stderr, "[ERROR] Cannot allocate coarray. Heap memory size of corray is too small.\n");
      fprintf(stderr, "        Please set the environmental variable \"XMP_COARRAY_HEAP_SIZE\".\n");
      fprintf(stderr, "        e.g.) export XMP_COARRAY_HEAP_SIZE=%zuM (or more).\n", (_xmp_gasnet_coarray_shift/1024/1024)+1);
    }
    _XMP_fatal_nomsg();
  }

  coarray->addr = each_addr;
  coarray->real_addr = each_addr[_XMP_world_rank];
  *addr = each_addr[_XMP_world_rank];
}
void _XMP_gasnet_sync_memory()
{
  for(int i=0;i<_xmp_gasnet_stride_wait_size;i++)
    GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[i] == 1);

  _xmp_gasnet_stride_wait_size = 0;

  gasnet_wait_syncnbi_puts();
}

void _XMP_gasnet_sync_all()
{
  _XMP_gasnet_sync_memory();
  GASNET_BARRIER();
}

static void _gasnet_c_to_c_put(const int target_image, const size_t dst_point, 
			       const size_t src_point, const _XMP_coarray_t *dst, 
			       const void *src, const size_t transfer_size)
{
  gasnet_put_nbi_bulk(target_image, dst->addr[target_image]+dst_point, ((char *)src)+src_point, 
		      transfer_size);

}

static int _is_all_elmt(const _XMP_array_section_t* array_info, const int dim)
{
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
static int _get_depth(const int dims, const _XMP_array_section_t* array_info)  // 7 >= dims >= 2
{
  if(dims == 2){
    if(array_info[1].stride == 1)
      return 1;
    else
      return 0;
  }
  else if(dims == 3){
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2)){
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
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) && 
       _is_all_elmt(array_info, 3)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3)){
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
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4)){
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
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
       _is_all_elmt(array_info, 5)){
      return 5;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) && 
	    _is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
	    _is_all_elmt(array_info, 5)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5)){
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
    if(_is_all_elmt(array_info, 1) && _is_all_elmt(array_info, 2) &&
       _is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
       _is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
      return 6;
    }
    else if(_is_all_elmt(array_info, 2) && _is_all_elmt(array_info, 3) &&
            _is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5) &&
	    _is_all_elmt(array_info, 6)){
      return 5;
    }
    else if(_is_all_elmt(array_info, 3) && _is_all_elmt(array_info, 4) &&
            _is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
      return 4;
    }
    else if(_is_all_elmt(array_info, 4) && _is_all_elmt(array_info, 5) &&
	    _is_all_elmt(array_info, 6)){
      return 3;
    }
    else if(_is_all_elmt(array_info, 5) && _is_all_elmt(array_info, 6)){
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
}

static void _pack_7_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[6].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[7];
  size_t stride_offset[7], length;

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

static void _pack_6_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			     const int continuous_dim)
{
  size_t element_size = src[5].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[6];
  size_t stride_offset[6], length;

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

static void _pack_5_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[4].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[5];
  size_t stride_offset[5], length;

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

static void _pack_4_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)
{
  size_t element_size = src[3].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[4];
  size_t stride_offset[4], length;

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

static void _pack_3_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr,
			      const int continuous_dim)  // continuous_dim is 0 or 1 or 2
{
  size_t element_size = src[2].distance;
  size_t start_offset = 0, archive_offset = 0, src_offset;
  int tmp[3];
  size_t stride_offset[3], length;

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

static void _pack_2_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr, 
			      const int continuous_dim) // continuous_dim is 0 or 1
{
  size_t element_size = src[1].distance;
  size_t start_offset = 0;
  size_t archive_offset = 0, src_offset;

  for(int i=0;i<2;i++)
    start_offset += src[i].start * src[i].distance;

  if(continuous_dim == 1){
    int length = element_size * src[1].length;
    size_t stride_offset = (src[0].stride * src[1].elmts) * element_size;
    for(int i=0;i<src[0].length;i++){
      src_offset = start_offset + stride_offset * i;
      memcpy(archive_ptr + archive_offset, src_ptr + src_offset, length);
      archive_offset += length;
    }
  }
  else{ // continuous_dim == 0
    size_t stride_offset[2];
    stride_offset[0] = src[0].stride * src[1].elmts * element_size;
    stride_offset[1] = src[1].stride * element_size;
    for(int i=0;i<src[0].length;i++){
      size_t tmp = stride_offset[0] * i;
      for(int j=0;j<src[1].length;j++){
	src_offset = start_offset + (tmp + stride_offset[1] * j);
	memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
	archive_offset += element_size;
      }
    }
  }
}

static void _pack_1_dim_array(const _XMP_array_section_t* src, char* archive_ptr, const char* src_ptr)
{
  // for(i=0;i<src[0].length;i++){
  //   src_offset = start_offset + (stride_offset * i);
  //   memcpy(archive_ptr + archive_offset, src_ptr + src_offset, element_size);
  //   archive_offset += element_size;
  // }
  size_t element_size = src[0].distance;
  int repeat = src[0].length / UNROLLING;
  int left   = src[0].length % UNROLLING;
  size_t start_offset  = src[0].start  * element_size;
  size_t stride_offset = src[0].stride * element_size;
  size_t archive_offset = 0, src_offset;
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

static void _pack_array(char* archive_ptr, const char* src_ptr, const int src_dims, 
			const _XMP_array_section_t* src)
{
  if(src_dims == 1){ 
    _pack_1_dim_array(src, archive_ptr, src_ptr);
    return;
  }

  // How depth is memory continuity ?
  int continuous_dim = _get_depth(src_dims, src);

  if(src_dims == 2){
    _pack_2_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 3){
    _pack_3_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 4){
    _pack_4_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 5){
    _pack_5_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 6){
    _pack_6_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else if(src_dims == 7){
    _pack_7_dim_array(src, archive_ptr, src_ptr, continuous_dim);
    return;
  }
  else{
    _XMP_fatal("Dimension of coarray is too big");
    return;
  }
}

static void _gasnet_nonc_to_c_put(const int target_image, const size_t dst_point, const int src_dims, 
				  const _XMP_array_section_t *src, const _XMP_coarray_t *dst, const void *src_ptr, 
				  const size_t transfer_size)
{
  char archive[transfer_size];
  _pack_array(archive, src_ptr, src_dims, src);
  _gasnet_c_to_c_put(target_image, dst_point, 0, dst, archive, transfer_size);
}

void _xmp_gasnet_unpack_reply(gasnet_token_t t, const int ith)
{
  _xmp_gasnet_stride_queue[ith] = 1;
}

static void _extend_stride_queue()
{
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

static void _unpack_7_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[6].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[7];
  size_t stride_offset[7], length;

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

static void _unpack_6_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
			       char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[5].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[6];
  size_t stride_offset[6], length;

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

static void _unpack_5_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
			       char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[4].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[5];
  size_t stride_offset[5], length;

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

static void _unpack_4_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, 
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[3].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[4];
  size_t stride_offset[4], length;

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

static void _unpack_3_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  size_t element_size = dst[2].distance;
  size_t start_offset = 0, src_offset = 0, dst_offset;
  int tmp[3];
  size_t stride_offset[3], length;

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

static void _unpack_2_dim_array(const _XMP_array_section_t* dst, const char* src_ptr,
				char* dst_ptr, const int continuous_dim)
{
  // continuous_dim is 0 or 1
  size_t element_size = dst[1].distance;
  size_t start_offset  = (dst[0].start * dst[1].elmts + dst[1].start) * element_size;
  size_t dst_offset, src_offset = 0;
  int i;

  if(continuous_dim == 1){
    int length = element_size * dst[1].length;
    size_t stride_offset = (dst[0].stride * dst[1].elmts) * element_size;
    for(i=0;i<dst[0].length;i++){
      dst_offset = start_offset + stride_offset * i;
      memcpy(dst_ptr + dst_offset, src_ptr + src_offset, length);
      src_offset += length;
    }
  }
  else{ // continuous_dim == 0
    int j;
    size_t stride_offset[2];
    stride_offset[0] = dst[0].stride * dst[1].elmts * element_size;
    stride_offset[1] = dst[1].stride * element_size;
    for(i=0;i<dst[0].length;i++){
      size_t tmp = stride_offset[0] * i;
      for(j=0;j<dst[1].length;j++){
        dst_offset = start_offset + (tmp + stride_offset[1] * j);
        memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
        src_offset += element_size;
      }
    }
  }
}

static void _unpack_1_dim_array(const _XMP_array_section_t* dst, const char* src_ptr, char* dst_ptr)
{
  //  for(i=0;i<dst[0].length;i++){
  //    dst_offset = start_offset + i * stride_offset;
  //    memcpy(dst_ptr + dst_offset, src_ptr + src_offset, element_size);
  //    src_offset += element_size;
  //  }
  size_t element_size = dst[0].distance;
  int repeat = dst[0].length / UNROLLING;
  int left   = dst[0].length % UNROLLING;
  size_t start_offset  = dst[0].start  * element_size;
  size_t stride_offset = dst[0].stride * element_size;
  size_t dst_offset, src_offset = 0;
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

static void _unpack_array(char *dst_ptr, const int dst_dims, const char* src_ptr, 
			  const _XMP_array_section_t* dst)
{
  if(dst_dims == 1){
    _unpack_1_dim_array(dst, src_ptr, dst_ptr);
    return;
  }

  int continuous_dim = _get_depth(dst_dims, dst);

  if(dst_dims == 2){
    _unpack_2_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 3){
    _unpack_3_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 4){
    _unpack_4_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 5){
    _unpack_5_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 6){
    _unpack_6_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else if(dst_dims == 7){
    _unpack_7_dim_array(dst, src_ptr, dst_ptr, continuous_dim);
    return;
  }
  else{
    _XMP_fatal("Dimension of coarray is too big.");
    return;
  }
}

void _xmp_gasnet_unpack_using_buf(gasnet_token_t t, const int addr_hi, const int addr_lo, 
				  const int dst_dims, const int ith)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  char* src_addr = _xmp_gasnet_buf[_XMP_world_rank];
  memcpy(dst, src_addr, dst_info_size);
  _unpack_array((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

void _xmp_gasnet_unpack(gasnet_token_t t, const char* src_addr, const size_t nbytes, 
			const int addr_hi, const int addr_lo, const int dst_dims, const int ith)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst = malloc(dst_info_size);
  memcpy(dst, src_addr, dst_info_size);
  _unpack_array((char *)UPCRI_MAKEWORD(addr_hi,addr_lo), dst_dims, src_addr+dst_info_size, dst);
  free(dst);
  gasnet_AMReplyShort1(t, _XMP_GASNET_UNPACK_REPLY, ith);
}

static void _stride_size_error(size_t request_size){
  if(_XMP_world_rank == 0){
    fprintf(stderr, "[ERROR] Memory size for coarray stride transfer is too small.\n");
    fprintf(stderr, "        Please set the environmental variable \"XMP_COARRAY_STRIDE_SIZE\".\n");
    fprintf(stderr, "        e.g.) export XMP_COARRAY_STRIDE_SIZE=%zuM (or more).\n", (request_size/1024/1024)+1);
  }
  _XMP_fatal_nomsg();
}

static void _gasnet_c_to_nonc_put(const int target_image, const size_t src_point, const int dst_dims, 
				  const _XMP_array_section_t *dst_info, 
				  const _XMP_coarray_t *dst, const void *src, size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  transfer_size += dst_info_size;
  char archive[transfer_size];
  memcpy(archive, dst_info, dst_info_size);
  memcpy(archive+dst_info_size, (char *)src+src_point, transfer_size - dst_info_size);

  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;
  if(transfer_size < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK, archive, (size_t)transfer_size,
			    HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims, 
			    _xmp_gasnet_stride_wait_size);
  }
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_put(target_image, _xmp_gasnet_buf[target_image], archive, (size_t)transfer_size);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target_image]), 
			   LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    _stride_size_error(transfer_size);
  }
  _xmp_gasnet_stride_wait_size++;
}

static void _gasnet_nonc_to_nonc_put(const int target_image, const int dst_dims, const int src_dims,
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src,
				     const _XMP_coarray_t *dst, const void *src_ptr, const size_t transfer_size)
{
  size_t dst_info_size = sizeof(_XMP_array_section_t) * dst_dims;
  size_t tsize = (size_t)transfer_size + dst_info_size;
  char archive[tsize];
  memcpy(archive, dst_info, dst_info_size);
  _pack_array(archive + dst_info_size, src_ptr, src_dims, src);
  _extend_stride_queue();
  _xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] = 0;

  if(tsize < gasnet_AMMaxMedium()){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_UNPACK, archive, tsize,
    			    HIWORD(dst->addr[target_image]), LOWORD(dst->addr[target_image]), dst_dims,
    			    _xmp_gasnet_stride_wait_size);
  }
  else if(tsize < _xmp_gasnet_stride_size){
    gasnet_put_bulk(target_image, _xmp_gasnet_buf[target_image], archive, tsize);
    gasnet_AMRequestShort4(target_image, _XMP_GASNET_UNPACK_USING_BUF, HIWORD(dst->addr[target_image]),
                           LOWORD(dst->addr[target_image]), dst_dims, _xmp_gasnet_stride_wait_size);
  }
  else{
    _stride_size_error(tsize);
  }

  GASNET_BLOCKUNTIL(_xmp_gasnet_stride_queue[_xmp_gasnet_stride_wait_size] == 1);
}

void _XMP_gasnet_put(const int dst_continuous, const int src_continuous, const int target_image, const int dst_dims, 
		     const int src_dims, const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, 
		     const _XMP_coarray_t *dst, const void *src, const size_t  length)
{
  size_t transfer_size = dst->elmt_size*length;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    size_t dst_point = get_offset(dst_info, dst_dims);
    size_t src_point = get_offset(src_info, src_dims);
    _gasnet_c_to_c_put(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    size_t dst_point = get_offset(dst_info, dst_dims);
    _gasnet_nonc_to_c_put(target_image, dst_point, src_dims, src_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    size_t src_point = get_offset(src_info, src_dims);
    _gasnet_c_to_nonc_put(target_image, src_point, dst_dims, dst_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
    _gasnet_nonc_to_nonc_put(target_image, dst_dims, src_dims, dst_info,
			     src_info, dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Unkown shape of coarray");
  }
}

static void _gasnet_c_to_c_get(const int target_image, const size_t dst_point, const size_t src_point, 
			       const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  gasnet_get_bulk(((char *)dst)+dst_point, target_image, ((char *)src->addr[target_image])+src_point,
		  transfer_size);

}

static void _gasnet_c_to_nonc_get(const int target_image, const size_t src_point, const int dst_dims, const _XMP_array_section_t *dst_info, 
				  const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
{
  if(transfer_size < _xmp_gasnet_stride_size){
    char* src_addr = (char *)_xmp_gasnet_buf[_XMP_world_rank];
    gasnet_get_bulk(src_addr, target_image, ((char *)src->addr[target_image])+src_point, (size_t)transfer_size);
    _unpack_array(((char *)dst), dst_dims, src_addr, dst_info);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

void _xmp_gasnet_pack(gasnet_token_t t, const char* info, const size_t am_request_size, 
		      const int src_addr_hi, const int src_addr_lo, const int src_dims, 
		      const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _pack_array(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyMedium2(t, _XMP_GASNET_UNPACK_GET_REPLY, archive, tansfer_size,
      			dst_addr_hi, dst_addr_lo);
}

void _xmp_gasnet_pack_get(gasnet_token_t t, const char* info, const size_t am_request_size,
			  const int src_addr_hi, const int src_addr_lo, const int src_dims, const int dst_dims,
			  const size_t tansfer_size, const int dst_addr_hi, const int dst_addr_lo)
{
  size_t src_size = sizeof(_XMP_array_section_t) * src_dims;
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *src_info = malloc(src_size);
  memcpy(src_info, info, src_size);
  char archive[tansfer_size + dst_size];
  memcpy(archive, info + src_size, dst_size);
  _pack_array(archive+dst_size, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  free(src_info);
  gasnet_AMReplyMedium3(t, _XMP_GASNET_UNPACK_GET_REPLY_NONC, archive, tansfer_size + dst_size,
                        dst_addr_hi, dst_addr_lo, dst_dims);
}

void _xmp_gasnet_unpack_get_reply_nonc(gasnet_token_t t, char *archive, size_t transfer_size,
				       const int dst_addr_hi, const int dst_addr_lo, const int dst_dims)
{
  size_t dst_size = sizeof(_XMP_array_section_t) * dst_dims;
  _XMP_array_section_t *dst_info = malloc(dst_size);
  memcpy(dst_info, archive, dst_size);

  _unpack_array((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), dst_dims, archive+dst_size, dst_info);
  done_get_flag = _XMP_N_INT_TRUE;
}


void _xmp_gasnet_unpack_get_reply(gasnet_token_t t, char *archive, size_t transfer_size, 
				  const int dst_addr_hi, const int dst_addr_lo)
{
  memcpy((char *)UPCRI_MAKEWORD(dst_addr_hi,dst_addr_lo), archive, transfer_size);
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_unpack_get_reply_using_buf(gasnet_token_t t)
{
  done_get_flag = _XMP_N_INT_TRUE;
}

void _xmp_gasnet_pack_using_buf(gasnet_token_t t, const char* info, const size_t am_request_size,
				const int src_addr_hi, const int src_addr_lo, const int src_dims,
				const int target_image)
{
  _XMP_array_section_t *src_info = (_XMP_array_section_t *)info;
  char *archive = _xmp_gasnet_buf[_XMP_world_rank];
  _pack_array(archive, (char *)UPCRI_MAKEWORD(src_addr_hi,src_addr_lo), src_dims, src_info);
  gasnet_AMReplyShort0(t, _XMP_GASNET_UNPACK_GET_REPLY_USING_BUF);
}

static void _gasnet_nonc_to_c_get(const int target_image, const int src_dims, const _XMP_array_section_t *src_info, 
				  const void *dst, const _XMP_coarray_t *src, const size_t transfer_size, const size_t dst_point)
{
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
  else if(transfer_size < _xmp_gasnet_stride_size){
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target_image, _xmp_gasnet_buf[target_image], transfer_size);
    memcpy(((char *)dst)+dst_point, _xmp_gasnet_buf[_XMP_world_rank], transfer_size);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

static void _gasnet_nonc_to_nonc_get(const int target_image, const int dst_dims, const int src_dims, 
				     const _XMP_array_section_t *dst_info, const _XMP_array_section_t *src_info, 
				     const void *dst, const _XMP_coarray_t *src, const size_t transfer_size)
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
  else if(transfer_size < _xmp_gasnet_stride_size){
    size_t am_request_size = sizeof(_XMP_array_section_t) * src_dims;
    char *archive = malloc(am_request_size);
    memcpy(archive, src_info, am_request_size);
    gasnet_AMRequestMedium4(target_image, _XMP_GASNET_PACK_USGIN_BUF, archive, am_request_size,
                            HIWORD(src->addr[target_image]), LOWORD(src->addr[target_image]), src_dims,
                            _XMP_world_rank);
    GASNET_BLOCKUNTIL(done_get_flag == _XMP_N_INT_TRUE);
    gasnet_get_bulk(_xmp_gasnet_buf[_XMP_world_rank], target_image, _xmp_gasnet_buf[target_image], 
		    transfer_size);
    _unpack_array((char *)dst, dst_dims, _xmp_gasnet_buf[_XMP_world_rank], dst_info);
    free(archive);
  }
  else{
    _stride_size_error(transfer_size);
  }
}

void _XMP_gasnet_get(const int src_continuous, const int dst_continuous, const int target_image, const int src_dims, 
		     const int dst_dims, const _XMP_array_section_t *src_info, const _XMP_array_section_t *dst_info, 
		     const _XMP_coarray_t *src, const void *dst, const size_t length)
{
  size_t transfer_size = src->elmt_size*length;

  if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_TRUE){
    size_t dst_point = get_offset(dst_info, dst_dims);
    size_t src_point = get_offset(src_info, src_dims);
    _gasnet_c_to_c_get(target_image, dst_point, src_point, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_TRUE && src_continuous == _XMP_N_INT_FALSE){
    size_t dst_point = get_offset(dst_info, dst_dims);
    _gasnet_nonc_to_c_get(target_image, src_dims, src_info, dst, src, transfer_size, dst_point);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_TRUE){
    size_t src_point = get_offset(src_info, src_dims);
    _gasnet_c_to_nonc_get(target_image, src_point, dst_dims, dst_info, dst, src, transfer_size);
  }
  else if(dst_continuous == _XMP_N_INT_FALSE && src_continuous == _XMP_N_INT_FALSE){
    _gasnet_nonc_to_nonc_get(target_image, dst_dims, src_dims, dst_info, src_info, dst, src, transfer_size);
  }
  else{
    _XMP_fatal("Unkown shape of coarray");
  }
}

