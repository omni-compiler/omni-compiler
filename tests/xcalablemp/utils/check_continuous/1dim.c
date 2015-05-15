#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#define SIZE 10
#define DIM 2

typedef struct _XMP_array_section{
  long long start;
  long long length;
  long long stride;
  long long elmts;
  long long distance;
} _XMP_array_section_t;

extern unsigned int _XMP_get_dim_of_allelmts(const int dims,
					     const _XMP_array_section_t* array_info);
extern int _check_continuous(const _XMP_array_section_t *array_info, const int dims, const int elmts);

static int _check_stride(_XMP_array_section_t* array_info, int dims, int elmts)
{
  int stride[elmts], tmp[dims], stride_offset[dims];
  
  for(int i=0;i<dims;i++)
    stride_offset[i] = array_info[i].stride * array_info[i].distance;

  for(int i=0,num=0;i<array_info[0].length;i++){
    tmp[0] = stride_offset[0] * i;
    stride[num++] = tmp[0];
  }

  for(int i=1;i<elmts;i++)
    if(array_info[dims-1].distance  != stride[i] - stride[i-1]){
      return false;
    }

  return true;
}

static void test(int start0, int len0, int stride0)
{
  _XMP_array_section_t array[DIM];
  array[0].start  = start0;
  array[0].length = len0;
  array[0].stride = stride0;
  array[0].elmts  = SIZE;
  array[0].distance = sizeof(int);

  int elmts = array[0].length;
#ifdef _CHECK
  if(_check_continuous(array, DIM, elmts)){ printf("a\n");
#else
    if(_check_stride(array, DIM, elmts)){ printf("b\n");
#endif
      if(len0 == 1)
	printf("a[%d]", start0);
      else if(stride0 == 1)
	printf("a[%d:%d]", start0, len0);
      else
	printf("a[%d:%d:%d]", start0, len0, stride0);

      printf("\n");
  }
}

int main()
{
  for(int start0=0;start0<SIZE;start0++){
    for(int len0=1;len0<=SIZE;len0++){
      for(int stride0=1;stride0<SIZE;stride0++){
	if(start0+(len0-1)*stride0 < SIZE && len0 != 1){
	  test(start0, len0, stride0);
	}
      }
    }
  }

  return 0;
}
