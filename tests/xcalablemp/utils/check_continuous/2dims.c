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
    for(int j=0;j<array_info[1].length;j++){
      tmp[1] = stride_offset[1] * j;
	stride[num++] = tmp[0] + tmp[1];
      }
    }

  for(int i=1;i<elmts;i++)
    if(array_info[dims-1].distance  != stride[i] - stride[i-1]){
      return false;
    }

  return true;
}

static void test(int start0, int len0, int stride0, int start1, int len1, int stride1)
{
  _XMP_array_section_t array[DIM];
  array[0].start  = start0;  array[1].start  = start1;
  array[0].length = len0;    array[1].length = len1;    
  array[0].stride = stride0; array[1].stride = stride1; 
  array[0].elmts  = SIZE;    array[1].elmts  = SIZE;    
  array[1].distance = sizeof(int);
  array[0].distance = array[1].elmts * array[1].distance;

  int elmts = array[0].length * array[1].length;
#ifdef _CHECK
  if(_check_continuous(array, DIM, elmts)){
#else
  if(_check_stride(array, DIM, elmts)){
#endif
      if(len0 == 1)
	printf("a[%d]", start0);
      else if(stride0 == 1)
	printf("a[%d:%d]", start0, len0);
      else
	printf("a[%d:%d:%d]", start0, len0, stride0);

      if(len1 == 1)
	printf("[%d]", start1);
      else if(stride1 == 1)
	printf("[%d:%d]", start1, len1);
      else
	printf("[%d:%d:%d]", start1, len1, stride1);

      printf("\n");
  }
}

int main()
{
  for(int start0=0;start0<SIZE;start0++){
    for(int len0=1;len0<=SIZE;len0++){
      for(int stride0=1;stride0<SIZE;stride0++){
	for(int start1=0;start1<SIZE;start1++){
	  for(int len1=1;len1<=SIZE;len1++){
	    for(int stride1=1;stride1<SIZE;stride1++){
	      if(start0+(len0-1)*stride0 < SIZE && start1+(len1-1)*stride1 < SIZE 
		 && len0 * len1 != 1){
		test(start0, len0, stride0, start1, len1, stride1);
	      }
	    }
	  }
	}
      }
    }
  }

  return 0;
}
