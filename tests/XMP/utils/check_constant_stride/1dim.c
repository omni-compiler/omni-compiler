#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include "common.h"
#define SIZE 10
#define DIM 1

static int wrapper_check(_XMP_array_section_t* array)
{
#ifdef _CHECK
  return _heavy_check_stride(array, DIM);
#else
  return _is_the_same_constant_stride(array, DIM);
#endif
}

static void test(int start0, int len0, int stride0)
{
  _XMP_array_section_t array[DIM];
  array[0].start  = start0;
  array[0].length = len0;
  array[0].stride = stride0;
  array[0].elmts  = SIZE;
  array[0].distance = sizeof(int);

  if(wrapper_check(array)){
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
    for(int len0=2;len0<=SIZE;len0++){
      for(int stride0=1;stride0<SIZE;stride0++){
	if(start0+(len0-1)*stride0 < SIZE){
	  test(start0, len0, stride0);
	}
      }
    }
  }

  return 0;
}
