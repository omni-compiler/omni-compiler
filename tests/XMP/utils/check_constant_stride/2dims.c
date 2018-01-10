#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include "common.h"
#define SIZE 10
#define DIM 2
#define LENGTH_CHECK(START, LEN, STRIDE) (START+(LEN-1)*STRIDE < SIZE)

static int wrapper_check(_XMP_array_section_t* array)
{
#ifdef _HEAVY
  return _heavy_check_stride(array, DIM);
#else
  return _is_the_same_constant_stride(array, DIM);
#endif
}

static void test(int start[DIM], int len[DIM], int stride[DIM])
{
  _XMP_array_section_t array[DIM];
  array[0].start  = start[0];  array[1].start  = start[1];
  array[0].length = len[0];    array[1].length = len[1];
  array[0].stride = stride[0]; array[1].stride = stride[1]; 
  array[0].elmts  = SIZE;      array[1].elmts  = SIZE;
  array[1].distance = sizeof(int);
  array[0].distance = array[1].distance * array[1].elmts;

  if(wrapper_check(array)){
  //  if(! _heavy_check_stride(array, DIM)){
  //    if(_is_the_same_constant_stride(array, DIM)){
    if(len[0] == 1)
      printf("a[%d]", start[0]);
    else if(stride[0] == 1)
      printf("a[%d:%d]", start[0], len[0]);
    else
      printf("a[%d:%d:%d]", start[0], len[0], stride[0]);
    
    if(len[1] == 1)
      printf("[%d]", start[1]);
    else if(stride[1] == 1)
      printf("[%d:%d]", start[1], len[1]);
    else
      printf("[%d:%d:%d]", start[1], len[1], stride[1]);

    printf("\n");
    }
  //  }
}

int main()
{
  int start[DIM], len[DIM], stride[DIM];

  for(start[0]=0;start[0]<SIZE;start[0]++){
    for(len[0]=2;len[0]<=SIZE;len[0]++){
      for(stride[0]=1;stride[0]<SIZE;stride[0]++){
	for(start[1]=0;start[1]<SIZE;start[1]++){
	  for(len[1]=2;len[1]<=SIZE;len[1]++){
	    for(stride[1]=1;stride[1]<SIZE;stride[1]++){
	      if(LENGTH_CHECK(start[0],len[0],stride[0]) && LENGTH_CHECK(start[1],len[1],stride[1])){
		test(start, len, stride);
	      }
	    }
	  }
	}
      }
    }
  }

  return 0;
}
