#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include "common.h"
#define SIZE0 6
#define SIZE1 7
#define SIZE2 6
#define SIZE3 3
#define SIZE4 6
#define DIM 5
#define LENGTH_CHECK(START, LEN, STRIDE, SIZE) (START+(LEN-1)*STRIDE < SIZE)

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
  array[0].start  = start[0];  array[1].start  = start[1];  array[2].start  = start[2];  array[3].start  = start[3];
  array[0].length = len[0];    array[1].length = len[1];    array[2].length = len[2];    array[3].length = len[3];
  array[0].stride = stride[0]; array[1].stride = stride[1]; array[2].stride = stride[2]; array[3].stride = stride[3];
  array[0].elmts  = SIZE0;     array[1].elmts  = SIZE1;     array[2].elmts  = SIZE2;     array[3].elmts  = SIZE3;
  array[4].start  = start[4];
  array[4].length = len[4];
  array[4].stride = stride[4];
  array[4].elmts  = SIZE4;

  array[4].distance = sizeof(int);
  array[3].distance = array[4].distance * array[4].elmts;
  array[2].distance = array[3].distance * array[3].elmts;
  array[1].distance = array[2].distance * array[2].elmts;
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

    if(len[2] == 1)
      printf("[%d]", start[2]);
    else if(stride[2] == 1)
      printf("[%d:%d]", start[2], len[2]);
    else
      printf("[%d:%d:%d]", start[2], len[2], stride[2]);

    if(len[3] == 1)
      printf("[%d]", start[3]);
    else if(stride[3] == 1)
      printf("[%d:%d]", start[3], len[3]);
    else
      printf("[%d:%d:%d]", start[3], len[3], stride[3]);

    if(len[4] == 1)
      printf("[%d]", start[4]);
    else if(stride[4] == 1)
      printf("[%d:%d]", start[4], len[4]);
    else
      printf("[%d:%d:%d]", start[4], len[4], stride[4]);

    printf("\n");
    }
  //  }
}

int main()
{
  int start[DIM], len[DIM], stride[DIM];

  for(start[0]=0;start[0]<SIZE0;start[0]++){
    for(len[0]=2;len[0]<=SIZE0;len[0]++){
      for(stride[0]=1;stride[0]<SIZE0;stride[0]++){
	for(start[1]=0;start[1]<SIZE1;start[1]++){
	  for(len[1]=2;len[1]<=SIZE1;len[1]++){
	    for(stride[1]=1;stride[1]<SIZE1;stride[1]++){
	      for(start[2]=0;start[2]<SIZE2;start[2]++){
		for(len[2]=2;len[2]<=SIZE2;len[2]++){
		  for(stride[2]=1;stride[2]<SIZE2;stride[2]++){
		    for(start[3]=0;start[3]<SIZE3;start[3]++){
		      for(len[3]=2;len[3]<=SIZE3;len[3]++){
			for(stride[3]=1;stride[3]<SIZE3;stride[3]++){
			  for(start[4]=0;start[4]<SIZE4;start[4]++){
			    for(len[4]=2;len[4]<=SIZE4;len[4]++){
			      for(stride[4]=1;stride[4]<SIZE4;stride[4]++){
				if(LENGTH_CHECK(start[0],len[0],stride[0],SIZE0) &&
				   LENGTH_CHECK(start[1],len[1],stride[1],SIZE1) &&
				   LENGTH_CHECK(start[2],len[2],stride[2],SIZE2) &&
				   LENGTH_CHECK(start[3],len[3],stride[3],SIZE3) &&
				   LENGTH_CHECK(start[4],len[4],stride[4],SIZE4)){
				  test(start, len, stride);
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  return 0;
}
