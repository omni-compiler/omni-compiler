#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>

#pragma xmp nodes p[4]
#pragma xmp template t[10]
#pragma xmp distribute t[block] onto p

int *array;
#pragma xmp align array[i] with t[i]

int *loArray;

int main()
{

  array = (int *)xmp_malloc(xmp_desc_of(array), 10);
  loArray = (int *)malloc(sizeof(int)*10);
	
#pragma xmp loop on t[i]
  for (int i = 0; i < 10; i++){
    array[i] = i;
  }
  
#pragma xmp gmove
  loArray[0:10] = array[0:10];

  for (int i = 0; i < 10; i++){
    if (loArray[i] != i){
      printf("Error! : a[%d] = %d\n", i, loArray[i]);
      xmp_exit(1);
    }
  }
  
#pragma xmp task on p[0]
  printf("PASS\n");

  return 0;
}
