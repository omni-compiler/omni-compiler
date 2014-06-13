#include <stdio.h>
#include <stdlib.h>
int main()
{
  int *array;
  int i,j;

  array = (int*)malloc(1000*sizeof(int));
  if(array == NULL){
    return 1;
  }
  for(i=0;i<1000;i++){
    array[i] = i;
  }

#pragma acc data copyin(array[0:1000])
  {
    int n;
    for(n=100; n<10000; n+=100){
      int sum = 0;
#pragma acc parallel loop collapse(2) firstprivate(n) reduction(+:sum)
      for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	  sum += array[(i+j)%1000];
	}
      }
    }
  }
  
  return 0;
}

      
