#include <stdio.h>

int main()
{
  int array[100];
  int i;

  for(i=0;i<100;i++){
    array[i] = 0;
  }
  
#pragma acc data copyin(array)
  {
#pragma acc parallel loop async(1)
    for(i=0;i<100;i++){
      array[i] = i;
    }
#pragma acc update host(array[0:20]) async(1)

#pragma acc update host(array[80:20]) async(1)
  }

  for(i=0;i<20;i++){
    if(array[i] != i){
      return 1;
    }
  }

  for(i=20;i<80;i++){
    if(array[i] != 0){
      return 2;
    }
  }

  for(i=80;i<100;i++){
    if(array[i] != i){
      return 3;
    }
  }

  printf("PASS\n");
  return 0;
}
