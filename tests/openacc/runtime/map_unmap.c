#include <stdio.h>
#include <openacc.h>

#define N 1024

int main()
{
  int array[N];
  int* dev_p;
  int i;
  
  for(i=0;i<N;i++) array[i] = 0;

  size_t size = sizeof(int)*N;
  dev_p = (int*)acc_malloc(size);

  //this kernel has effect
#pragma acc parallel loop pcopy(array)
  for(i=0;i<N;i++) array[i] += 1;

  for(i=0;i<N;i++){
    if(array[i] != 1) return 1;
  }

  acc_map_data(array, dev_p, size);

  //this kernel has no effect
#pragma acc parallel loop pcopy(array)
  for(i=0;i<N;i++) array[i] += 1;

  for(i=0;i<N;i++){
    if(array[i] != 1) return 2;
  }

  acc_unmap_data(array);

  //this kernel has effect
#pragma acc parallel loop pcopy(array)
  for(i=0;i<N;i++) array[i] += 1;

  for(i=0;i<N;i++){
    if(array[i] != 2) return 3;
  }

  acc_free(dev_p);
  
  printf("PASS\n");
  return 0;
}


  
  
