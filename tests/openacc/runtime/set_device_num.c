#include <stdio.h>
#include <openacc.h>
#define N 100

int main()
{
  acc_set_device_num(1, acc_device_default);
  
  int array[N];
  int sum = 0, i;

  for(i=0;i<N;i++) array[i] = i;

#pragma acc parallel loop reduction(+:sum)
  for(i=0;i<N;i++){
    sum += array[i];
  }

  if(sum != (N-1)*N/2){
    printf("sum=%d\n", sum);
    return 1;
  }

  printf("PASS\n");
  return 0;
}
