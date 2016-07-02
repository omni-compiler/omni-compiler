#include <stdio.h>
#include <openacc.h>
#define N 100
#define ITER 100

int main()
{
  int iter;
  for(iter = 0; iter < ITER; iter++){
    acc_init(acc_device_default);
  
    int array[N];
    int sum = iter, i;

    for(i=0;i<N;i++) array[i] = i;

#pragma acc parallel loop reduction(+:sum)
    for(i=0;i<N;i++){
      sum += array[i];
    }

    acc_shutdown(acc_device_default);

    if(sum != (N-1)*N/2 + iter){
      printf("iter=%d, sum=%d\n", iter, sum);
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}
