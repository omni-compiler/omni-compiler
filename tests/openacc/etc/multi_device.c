#include<stdio.h>
#include<openacc.h>

#define M 1000
#define N 2000

int main()
{
  int a[M];
  int b[N];
  int i;

  int num_device = acc_get_num_devices( acc_device_default );
  //printf("num_device = %d\n", num_device);                                                                                                                                                                                                                                       

  if(num_device < 2){
    return 0;
  }

#pragma acc parallel loop
  for(i=0;i<M;i++){
    a[i] = i;
  }

  acc_set_device_num(2, acc_device_default);

#pragma acc parallel loop
  for(i=0;i<N;i++){
    b[i] = i;
  }

  for(i=0;i<M;i++){
    if(a[i] != i){
      return 1;
    }
  }

  for(i=0;i<N;i++){
    if(b[i] != i){
      return 1;
    }
  }
  return 0;
}
