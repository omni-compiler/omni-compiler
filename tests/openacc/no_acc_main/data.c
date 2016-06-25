#include <stdlib.h>
#include "acc_func.h"
#define N 10

int acc_func()
{
  int i,a[N];
#pragma acc data copyout(a)
  {
#pragma acc parallel loop
    for(i=0;i<N;i++){
      a[i] = i;
    }
  }

  for(i=0;i<N;i++){
    if(a[i] != i){
      return 1;
    }
  }

  return 0;
}
