#include<stdio.h>
#define NUM_NODES (4)
#define N (128)

int array[N];

#include "macro_in_pragma.h"

int main()
{
  int i;
  int sum = 0;
  char str[] = "test: '//' and '/* ... */' are not comment sign in string literal\n";
  
#pragma xmp loop (i) on t(i)
  for(i=0;i<N;i++){
    array[i] = i*i;
  }

#pragma xmp loop (i) on t(i) reduction(+:sum)
  for(i=0;i<N;i++){
    sum += array[i];
  }

  if(sum != (N*(N-1)*(2*N-1))/6){
    return 1;
  }
  return 0;
}
