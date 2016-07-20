#include <stdio.h>
#define N 1000

int main()
{
  int i;
  int a[N];
#pragma acc parallel loop num_gangs(3) vector_length(32) copyout(a)
  for(i = 0; i < N; i++){
    a[i] = i;
  }

  for(i = 0; i < N; i++){
    if(a[i] != i) return 1;
  }

  printf("PASS\n");
  return 0;
}
