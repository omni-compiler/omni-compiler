#include <stdio.h>
#define N 100


int main()
{
  int a[N];
  int i;

  a[0] = 1;
  for(i = 1; i < N; i++) a[i] = 0;
#pragma acc parallel
#pragma acc loop seq //private(i)
  for(i = 1; i < N; i++){
    a[i] += a[i-1] + 2;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 2 * i + 1) return 1;
  }


#if 0 //bug
  a[0] = 1;
  for(i = 1; i < N; i++) a[i] = 0;
#pragma acc parallel
#pragma acc loop seq private(i)
  for(i = 1; i < N; i++){
    a[i] += a[i-1] + 3;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 3 * i + 1) return 2;
  }
#endif
  
  printf("PASS\n");
  return 0;
}
