#include <stdio.h>
#define N 100

int main()
{
  int i;
  int a[N];
  char flag;

  // true (const)
  for(i = 0; i < N; i++) a[i] = 0;
#pragma acc data copy(a) if(4)
  {
    for(i = 0; i < N; i++) a[i] = 1;
#pragma acc parallel loop
    for(i = 0; i < N; i++) a[i] += 1;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 1) return 1;
  }


#if 0
  // false (const)
  for(i = 0; i < N; i++) a[i] = 0;
#pragma acc data copy(a) if(0)
  {
    for(i = 0; i < N; i++) a[i] = 2;
#pragma acc parallel loop
    for(i = 0; i < N; i++) a[i] += 1;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 3) return 2;
  }
#endif
  

  // true (variable)
  flag = 12;
  for(i = 0; i < N; i++) a[i] = 0;
#pragma acc data copy(a) if(flag)
  {
    for(i = 0; i < N; i++) a[i] = 1;
#pragma acc parallel loop
    for(i = 0; i < N; i++) a[i] += 1;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 1) return 3;
  }


#if 0
  // false (variable)
  flag = 0;
  for(i = 0; i < N; i++) a[i] = 0;
#pragma acc data copy(a) if(flag)
  {
    for(i = 0; i < N; i++) a[i] = 2;
#pragma acc parallel loop
    for(i = 0; i < N; i++) a[i] += 1;
  }
  //check
  for(i = 0; i < N; i++){
    if(a[i] != 3) return 4;
  }
#endif


  printf("PASS\n");
  return 0;
}
