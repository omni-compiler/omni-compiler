#include <stdio.h>
#define N 100

int a[N];
double b[N];
float c;
long long d[N];

#pragma acc declare create(a, b)
#pragma acc declare copyin(c, d)

int main()
{
  int i;

#pragma acc parallel loop
  for(i=0;i<N;i++){
    a[i] = i;
  }

#pragma acc parallel loop
  for(i=0;i<N;i++){
    b[i] = i + 1.0;
  }

#pragma acc parallel
  c = 2.5;

#pragma acc parallel loop
  for(i=0;i<N;i++){
    d[i] = i + 10000;
  }

#pragma acc update host(a,b,c,d)

  //check
  for(i = 0; i < N; i++){
    if(a[i] != i) return 1;
  }

  for(i = 0; i < N; i++){
    if(b[i] != i + 1.0) return 2;
  }

  if(c != 2.5) return 3;

  for(i=0;i<N;i++){
    if(d[i] != i + 10000) return 4;
  }

  printf("PASS\n");
  return 0;
}
