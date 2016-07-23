#include <stdio.h>
#define N 100

int main(void)
{
  int i;
  int a = 1;
  double b = 2;
  float c = 3.0f;
  long long d[N] = {0};
#pragma acc declare copy(a) copyin(b)
#pragma acc declare copyout(c) create(d)

#pragma acc parallel
  a += 1;

#pragma acc parallel
  b += 1.0;

#pragma acc parallel
  c = 2.5;

#pragma acc parallel loop
  for(i=0;i<N;i++){
    d[i] = i + 10000;
  }

#pragma acc update host(a,b,c,d)

  //check
  if(a != 2) return 1;

  if(b != 3.0) return 2;

  if(c != 2.5) return 3;

  for(i=0;i<N;i++){
    if(d[i] != i + 10000) return 4;
  }

  printf("PASS\n");
  return 0;
}
