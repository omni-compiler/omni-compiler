#include "declare_extern.h"

float c;
long long d[N];

void func()
{
  int i;
#pragma acc parallel loop
  for(i=0;i<N;i++){
    a[i] = i;
  }

#pragma acc data
  {
#pragma acc parallel loop
    for(i=0;i<N;i++){
      b[i] = i + 1.0;
    }
  }

#pragma acc parallel
  c = 2.5;

#pragma acc parallel loop
  for(i=0;i<N;i++){
    d[i] = i + 10000;
  }
}
