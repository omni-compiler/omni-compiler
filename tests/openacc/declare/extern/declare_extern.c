#include "declare_extern.h"

int a[N];
double b[N];

int main()
{
  int i;

  func();

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
