#include <stdio.h>
#define N 100

int a[N];
double b[N];
float c;
long long d[N];
long *e;

#pragma acc declare create(a, b)
#pragma acc declare copyin(c, d)
#pragma acc declare deviceptr(e)

void func_devptr()
{
#pragma acc parallel
  e[0] += 10;
}

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
  
  long host_e = 5;
#pragma acc data copy(host_e)
  {
#pragma acc host_data use_device(host_e)
    e = &host_e;

    func_devptr();
  }

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

  if(host_e != 15) return 5;

  printf("PASS\n");
  return 0;
}
