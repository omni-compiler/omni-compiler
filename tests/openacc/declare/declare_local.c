#include <stdio.h>
#define N 100

void func_devptr(int dev_a[N]);
void func_present(int *a, double *b, float *c, long long d[N], int *e);

int main(void)
{
  int i;
  int a = 1;
  double b = 2;
  float c = 3.0f;
  long long d[N] = {0};
  int f[N] = {0};
#pragma acc declare copy(a) copyin(b)
#pragma acc declare copyout(c) create(d)
#pragma acc declare copy(f)

#pragma acc parallel
  {
    a += 1;
    b += 1.0;
    c = 2.5;
  }

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



  int e = 1;
#pragma acc data copy(e)
  func_present(&a, &b, &c, d, &e);
#pragma acc update host(a,b,c,d)

  //check
  if(a != 3) return 5;

  if(b != 4.0) return 6;

  if(c != 5.0) return 7;

  for(i=0;i<N;i++){
    if(d[i] != i + 10100) return 8;
  }
  if(e != 4) return 9;



#pragma acc host_data use_device(f)
  func_devptr(f);
#pragma acc update host(f)
  //check
  for(i = 0; i < N; i++){
    if(f[i] != i * 3 + 1) return 10;
  }


  printf("PASS\n");
  return 0;
}


void func_present(int *a, double *b, float *c, long long d[N], int *e)
{
  int i;
#pragma acc declare pcopy(a[0:1]) pcopyin(b[0:1])
#pragma acc declare pcopyout(c[0:1]) pcreate(d[0:N]) present(e[0:1])

#pragma acc parallel
  {
    *a += 1;
    *b += 1.0;
    *c += 2.5;
    *e += 3;
  }

#pragma acc parallel loop
  for(i=0;i<N;i++){
    d[i] += 100;
  }
}

void func_devptr(int dev_a[N])
{
  int i;
#pragma acc declare deviceptr(dev_a)
#pragma acc parallel loop
  for(i = 0; i < N; i++){
    dev_a[i] += i * 3 + 1;
  }
}
