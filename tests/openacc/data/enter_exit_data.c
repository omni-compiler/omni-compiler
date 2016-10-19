#include <stdio.h>
#include <stdlib.h>
#define N 100000

int main(void)
{
  int a[100]; int b[200];
  int *c = (int*)malloc(sizeof(int) * N);
  int i;

  if(c == NULL){
    return -1;
  }

  //test 1
#pragma acc enter data create(a)
#pragma acc parallel loop present(a)
  for(i=0;i<100;i++){
    a[i] = i*2;
  }
#pragma acc exit data copyout(a)

  //verify
  for(i=0;i<100;i++){
    if(a[i] != i*2) return 1;
  }


  //test 2
  for(i=0;i<200;i++){
    b[i] = i;
  }

#pragma acc enter data copyin(b)
#pragma acc parallel loop present(b)
  for(i=0;i<200;i++){
    b[i] += i;
  }
#pragma acc exit data copyout(b)

  for(i=0;i<200;i++){
    if(b[i] != i*2) return 2;
  }

  //test3
  for(i = 0; i < N; i++){
    c[i] = i;
  }

#pragma acc enter data copyin(c[0:N]) async(5)
#pragma acc parallel loop present(c[0:N]) async(5)
  for(i = 0; i < N; i++){
    c[i] += i;
  }
#pragma acc exit data copyout(c[0:N]) async(5)
#pragma acc wait(5)

  for(i = 0; i < N; i++){
    if(c[i] != i * 2) return 3;
  }

  return 0;
}
