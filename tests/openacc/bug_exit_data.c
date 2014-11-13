#include<stdio.h>
#include<stdlib.h>
#define N (1024*1024)

int main(){
  double *a;
  a = (double*)malloc(sizeof(double)*N);

  for(int i=0;i<1024;i++){
#pragma acc enter data copyin(a[0:N])
#pragma acc exit data delete(a[0:N])
  }

  return 0;
}
