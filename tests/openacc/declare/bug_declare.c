#include <stdio.h>
#define N 100
int a[N];

#pragma acc declare create(a)

int main(void){

  int i;
  for(i=0;i<N;i++) a[i] = i+5;

#pragma acc update device(a)

#pragma acc parallel loop present(a)
  for(i=0;i<N;i++) a[i]++;

#pragma acc update host(a)

  //verify
  for(i=0;i<N;i++){
    if(a[i] != i + 6){
      return 1;
    }
  }

  printf("PASS\n");

  return 0;
}
