#include <stdio.h>
#include <stdlib.h>
#define N (32 * 1024 * 1024)


int main()
{
  int i;
  int *a = (int*)malloc(sizeof(int) * N);
  int *b = (int*)malloc(sizeof(int) * N);
  int *c = (int*)malloc(sizeof(int) * N);

  if(!(a && b && c)){
    printf("malloc err\n");
    return 1;
  }

#pragma acc data copyout(a[0:N], b[0:N], c[0:N])
  {
    //for avoiding initial cost
#pragma acc parallel async(2)
    a[0] = 0;
#pragma acc parallel async(10)
    a[1] = 1;
#pragma acc wait


    //A
#pragma acc parallel loop async(2) num_gangs(1)
    for(i = 0; i < N; i++){
      a[i] = i * 2;
    }

    //B, this will finish before A
#pragma acc parallel loop async(10)
    for(i = 0; i < N; i++){
      b[i] = 1;
    }

#pragma acc wait(10)

#pragma acc parallel loop async(2)
    for(i = 0; i < N; i++){
      c[i] = a[i] + b[i];
    }
#pragma acc wait
  }

  //check
  for(i = 0; i < N; i++){
    if(c[i] != i * 2 + 1){
      return 1;
    }
  }

  free(a);
  free(b);
  free(c);

  printf("PASS\n");
  return 0;
}
