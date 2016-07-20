#include<stdio.h>
#define N 10
#define M 20

int main()
{
  int array_1d[N];
  int array_2d[N][M];
  int i,j;

  for(i=0;i<N;i++){
    array_1d[i] = 0;
    for(j=0;j<M;j++){
      array_2d[i][j] = 0;
    }
  }

#pragma acc data create(array_1d, array_2d)
  {
#pragma acc update device(array_1d)
#pragma acc update device(array_2d)

#pragma acc parallel loop
    for(i=0;i<N;i++)
      array_1d[i] += i;

#pragma acc parallel loop collapse(2)
    for(i=0;i<N;i++){
      for(j=0;j<M;j++){
	array_2d[i][j] += i*N+j;
      }
    }

#pragma acc update host(array_2d)
#pragma acc update host(array_1d)
  }

  for(i=0;i<N;i++){
    if(array_1d[i] != i) return 1;
  }

  for(i=0;i<N;i++){
    for(j=0;j<M;j++){
      if(array_2d[i][j] != i*N+j) return 2;
    }
  }

  printf("PASS\n");
  return 0;
}
