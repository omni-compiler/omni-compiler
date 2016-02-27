#include <stdio.h>
#include <stdlib.h>
#define N 10
#pragma xmp nodes p(2,2)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int a[N][N];
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp shadow a[1][2]

int main(){

#pragma xmp loop (i,j) on t(j,i)
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      a[i][j] = N * i + j;
  
#pragma xmp loop (i,j) on t(j,i)
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      int v = a[i][j];
      if(v != a[i][j]){
	printf("Error! : a[%d][%d] = %d, v = %d\n", i, j, a[i][j], v);
	exit(1);
      }
    }
  }

#pragma xmp loop (j,i) on t(j,i)
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      int v = a[i][j];
      if(v != a[i][j]){
	printf("Error! : a[%d][%d] = %d, v = %d\n", i, j, a[i][j], v);
	exit(1);
      }
    }
  }
  
#pragma xmp task on p(1,1)
  printf("PASS\n");
  
  return 0;
}

