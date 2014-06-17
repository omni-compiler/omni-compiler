#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100
int a[N][N], result = 0;
double b[N][N];
float c[N][N];
#pragma xmp nodes p(*)
#pragma xmp template t(0:99)
#pragma xmp distribute t(cyclic(2)) onto p
#pragma xmp align a[*][i] with t(i)
#pragma xmp align b[*][i] with t(i)
#pragma xmp align c[*][i] with t(i)

int main(void)
{
  for(int i=0;i<N;i++){
#pragma xmp loop on t(j)
    for(int j=0;j<N;j++){
      a[i][j] = j*N+i;
      b[i][j] = (double)(j*N+i);
      c[i][j] = (float)(j*N+i);
    }
  }
  
  for(int i=0;i<N;i++){
#pragma xmp array on t(:)
  a[i][:] = 1;
#pragma xmp array on t(:)
  b[i][:] = 2.0;
#pragma xmp array on t(:)
  c[i][:] = 3.0;
  }

  for(int i=0;i<N;i++){
#pragma xmp loop on t(j)
    for(int j=0;j<N;j++){
      if(a[i][j]!=1 || b[i][j]!=2.0 || c[i][j]!=3.0){
	result = -1;
      }
    }
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
  {
    if(result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }
  return 0;
}
