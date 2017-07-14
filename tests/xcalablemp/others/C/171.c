#include <stdio.h>
#include <stdbool.h>
#define N 10

bool test_1dim()
{
  int a[N], len = 5;
  for(int i=0;i<N;i++)
    a[i] = i;

  a[1:len] = a[0:len];

  for(int i=0;i<N;i++)
    if(i >= 1 && i < 1+len){
      if(a[i] != i-1) return false;
    }
    else
      if(a[i] != i)   return false;

  return true;
}

bool test_2dim()
{
  int a[N][N], len = 3;

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      a[i][j] = i * N + j;

  a[1:3][1:3] = a[0:3][0:3];

  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      if(i >= 1 && i < 4 && j >= 1 && j < 4){
	if(a[i][j] != (i-1)*N+(j-1)) return false;
      }
      else{
	if(a[i][j] != i*N+j) return false;
      }
  
}

int main()
{
  bool flag1 = test_1dim();
  bool flag2 = test_2dim();

  if(flag1 && flag2){
    printf("PASS\n");
    return 0;
  }
  else{
    printf("ERROR\n");
    return 1;
  }
}
  
