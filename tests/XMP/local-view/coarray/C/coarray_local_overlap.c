#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#define N 5
int a1[N]:[*], a2[N];
int b1[N][N]:[*], b2[N][N];
bool flag = true;

void test_1dim()
{
  for(int i=0;i<N;i++){
    a1[i] = i;
    a2[i] = a1[i];
  }
  
  a1[1:N-1]:[0] = a1[0:N-1];
  memmove(&a2[1], &a2[0], sizeof(int)*(N-1));
  
  for(int i=0;i<N;i++)
    if(a1[i] != a2[i])
      flag = false;
}

void test_2dim()
{
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      b1[i][j] = i * N + j;
      b2[i][j] = b1[i][j];
    }
  }

  b1[1:N-1][1:N-1]:[0] = b1[0:N-1][0:N-1];
  char *tmp = malloc(sizeof(int)*(N-1)*(N-1));
  size_t offset = 0;
  for(int i=0;i<N-1;i++)
    for(int j=0;j<N-1;j++){
      memcpy(tmp+offset, &b2[i][j], sizeof(int));
      offset += sizeof(int);
    }

  offset = 0;
  for(int i=1;i<N;i++)
    for(int j=1;j<N;j++){
      memcpy(&b2[i][j], tmp+offset, sizeof(int));
      offset += sizeof(int);
    }
  
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      if(b1[i][j] != b2[i][j])
	flag = false;
}

int main()
{
    test_1dim();
  //  test_2dim();

  if(flag)
    printf("PASS\n");
  else
    printf("Error !\n");
  
  return (flag)? 0 : 1;
}
