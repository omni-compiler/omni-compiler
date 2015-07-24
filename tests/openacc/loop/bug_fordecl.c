#include <stdio.h>

int main()
{
  int a[10][20];

#pragma acc parallel loop
  for(int i = 0; i < 10; i++){
    //#pragma acc loop
    for(int j = 0; j < 20; j++){
      a[i][j] = i*j;
    }
  }

  for(int i = 0; i < 10; i++){
    for(int j = 0; j < 20; j++){
      if(a[i][j] != i*j) return 1;
    }
  }

  printf("PASS\n");
  return 0;
}
