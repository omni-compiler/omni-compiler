#include <stdio.h>

int main()
{
  int a[10][20][30];

#pragma acc parallel loop
  for(int i = 0; i < 10; i++){
#pragma acc loop
    for(int j = 0; j < 20; j++){
#pragma acc loop
      for(int k = 0; k < 30; k++){
	a[i][j][k] = i*j+k;
      }
    }
  }

  for(int i = 0; i < 10; i++){
    for(int j = 0; j < 20; j++){
      for(int k = 0; k < 30; k++){
	if(a[i][j][k] != i*j+k) return 1;
      }
    }
  }

  printf("PASS\n");
  return 0;
}
