#include<stdio.h>

int main()
{
  int array_1d[10];
  int array_2d[10][20];
  int i,j;

  for(i=0;i<10;i++){
    array_1d[i] = 0;
    for(j=0;j<20;j++){
      array_2d[i][j] = 0;
    }
  }

#pragma acc data copyin(array_1d, array_2d)
  {
#pragma acc parallel loop
    for(i=0;i<10;i++)
      array_1d[i] = i;

#pragma acc parallel loop collapse(2)
    for(i=0;i<10;i++){
      for(j=0;j<20;j++){
	array_2d[i][j] = i*10+j;
      }
    }

#pragma acc update host(array_2d[2:3][5:8])
#pragma acc update host(array_1d[4:3], array_2d[5:4][1:2])
  }

  for(i=0;i<10;i++){
    if(i>=4 && i <= 6){
      if(array_1d[i] != i) return 1;
    }else{
      if(array_1d[i] != 0) return 1;
    }
  }
  /*
  for(i=0;i<10;i++){
    printf("%d,", array_1d[i]);
  }
  printf("\n\n");
  */

  for(i=0;i<10;i++){
    for(j=0;j<20;j++){
      if(i >=2 && i <= 4 && j >= 5 && j <= 12){
	if(array_2d[i][j] != i*10+j) return 2;
      }else if(i >= 5 && i <= 8 && j >= 1 && j <= 2){
	if(array_2d[i][j] != i*10+j) return 2;
      }else{
	if(array_2d[i][j] != 0) return 2;
      }
    }
  }

  /*
  for(i=0;i<10;i++){
    for(j=0;j<10;j++){
      printf("%2d,", array_2d[i][j]);
    }
    printf("\n");
  }
  */
  
  return 0;
}
