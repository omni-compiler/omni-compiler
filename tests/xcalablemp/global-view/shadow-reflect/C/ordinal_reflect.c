#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

#pragma xmp nodes p(2,2,2)

#pragma xmp template t(0:63,0:63,0:63)
#pragma xmp distribute t(block,block,block) onto p

int a[64][64][64], b[64][64][64], c[64][64][64];
#pragma xmp align a[i][j][k] with t(i,j,k)
#pragma xmp align b[i][j][k] with t(i,j,k)
#pragma xmp align c[i][j][k] with t(i,j,k)
#pragma xmp shadow a[1][1][1]
#pragma xmp shadow b[1][1][1]
#pragma xmp shadow c[1][1][1]

int main(void){

#pragma xmp loop (i,j,k) on t(i,j,k)
  for (int i = 0; i < 64; i++){
    for (int j = 0; j < 64; j++){
      for (int k = 0; k < 64; k++){
	a[i][j][k] = i * 10000 + j * 100 + k;
	b[i][j][k] = i * 10000 + j * 100 + k;
	c[i][j][k] = i * 10000 + j * 100 + k;
      }
    }
  }

#pragma xmp reflect (a) width(/periodic/1:1, 0, 0)
#pragma xmp reflect (a) width(0, /periodic/1:1, 0)
#pragma xmp reflect (a) width(0, 0, /periodic/1:1)

#pragma xmp reflect (b) width(/periodic/1:1, /periodic/1:1, /periodic/1:1)

#pragma xmp reflect (c) width(/periodic/1:1, /periodic/1:1, /periodic/1:1) async(100)
#pragma xmp wait_async(100)

  int result = 0;

#pragma xmp loop (i,j,k) on t(i,j,k) reduction(+:result)
  for (int i = 0; i < 64; i++){
  for (int j = 0; j < 64; j++){
  for (int k = 0; k < 64; k++){

    for (int kk = -1; kk <= 1; kk++){
    for (int jj = -1; jj <= 1; jj++){
    for (int ii = -1; ii <= 1; ii++){

      if (a[i+ii][j+jj][k+kk] != b[i+ii][j+jj][k+kk]){
	result = -1;
      }

      if (a[i+ii][j+jj][k+kk] != c[i+ii][j+jj][k+kk]){
	result = -1;
      }

    }
    }
    }

  }
  }
  }

#pragma xmp task on p(1,1,1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;

}
