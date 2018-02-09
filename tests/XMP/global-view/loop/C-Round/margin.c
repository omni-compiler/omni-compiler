#include <stdlib.h>
#include <stdio.h>

int main(){

#pragma xmp nodes p(2,2)
#pragma xmp template t(0:7,0:7)
#pragma xmp distribute t(block,block) onto p

  int a[8][8];
#pragma xmp align a[i][j] with t(i,j)
#pragma xmp shadow a[1][1]

  int b[8][8];
#pragma xmp align b[i][j] with t(i,j)
#pragma xmp shadow b[1][1]

  int result = 0;
  
#pragma xmp array on t(:,:)
  a[:][:] = 1;

#pragma xmp reflect (a) width(/periodic/1,/periodic/1)

#pragma xmp array on t(:,:)
  b[:][:] = 1;

#pragma xmp loop (i,j) on t(i,j) margin(/unbound/1,/unbound/1)
  for (int i = 0; i < 8; i++){
    for (int j = 0; j < 8; j++){
      b[i][j] = 1;
    }
  }
  
#pragma xmp loop (i,j) on t(i,j) reduction(+:result) expand(/unbound/1,/unbound/1)
  for (int i = 0; i < 8; i++){
    for (int j = 0; j < 8; j++){
      if (a[i][j] != b[i][j]) result = 1;
    }
  }

#pragma xmp task on p(1,1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      exit(1);
    }
  }

  return 0;
  
}
