#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

#pragma xmp nodes p(4)
#pragma xmp template t(0:7)
#pragma xmp distribute t(block) onto p

  int a[8][8];
#pragma xmp align a[i][*] with t(i)
  int b[8][8];
#pragma xmp align b[*][j] with t(j)

int main(void){

#pragma xmp loop (j) on t(j)
  for (int j = 0; j < 8; j++){
    for (int i = 0; i < 8; i++){
      b[i][j] = i * 10 + j;
    }
  }

#pragma xmp gmove async(10)
  a[:][:] = b[:][:];

#pragma xmp wait_async(10)

  int ierr = 0;

#pragma xmp loop (i) on t(i) reduction(+:ierr)
  for (int i = 0; i < 8; i++){
    for (int j = 0; j < 8; j++){
      if (a[i][j] != i * 10 + j) ierr++;
    }
  }

#pragma xmp task on p(1)
  {
    if (ierr == 0){
      printf("PASS\n");
    }
    else {
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }

  return 0;

}
