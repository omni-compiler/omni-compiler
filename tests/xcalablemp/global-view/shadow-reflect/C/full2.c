#include <stdio.h>
#include <stdlib.h>

#pragma xmp nodes p(2,2)

#pragma xmp template t(0:3,0:3)
#pragma xmp distribute t(cyclic,cyclic) onto p

int a[4][4];
#pragma xmp align a[i][j] with t(i,j)
#pragma xmp shadow a[*][*]

int result = 0;

int main(){

#pragma xmp loop (i,j) on t(i,j)
  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 4; j++){
      a[i][j] = i * 10 + j;
    }
  }

#pragma xmp reflect (a)

  for (int i = 0; i < 4; i++){
    for (int j = 0; j < 4; j++){
      if (a[i][j] != i * 10 + j) result = 1;
    }
  }

#pragma xmp reduction(+:result)

#pragma xmp task on p(1,1)
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
