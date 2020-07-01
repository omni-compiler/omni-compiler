#include <stdlib.h>
#include <stdio.h>

#pragma xmp nodes p[4][4]

#pragma xmp template t[1:100][1:100]
#pragma xmp distribute t[cyclic(5)][block] onto p

int main(void){

  int k;

  k = 0;

#pragma xmp task on p[0:4:2][1:2]
  {
    k = 1;
  }

#pragma xmp task on p[0:4:2][1:2] nocomm
  {
    k = 0;
  }

#pragma xmp reduction (+:k)

#pragma xmp task on p[0][0]
  {
    if (k != 0){
      printf("ERROR\n");
      exit(1);
    }
  }
  
  k = 0;

#pragma xmp task on t[1][1]
  {
    k = 1;
  }

#pragma xmp task on t[1][1] nocomm
  {
    k = 0;
  }

#pragma xmp reduction (+:k)

#pragma xmp task on p[0][0]
  {
    if (k != 0){
      printf("ERROR\n");
      exit(1);
    }
    else {
      printf("PASS\n");
    }
  }

  return 0;

}
