#include <stdio.h>
#include "xmp.h"

int main(){
  
#pragma xmp nodes p(4)
#pragma xmp template t(0:3)
#pragma xmp distribute t(block) onto p
  int a[3];
#pragma xmp align a[i] with t(i)

  int me = xmp_node_num();
  
#pragma xmp loop on t(i)
  for (int i = 0; i < 3; i++){
    if (me == 1)
      a[0] = 0;
    if (me == 2)
      a[1] = 1;
    if (me == 3)
      a[2] = 2;
    if (me == 4)
      a[3] = 3;
  }

  int result = 0;
  
#pragma xmp loop on t(i) reduction(+:result)
  for (int i = 0; i < 3; i++){
    if (a[i] != i) result = 1;
  }

#pragma xmp task on p(1)
  {
    if (result == 0){
      printf("PASS\n");
    }
    else{
      printf("ERROR\n");
      xmp_exit(1);
    }
  }

  return 0;

  
}
