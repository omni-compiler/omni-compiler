#include <stdio.h>
#include "xmp.h"

int a[10], b[10];
#pragma xmp nodes p(2)
#pragma xmp template t(0:9)
#pragma xmp distribute t(cyclic) onto p
#pragma xmp align a[i] with t(i)

int main(){

#pragma xmp loop (i) on t(i)
  for (int i = 0; i < 10; i++){
    a[i] = 1;
  }

  for (int i = 0; i < 10; i++){
    b[i] = 0;
  }

#pragma xmp gmove
  b[0] = a[0];

  //  printf("rank=%d, B[0]=%d\n", xmp_node_num(), B[0]);

  int result = 0;
  
  if (b[0] != 1) result = 1;

#pragma xmp reduction (+:result)
  
#pragma xmp task on p(1) nocomm
  {
    if (result != 0){
      printf("ERROR in gmove_s_ge\n");
      xmp_exit(1);
    }
    else {
      printf("PASS\n");
    }
  }

  return 0;
}
