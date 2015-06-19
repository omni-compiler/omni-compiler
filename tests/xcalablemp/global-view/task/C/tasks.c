#include <stdio.h>

#pragma xmp nodes p(8)

int main(){

#pragma xmp tasks
  {

#pragma xmp task on p(1:4)
    {
#pragma xmp barrier on p
    }

#pragma xmp task on p(5:8)
    {
#pragma xmp barrier on p
    }

  }

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }

  return 0;

}
