#include <stdio.h>

#pragma xmp nodes p[8]

int main(){

#pragma xmp tasks
  {

#pragma xmp task on p[0:4]
    {
#pragma xmp barrier on p
    }

#pragma xmp task on p[4:4]
    {
#pragma xmp barrier on p
    }

  }

#pragma xmp task on p[0]
  {
    printf("PASS\n");
  }

  return 0;

}
