#include <stdio.h>
#include "xmp.h"

#pragma xmp nodes p(1)

#pragma xmp template t(0:7)
#pragma xmp distribute t(block) onto p

int a[8];
#pragma xmp align a[i] with t(i)
#pragma xmp shadow a[1]

int main()
{
#pragma xmp loop on t(i)
  for(int i=0;i<8;i++)
    a[i] = i;

#pragma xmp reflect (a) width(/periodic/1:1)

#pragma xmp task on p(1)
  printf("[1] %d %d\n", a[-1], a[8]);

#pragma xmp task on p(1)
  {
    if (a[-1] != 7 || a[8] != 0){
      printf("ERROR\n");
      xmp_exit(1);
    }
    else {
      printf("PASS\n");
    }
  }
  
  return 0;
}
