#include <stdio.h>
#pragma xmp nodes p(*)

int main()
{
#pragma xmp task on p(1)
  printf("PASS\n");

  return 0;
}
