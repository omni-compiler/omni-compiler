/*
  Current omni compiler creates a task descripter of each functions.
  This test is to confirm "free of task descripter" in function foo().
  In the future, the omni compiler will create one task queue in a program.
  Then, this test will be unneeded.
 */
#include <stdio.h>
#pragma xmp nodes p(*)
#define TIMES 50000

void foo(){
  int a;
#pragma xmp bcast (a) on p(:)
}

int main(){
  for(int i=0;i<TIMES;i++)
    foo();

#pragma xmp task on p(1)
  {
    printf("PASS\n");
  }

  return 0;
}
