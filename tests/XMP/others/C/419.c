#include <stdio.h>

#pragma xmp template t(0:19)
#pragma xmp nodes p(4)
#pragma xmp distribute t(block) onto p
int a[20];
#pragma xmp align a[i] with t(i)

int main(){

#pragma xmp loop on t(i)
#pragma omp parallel for
  for(int i = 0; i < 20; i++)
    a[i] = i;

#pragma xmp task on p(1)
  printf("PASS\n");

  return 0;
}

