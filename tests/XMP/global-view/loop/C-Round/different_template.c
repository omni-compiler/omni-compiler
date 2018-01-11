#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p(*)
#pragma xmp template t0(0:9)
#pragma xmp template t1(0:9)
#pragma xmp distribute t0(block) onto p
#pragma xmp distribute t1(block) onto p
float a[10];
#pragma xmp align a[j] with t0(j)

int main(void){

#pragma xmp loop on t1(i)
  for(int i=0; i<10; i++)
      a[i] = i;

  if(xmp_node_num() == 1)
    printf("PASS\n");

  return 0;
}
