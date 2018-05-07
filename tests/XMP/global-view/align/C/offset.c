#include <stdio.h>
#include <xmp.h>
#pragma xmp nodes p[*]
#pragma xmp template t[2:10]
#pragma xmp distribute t[block] onto p
int a[10];
#pragma xmp align a[i] with t[i+2]

int main(void){
#pragma xmp loop on t[i]
  for(int i=2;i<10;i++)
    a[i] = i;
  
#pragma xmp task on p[0]
  {
    printf("PASS\n");
  }
  return 0;
}
