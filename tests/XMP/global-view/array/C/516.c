#include <xmp.h>

int a[10];
#pragma xmp nodes p[1]
#pragma xmp template t1[10]
#pragma xmp distribute t1[block] onto p

int main(int argc, char** argv){
  #pragma xmp align a[i] with t1[i]

  int *b;
#pragma xmp template t2[:]
  #pragma xmp distribute t2[block] onto p
  #pragma xmp align b[i] with t2[i]
  #pragma xmp template_fix[block] t2[10]
  b = xmp_malloc(xmp_desc_of(b), 10);

  return 0;
}
