#include "xmp.h"

#pragma xmp nodes p(4)
#pragma xmp template t(0:99)
#pragma xmp distribute t(block) onto p

float *a, *b;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)

int main(void){

  a = (float *)xmp_malloc(xmp_desc_of(a), 100);
  b = (float *)xmp_malloc(xmp_desc_of(b), 100);

#pragma xmp array on t(:)
  a[0:99] = b[0:99];

  return 0;

}
