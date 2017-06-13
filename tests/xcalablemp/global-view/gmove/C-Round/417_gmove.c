#include <xmp.h>
#define N 10
#pragma xmp nodes p(2)
#pragma xmp template t(:)
#pragma xmp distribute t(block) onto p
float *pa, *pb;
#pragma xmp align pa[i] with t(i)
#pragma xmp align pb[i] with t(i)

int main()
{
#pragma xmp template_fix t(0:N-1)
  pa = (float *)xmp_malloc(xmp_desc_of(pa), N);
  pb = (float *)xmp_malloc(xmp_desc_of(pb), N);

#pragma xmp gmove
  pa[0:N] = pb[0:N];

  return 0;
}
