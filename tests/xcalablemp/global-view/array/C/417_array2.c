#include "xmp.h"

#pragma xmp nodes p(2,2)
#pragma xmp template t(0:99,0:99)
#pragma xmp distribute t(block,block) onto p

float (*a)[100], (*b)[100];
//float a[100], b[100];
#pragma xmp align a[i][j] with t(i,j)
#pragma xmp align b[i][j] with t(i,j)

int main(void){

  a = (float (*)[100])xmp_malloc(xmp_desc_of(a), 100, 100);
  b = (float (*)[100])xmp_malloc(xmp_desc_of(b), 100, 100);

#pragma xmp array on t(:,:)
  a[0:99][0:99] = b[0:99][0:99];

  return 0;

}
