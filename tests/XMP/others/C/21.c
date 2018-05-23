#include "xmp.h"

int main(){

  int n1 = 100;
  int n2 = 100;

#pragma xmp nodes p(4,4)

#pragma xmp template t(0:99,0:99)
#pragma xmp distribute t(block,block) onto p

  float (*a)[n2];
#pragma xmp align a[i][j] with t(i,j)


  a = (float (*)[n2])xmp_malloc(xmp_desc_of(a), n1, n2);

#pragma xmp task on p(1,1)
  {
    a[0][0] = 0;
  }

}
