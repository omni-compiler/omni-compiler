#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>

int n=8;
int a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with ty(i)

int main(){

  int i0,ierr=0;

#pragma xmp loop (i0) on tx(i0)
  for (i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i0) on ty(i0)
  for (i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[1:4]=a[4:4];

#pragma xmp loop (i0) on ty(i0)
  for (i0=1;i0<5;i0++){
    ierr=ierr+abs(b[i0]-(i0+4));
//    printf("i0=%d,b=%d\n",i0,b[i0]);
  }

#pragma xmp reduction (+:ierr)
  printf("max error=%d\n",ierr);

  return 0;

}
