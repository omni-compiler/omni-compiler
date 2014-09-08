#include <stdio.h>
#include <stdlib.h>

extern int chk_int(int ierr);

int n=8;
int a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic(2)) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

int main(){

  int i,ierr;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[0:n]=a[0:n];

  ierr=0;
#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    ierr=ierr+abs(b[i]-a[i]);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(ierr);
}

