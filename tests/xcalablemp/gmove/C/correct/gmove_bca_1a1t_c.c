#include <stdio.h>
#include <stdlib.h>

int n=8;
int a[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i] with tx(i)

int main(){

  int i,ierr,b[n];

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[0:n]=a[0:n];

  ierr=0;
  for(i=0;i<n;i++){
    ierr=ierr+abs(b[i]-(i+1));
  }

  printf("max error=%d\n",ierr);

  return 0;

}

