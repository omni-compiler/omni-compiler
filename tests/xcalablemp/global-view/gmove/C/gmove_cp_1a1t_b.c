#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

int main(){

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[1:4]=a[4:4];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=1;i<5;i++){
      err=err+fabs(b[i]-(i+4));
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);
}

