#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

int main(){

  int i,j,ierr;
  double err;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:4][0:4]=a[4:4][4:4];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      err=err+fabs(b[i][j]-(i+5+j+5));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);
}

