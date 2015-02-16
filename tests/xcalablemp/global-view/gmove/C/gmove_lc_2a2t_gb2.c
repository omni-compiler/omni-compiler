#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=8;
double a[n][n],b[n][n];
int mx[2]={2,6},my[2]={2,6};
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(gblock(mx),gblock(mx)) onto p
#pragma xmp distribute ty(gblock(my),gblock(my)) onto p
#pragma xmp align a[i][j] with tx(j,i)
#pragma xmp align b[i][j] with ty(j,i)

int main(){

  int i,j,ierr;
  double err ;

#pragma xmp loop (i,j) on tx(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on ty(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on ty(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-i-j-2);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);
}

