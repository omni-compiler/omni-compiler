#define N 2
#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=N;
double a[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][*][i] with tx(*,*,i)

int main(){

  int i0,i1,i2,ierr;
  double b[N][N][N],err;

  for(i2=0;i2<n;i2++){
    for(i1=0;i1<n;i1++){
#pragma xmp loop (i0) on tx(*,*,i0)
      for(i0=0;i0<n;i0++){
        a[i2][i1][i0]=i2+i0+i1+1;
      }
    }
  }

  for(i2=0;i2<n;i2++){
    for(i1=0;i1<n;i1++){
      for(i0=0;i0<n;i0++){
        b[i2][i1][i0]=0.0;
      }
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1][1:n-1]=a[1:n-1][1:n-1][1:n-1];

  err=0.0;
  for(i2=1;i2<n;i2++){
    for(i1=1;i1<n;i1++){
      for(i0=1;i0<n;i0++){
        err=err+fabs(b[i2][i1][i0]-i2-i0-i1-1);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);
}
