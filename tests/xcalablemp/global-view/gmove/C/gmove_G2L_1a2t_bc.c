#define N 2
#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=N;
double a[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][i] with tx(*,i)

int main(){

  int i0,i1,ierr;
  double b[N][N],err;

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on tx(*,i1)
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      b[i0][i1]=0.0;
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1]=a[1:n-1][1:n-1];

  err=0.0;
  for(i0=1;i0<n;i0++){
    for(i1=1;i1<n;i1++){
      err=err+fabs(b[i0][i1]-i0-i1-1);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);
}
