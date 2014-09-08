#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);

int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

int main(){

  int i,j,k,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+2;
      }
    }
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
#pragma xmp loop (k) on tx(k)
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[:][:][:]=a[:][:][:];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
#pragma xmp loop (k) on tx(k)
      for(k=0;k<n;k++){
/*        printf("i=%d,j=%d,k=%d,b=%f,a=%f\n",i,j,k,b[i][j][k],a[i][j][k],myrank);i*/
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(ierr);

}

