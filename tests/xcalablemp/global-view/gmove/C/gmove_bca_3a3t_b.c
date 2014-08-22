#define NMAX 4
#include <stdio.h>
#include <stdlib.h>

extern int chk_int(int ierr);

int n=NMAX;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i0][i1][i2] with tx(i0,i1,i2)
#pragma xmp align b[*][i1][i2] with tx(*,i1,i2)

int main(){

  int i0,i1,i2,ierr;

#pragma xmp loop (i0,i1,i2) on tx(i0,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        a[i0][i1][i2]=i0+i1+i2+1;
      }
    }
  }

#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  ierr=0;
#pragma xmp loop (i1,i2) on tx(*,i1,i2)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        ierr=ierr+abs(b[i0][i1][i2]-i0-i1-i2-1);
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(ierr);
}
