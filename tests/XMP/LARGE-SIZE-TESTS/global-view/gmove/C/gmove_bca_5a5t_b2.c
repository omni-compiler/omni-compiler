#define NMAX 8
#include <stdio.h>
#include <stdlib.h>

extern int chk_int(int ierr);

int n=NMAX;
int a[n][n][n][n][n],b[n][n][n][n][n];
#pragma xmp nodes p(2,2,2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block,block,block) onto p
#pragma xmp align a[i0][i1][i2][i3][i4] with tx(i0,i1,i2,i3,i4)
#pragma xmp align b[*][i1][*][i3][i4] with tx(*,i1,*,i3,i4)

int main(){

  int i0,i1,i2,i3,i4,ierr;

#pragma xmp loop on tx(i0,i1,i2,i3,i4)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            a[i0][i1][i2][i3][i4]=i0+i1+i2+i3+i4+1;
          }
        }
      }
    }
  }

#pragma xmp loop on tx(*,i1,*,i3,i4)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            b[i0][i1][i2][i3][i4]=0;
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1][1:n-1][1:n-1][1:n-1]=a[1:n-1][1:n-1][1:n-1][1:n-1][1:n-1];

  ierr=0;
#pragma xmp loop on tx(*,i1,*,i3,i4)
  for(i0=1;i0<n;i0++){
    for(i1=1;i1<n;i1++){
      for(i2=1;i2<n;i2++){
        for(i3=1;i3<n;i3++){
          for(i4=1;i4<n;i4++){
            ierr=ierr+abs(b[i0][i1][i2][i3][i4]-i0-i1-i2-i3-i4-1);
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(ierr);
}
