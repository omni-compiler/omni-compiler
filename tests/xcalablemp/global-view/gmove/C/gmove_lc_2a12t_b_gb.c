#include <stdio.h>
#include <stdlib.h>

extern int chk_int(int ierr);

int n=16;
int a[n][n],b[n][n];
int m1[2]={9,7};
#pragma xmp nodes p(4)
#pragma xmp nodes q(2,2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(gblock(m1),gblock(m1)) onto q
#pragma xmp align a[i0][*] with tx(i0)
#pragma xmp align b[i0][i1] with ty(i0,i1)

int main(){

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

#pragma xmp loop (i0,i1) on ty(i0,i1)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      b[i0][i1]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  ierr=0;
#pragma xmp loop (i0,i1) on ty(i0,i1)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      ierr=ierr+abs(b[i0][i1]-i0-i1-1);
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(ierr);
}
