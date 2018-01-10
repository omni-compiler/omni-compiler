#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_bca_4a4t_b(){

char name[NAMELEN]="gmove_bca_4a4t_b";
int n=8;
int a[n][n][n][n],b[n][n][n][n];
#pragma xmp nodes p(2,2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block,block) onto p
#pragma xmp align a[i0][i1][i2][i3] with tx(i0,i1,i2,i3)
#pragma xmp align b[*][i1][i2][i3] with tx(*,i1,i2,i3)

  int i0,i1,i2,i3,ierr;

#pragma xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          a[i0][i1][i2][i3]=i0+i1+i2+i3+1;
        }
      }
    }
  }

#pragma xmp loop (i1,i2,i3) on tx(*,i1,i2,i3)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1][1:n-1][1:n-1]=a[1:n-1][1:n-1][1:n-1][1:n-1];

  ierr=0;
#pragma xmp loop (i1,i2,i3) on tx(*,i1,i2,i3)
  for(i0=1;i0<n;i0++){
    for(i1=1;i1<n;i1++){
      for(i2=1;i2<n;i2++){
        for(i3=1;i3<n;i3++){
          ierr=ierr+abs(b[i0][i1][i2][i3]-i0-i1-i2-i3-1);
        }
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_4a4t_b2(){

char name[NAMELEN]="gmove_bca_4a4t_b2";
int n=8;
int a[n][n][n][n],b[n][n][n][n];
#pragma xmp nodes p(2,2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block,block) onto p
#pragma xmp align a[i0][i1][i2][i3] with tx(i0,i1,i2,i3)
#pragma xmp align b[*][i1][i2][*] with tx(*,i1,i2,*)

  int i0,i1,i2,i3,ierr;

#pragma xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          a[i0][i1][i2][i3]=i0+i1+i2+i3+1;
        }
      }
    }
  }

#pragma xmp loop (i1,i2) on tx(*,i1,i2,*)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1][1:n-1][1:n-1]=a[1:n-1][1:n-1][1:n-1][1:n-1];

  ierr=0;
#pragma xmp loop (i1,i2) on tx(*,i1,i2,*)
  for(i0=1;i0<n;i0++){
    for(i1=1;i1<n;i1++){
      for(i2=1;i2<n;i2++){
        for(i3=1;i3<n;i3++){
          ierr=ierr+abs(b[i0][i1][i2][i3]-i0-i1-i2-i3-1);
        }
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_bca_4a4t_b();
  gmove_bca_4a4t_b2();

  return 0;

}
