#define NAMELEN 25
#define N 8
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_G2L_1a4t_bc(){

char name[NAMELEN]="gmove_G2L_1a4t_bc";
int n=N;
double a[n][n][n][n];
#pragma xmp nodes p(2,2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][*][*][i] with tx(*,*,*,i)

  int i0,i1,i2,i3,ierr;
  double b[N][N][N][N],err;

  for(i3=0;i3<n;i3++){
    for(i2=0;i2<n;i2++){
      for(i1=0;i1<n;i1++){
#pragma xmp loop (i0) on tx(*,*,*,i0)
        for(i0=0;i0<n;i0++){
          a[i3][i2][i1][i0]=i3+i2+i0+i1+1;
        }
      }
    }
  }

  for(i3=0;i3<n;i3++){
    for(i2=0;i2<n;i2++){
      for(i1=0;i1<n;i1++){
        for(i0=0;i0<n;i0++){
          b[i3][i2][i1][i0]=0.0;
        }
      }
    }
  }

#pragma xmp gmove
  b[1:n-1][1:n-1][1:n-1][1:n-1]=a[1:n-1][1:n-1][1:n-1][1:n-1];

  err=0.0;
  for(i3=1;i3<n;i3++){
    for(i2=1;i2<n;i2++){
      for(i1=1;i1<n;i1++){
        for(i0=1;i0<n;i0++){
          err=err+fabs(b[i3][i2][i1][i0]-i3-i2-i0-i1-1);
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_G2L_4a4t_bc(){

char name[NAMELEN]="gmove_G2L_4a4t_bc";
int n=N;
int a[n][n][n][n];
#pragma xmp nodes p(2,2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i0][i1][i2][i3] with tx(i0,i1,i2,i3)

  int i0,i1,i2,i3,b[N][N][N][N],ierr=0;

#pragma xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
  for (i3=0;i3<n;i3++){
    for (i2=0;i2<n;i2++){
      for (i1=0;i1<n;i1++){
        for (i0=0;i0<n;i0++){
          a[i0][i1][i2][i3]=(i0+1)+(i1+1)+(i2+1)+(i3+1);
        }
      }
    }
  }

  for (i3=0;i3<n;i3++){
    for (i2=0;i2<n;i2++){
      for (i1=0;i1<n;i1++){
        for (i0=0;i0<n;i0++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[1:4][1:4][1:4][1:4]=a[4:4][4:4][4:4][4:4];

  for (i3=1;i3<5;i3++){
    for (i2=1;i2<5;i2++){
      for (i1=1;i1<5;i1++){
        for (i0=1;i0<5;i0++){
          ierr=ierr+abs(b[i0][i1][i2][i3]-(i0+4)-(i1+4)-(i2+4)-(i3+4));
        }
      }
    }
  }

#pragma xmp reduction (+:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_G2L_1a4t_bc();
  gmove_G2L_4a4t_bc();

  return 0;

}
