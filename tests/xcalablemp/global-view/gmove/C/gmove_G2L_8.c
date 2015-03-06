#define NAMELEN 25
#define N 8
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_G2L_1a3t_bc(){

char name[NAMELEN]="gmove_G2L_1a3t_bc";
int n=N;
double a[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][*][i] with tx(*,*,i)

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
  chk_int(name, ierr);

}

void gmove_G2L_1a3t_bc2(){

char name[NAMELEN]="gmove_G2L_1a3t_bc2";
int n=8;
double a[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][*][i] with tx(*,*,i)

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
  b[:][:][:]=a[:][:][:];

  err=0.0;
  for(i2=0;i2<n;i2++){
    for(i1=0;i1<n;i1++){
      for(i0=0;i0<n;i0++){
        err=err+fabs(b[i2][i1][i0]-i2-i0-i1-1);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_G2L_3a3t_bc(){

char name[NAMELEN]="gmove_G2L_3a3t_bc";
int n=8;
double a[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double b[N][N][N],err;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:4][0:4][0:4]=a[4:4][4:4][4:4];

  err=0.0;
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      for(k=0;k<4;k++){
        err=err+fabs(b[i][j][k]-(i+5+j+5+k+5));
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_G2L_1a3t_bc();
  gmove_G2L_1a3t_bc2();
  gmove_G2L_3a3t_bc();

  return 0;

}
