#define NAMELEN 25
#define N 8
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_G2L_1a2t_bc(){

char name[NAMELEN]="gmove_G2L_1a2t_bc";
int n=N;
double a[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][i] with tx(*,i)

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
  chk_int(name, ierr);

}

void gmove_G2L_2a2t_bc(){

char name[NAMELEN]="gmove_G2L_2a2t_bc";
int n=N;
double a[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)

  int i,j,ierr;
  double b[N][N],err;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:4][0:4]=a[4:4][4:4];

  err=0.0;
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      err=err+fabs(b[i][j]-(i+5+j+5));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_G2L_1a2t_bc();
  gmove_G2L_2a2t_bc();

  return 0;

}
