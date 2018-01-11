#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_cp_3a3t_b_bc(){

char name[NAMELEN]="gmove_cp_3a3t_b_bc";
int n=8;
int a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp distribute ty(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i0][i1][i2] with tx(i0,i1,i2)
#pragma xmp align b[i0][i1][i2] with ty(i0,i1,i2)

  int i0,i1,i2,ierr=0;

#pragma xmp loop (i0,i1,i2) on tx(i0,i1,i2)
  for (i2=0;i2<n;i2++){
    for (i1=0;i1<n;i1++){
      for (i0=0;i0<n;i0++){
        a[i0][i1][i2]=(i0+1)+(i1+1)+(i2+1);
      }
    }
  }

#pragma xmp loop (i0,i1,i2) on ty(i0,i1,i2)
  for (i2=0;i2<n;i2++){
    for (i1=0;i1<n;i1++){
      for (i0=0;i0<n;i0++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[1:4][1:4][1:4]=a[4:4][4:4][4:4];

#pragma xmp loop (i0,i1,i2) on ty(i0,i1,i2)
  for (i2=1;i2<5;i2++){
    for (i1=1;i1<5;i1++){
      for (i0=1;i0<5;i0++){
        ierr=ierr+abs(b[i0][i1][i2]-(i0+4)-(i1+4)-(i2+4));
      }
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_cp_3a3t_b(){

char name[NAMELEN]="gmove_cp_3a3t_b";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
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
#pragma xmp loop (i,j,k) on tx(i,j,k)
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

void gmove_cp_3a3t_bc(){

char name[NAMELEN]="gmove_cp_3a3t_bc";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
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
#pragma xmp loop (i,j,k) on tx(i,j,k)
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

void gmove_cp_3a3t_c(){

char name[NAMELEN]="gmove_cp_3a3t_c";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic,cyclic) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
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
#pragma xmp loop (i,j,k) on tx(i,j,k)
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

  gmove_cp_3a3t_b_bc();
  gmove_cp_3a3t_b();
  gmove_cp_3a3t_bc();
  gmove_cp_3a3t_c();

  return 0;

}
