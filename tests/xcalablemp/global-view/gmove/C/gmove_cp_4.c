#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_cp_2a2t_b(){

char name[NAMELEN]="gmove_cp_2a2t_b";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[1:4][1:4]=a[4:4][4:4];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=1;i<5;i++){
    for(j=1;j<5;j++){
      err=err+fabs(b[i][j]-(i+4+j+4));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_cp_2a2t_b_bc(){

char name[NAMELEN]="gmove_cp_2a2t_b_bc";
int n=8;
int a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i0][i1] with tx(i0,i1)
#pragma xmp align b[i0][i1] with ty(i0,i1)

  int i0,i1,ierr=0;

#pragma xmp loop (i0,i1) on tx(i0,i1)
  for (i1=0;i1<n;i1++){
    for (i0=0;i0<n;i0++){
      a[i0][i1]=(i0+1)+(i1+1);
    }
  }

#pragma xmp loop (i0,i1) on ty(i0,i1)
  for (i1=0;i1<n;i1++){
    for (i0=0;i0<n;i0++){
      b[i0][i1]=0;
    }
  }

#pragma xmp gmove
  b[1:4][1:4]=a[4:4][4:4];

#pragma xmp loop (i0,i1) on ty(i0,i1)
  for (i1=1;i1<5;i1++){
    for (i0=1;i0<5;i0++){
      ierr=ierr+abs(b[i0][i1]-(i0+4)-(i1+4));
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_cp_2a2t_bc(){

char name[NAMELEN]="gmove_cp_2a2t_bc";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:4][0:4]=a[4:4][4:4];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      err=err+fabs(b[i][j]-(i+5+j+5));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_cp_2a2t_c(){

char name[NAMELEN]="gmove_cp_2a2t_c";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:4][0:4]=a[4:4][4:4];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
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

  gmove_cp_2a2t_b();
  gmove_cp_2a2t_b_bc();
  gmove_cp_2a2t_bc();
  gmove_cp_2a2t_c();

  return 0;

}
