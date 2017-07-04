#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_lc_21a1t_b(){

char name[NAMELEN]="gmove_lc_21a1t_b";
int n=16;
int a[n][n],b[n];
#pragma xmp nodes p(4)
#pragma xmp nodes q(4)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(block) onto q
#pragma xmp align a[i0][*] with tx(i0)
#pragma xmp align b[i0] with ty(i0)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[0:n]=a[0][0:n];

  ierr=0;
#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    ierr=ierr+abs(b[i0]-i0-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_lc_21a1t_b_gb2(){

char name[NAMELEN]="gmove_lc_21a1t_b_gb2";
int n=16;
int a[n][n],b[n];
int m1[4]={3,4,3,6};
#pragma xmp nodes p(4)
#pragma xmp nodes q(4)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(gblock(m1)) onto q
#pragma xmp align a[i0][*] with tx(i0)
#pragma xmp align b[i0] with ty(i0)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[0:n]=a[0][0:n];

  ierr=0;
#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    ierr=ierr+abs(b[i0]-i0-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_lc_21a1t_b_gb(){

char name[NAMELEN]="gmove_lc_21a1t_b_gb";
int n=16;
int a[n][n],b[n];
int m1[4]={3,4,3,6};
#pragma xmp nodes p(4)
#pragma xmp nodes q(4)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(gblock(m1)) onto q
#pragma xmp align a[i0][*] with tx(i0)
#pragma xmp align b[i0] with ty(i0)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[0:n]=a[0:n][0];

  ierr=0;
#pragma xmp loop (i0) on ty(i0)
  for(i0=0;i0<n;i0++){
    ierr=ierr+abs(b[i0]-i0-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_lc_21a2t_b(){

char name[NAMELEN]="gmove_lc_21a2t_b";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0]=a[0][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<1;j++){
      err=err+fabs(b[i][j]-i-2);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a12t_b_gb(){

char name[NAMELEN]="gmove_lc_2a12t_b_gb";
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
  chk_int(name, ierr);

}

void gmove_lc_2a2t_b(){

char name[NAMELEN]="gmove_lc_2a2t_b";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_bc(){

char name[NAMELEN]="gmove_lc_2a2t_bc";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_b_h(){

char name[NAMELEN]="gmove_lc_2a2t_b_h";
int n=9;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_c(){

char name[NAMELEN]="gmove_lc_2a2t_c";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_c_h(){

char name[NAMELEN]="gmove_lc_2a2t_c_h";
int n=9;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_gb2(){

char name[NAMELEN]="gmove_lc_2a2t_gb2";
int n=8;
double a[n][n],b[n][n];
int mx[2]={2,6},my[2]={2,6};
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(gblock(mx),gblock(mx)) onto p
#pragma xmp distribute ty(gblock(my),gblock(my)) onto p
#pragma xmp align a[i][j] with tx(j,i)
#pragma xmp align b[i][j] with ty(j,i)

  int i,j,ierr;
  double err ;

#pragma xmp loop (i,j) on tx(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on ty(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on ty(j,i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-i-j-2);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_gb3(){

char name[NAMELEN]="gmove_lc_2a2t_gb3";
int n=8;
double a[n][n],b[n][n];
int mx[2]={2,6},my[2]={2,6};
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(gblock(mx),gblock(mx)) onto p
#pragma xmp distribute ty(gblock(my),gblock(my)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)

  int i,j,ierr;
  double err ;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-i-j-2);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_2a2t_gb(){

char name[NAMELEN]="gmove_lc_2a2t_gb";
int n=8;
double a[n][n],b[n][n];
int m[2]={3,5};
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(gblock(m),gblock(m)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

  int i,j,ierr;
  double err ;

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
  b[1:n-1][1:n-1]=a[1:n-1][1:n-1];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=1;i<n;i++){
    for(j=1;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_lc_21a1t_b();
  gmove_lc_21a1t_b_gb2();
  gmove_lc_21a1t_b_gb();
  gmove_lc_21a2t_b();
  gmove_lc_2a12t_b_gb();
  gmove_lc_2a2t_b();
  gmove_lc_2a2t_bc();
  gmove_lc_2a2t_b_h();
  gmove_lc_2a2t_c();
  gmove_lc_2a2t_c_h();
  gmove_lc_2a2t_gb2();
  gmove_lc_2a2t_gb3();
  gmove_lc_2a2t_gb();

  return 0;

}
