#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_cp_1a1t_b(){

char name[NAMELEN]="gmove_cp_1a1t_b";
int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

 int i,/*j,*/ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[1:4]=a[4:4];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=1;i<5;i++){
      err=err+fabs(b[i]-(i+4));
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_cp_1a1t_b_c(){

char name[NAMELEN]="gmove_cp_1a1t_b_c";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with ty(i)

  int i0,ierr=0;

#pragma xmp loop (i0) on tx(i0)
  for (i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i0) on ty(i0)
  for (i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[1:4]=a[4:4];

#pragma xmp loop (i0) on ty(i0)
  for (i0=1;i0<5;i0++){
    ierr=ierr+abs(b[i0]-(i0+4));
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_cp_1a1t_bc(){

char name[NAMELEN]="gmove_cp_1a1t_bc";
int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic(2)) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

 int i,/*j,*/ierr;
 double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[4:4]=a[0:4];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=4;i<8;i++){
    err=err+fabs(b[i]-(i-3));
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_cp_1a1t_c(){

char name[NAMELEN]="gmove_cp_1a1t_c";
int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

 int i,/*j,*/ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[3:4]=a[0:4];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=3;i<7;i++){
      err=err+fabs(b[i]-(i-2));
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_cp_1a1t_gb(){

char name[NAMELEN]="gmove_cp_1a1t_gb";
int n=8;
double a[n],b[n];
int mx[4]={2,6};
int my[4]={6,2};
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(gblock(mx)) onto p
#pragma xmp distribute ty(gblock(my)) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with ty(i)

 int i,/*j,*/ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[:]=a[:];

  err=0.0;
#pragma xmp loop (i) on ty(i)
  for(i=1;i<n;i++){
      err=err+fabs(b[i]-i-1);
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_cp_1a1t_b();
  gmove_cp_1a1t_b_c();
  gmove_cp_1a1t_bc();
  gmove_cp_1a1t_c();
  gmove_cp_1a1t_gb();

  return 0;

}
