#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_bca_1a2t_b3(){

char name[NAMELEN]="gmove_bca_1a2t_b3";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(2,1)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i0] with tx(i0,*)
#pragma xmp align b[i1] with tx(*,i1)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i1) on tx(*,i1)
  for(i1=0;i1<n;i1++){
    b[i1]=0;
  }

#pragma xmp gmove
  b[1:n-1]=a[1:n-1];

  ierr=0;
#pragma xmp loop (i1) on tx(*,i1)
  for(i1=1;i1<n;i1++){
    ierr=ierr+abs(b[i1]-i1-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_1a2t_b4(){

char name[NAMELEN]="gmove_bca_1a2t_b4";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(1,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i0] with tx(i0,*)
#pragma xmp align b[i1] with tx(*,i1)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i1) on tx(*,i1)
  for(i1=0;i1<n;i1++){
    b[i1]=0;
  }

#pragma xmp gmove
  b[1:n-1]=a[1:n-1];

  ierr=0;
#pragma xmp loop (i1) on tx(*,i1)
  for(i1=1;i1<n;i1++){
    ierr=ierr+abs(b[i1]-i1-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_1a2t_b5(){

char name[NAMELEN]="gmove_bca_1a2t_b5";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(1,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i1] with tx(*,i1)
#pragma xmp align b[i0] with tx(i0,*)

  int i0,i1,ierr;

#pragma xmp loop (i1) on tx(*,i1)
  for(i1=0;i1<n;i1++){
    a[i1]=i1+1;
  }

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[:]=a[:];

  ierr=0;
#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    ierr=ierr+abs(b[i0]-i0-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_1a2t_b6(){

char name[NAMELEN]="gmove_bca_1a2t_b6";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(block,*) onto p
#pragma xmp distribute ty(*,block) onto p
#pragma xmp align a[i0] with tx(i0,*)
#pragma xmp align b[i1] with ty(*,i1)

  int i0,i1,ierr;

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i1) on ty(*,i1)
  for(i1=0;i1<n;i1++){
    b[i1]=0;
  }

#pragma xmp gmove
  b[:]=a[:];

  ierr=0;
#pragma xmp loop (i1) on ty(*,i1)
  for(i1=0;i1<n;i1++){
    ierr=ierr+abs(b[i1]-i1-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_bca_1a2t_b3();
  gmove_bca_1a2t_b4();
  gmove_bca_1a2t_b5();
  gmove_bca_1a2t_b6();

  return 0;

}
