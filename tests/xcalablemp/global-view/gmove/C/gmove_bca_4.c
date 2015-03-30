#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_bca_1a2t_b2(){

char name[NAMELEN]="gmove_bca_1a2t_b2";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i0] with tx(i0,*)
#pragma xmp align b[i0] with tx(i0,*)

 int i0,/*i1,*/ierr;

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    a[i0]=i0+1;
  }

#pragma xmp loop (i0) on tx(i0,*)
  for(i0=0;i0<n;i0++){
    b[i0]=0;
  }

#pragma xmp gmove
  b[1:n-1]=a[1:n-1];

  ierr=0;
#pragma xmp loop (i0) on tx(i0,*)
  for(i0=1;i0<n;i0++){
    ierr=ierr+abs(b[i0]-i0-1);
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

void gmove_bca_1a2t_b(){

char name[NAMELEN]="gmove_bca_1a2t_b";
int n=8;
int a[n],b[n];
#pragma xmp nodes p(2,2)
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

void gmove_bca_2a3t_b2(){

char name[NAMELEN]="gmove_bca_2a3t_b2";
int n=4;
int a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,*,block) onto p
#pragma xmp distribute ty(*,block,block) onto p
#pragma xmp align a[i0][*] with tx(i0,*,*)
#pragma xmp align b[*][i2] with ty(*,*,i2)

  int i0,i1,i2,ierr;

#pragma xmp loop (i0) on tx(i0,*,*)
  for(i0=0;i0<n;i0++){
    for(i2=0;i2<n;i2++){
      a[i0][i2]=i0+i2+1;
    }
  }

  for(i1=0;i1<n;i1++){
#pragma xmp loop (i2) on ty(*,*,i2)
    for(i2=0;i2<n;i2++){
      b[i1][i2]=0;
    }
  }

#pragma xmp gmove
  b[:][:]=a[:][:];

  ierr=0;
  for(i1=0;i1<n;i1++){
#pragma xmp loop (i2) on ty(*,*,i2)
    for(i2=0;i2<n;i2++){
      ierr=ierr+abs(b[i1][i2]-i1-i2-1);
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

}

int main(){

  gmove_bca_1a2t_b2();
  gmove_bca_1a2t_b();
  gmove_bca_2a3t_b2();

  return 0;

}
