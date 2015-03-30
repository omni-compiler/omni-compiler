#define NAMELEN 25
#define NMAX 4
#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

extern int chk_int(char name[], int ierr);

char name[NAMELEN]="337";
int n=NMAX;
int a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i0][i1] with tx(i1,i0)
#pragma xmp align b[*][i1] with tx(i1,*)

int main(){

  int i0,i1,ierr;//,irank;
  //  irank=xmp_node_num();

#pragma xmp loop (i0,i1) on tx(i1,i0)
  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      a[i0][i1]=i0+i1+1;
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on tx(i1,*)
    for(i1=0;i1<n;i1++){
      b[i0][i1]=0;
    }
  }

#pragma xmp gmove
  b[0:2][0:n]=a[0:2][0:n];

  ierr=0;
  for(i0=0;i0<2;i0++){
#pragma xmp loop (i1) on tx(i1,*)
    for(i1=0;i1<n;i1++){
      ierr=ierr+abs(b[i0][i1]-i0-i1-1);
    }
  }

#pragma xmp reduction (MAX:ierr)
  chk_int(name, ierr);

  return 0;
}
