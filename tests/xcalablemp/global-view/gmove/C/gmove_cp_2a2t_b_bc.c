#include <stdio.h>
#include <stdlib.h>
#include <xmp.h>

int n=8;
int a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i0][i1] with tx(i0,i1)
#pragma xmp align b[i0][i1] with ty(i0,i1)

int main(){

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
//      printf("i0=%d,b=%d\n",i0,b[i0][i1]);
    }
  }

  int irank= xmp_node_num();
#pragma xmp reduction (MAX:ierr)
  if (irank == 1){
    printf("max error=%d\n",ierr);
  }
  return ierr;

}
