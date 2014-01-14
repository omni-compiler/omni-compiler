#include <stdio.h>
#include <xmp.h>

int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][*] with tx(i,*)

int main(){

  int i,j,ierr;
  int myrank;

  myrank=xmp_all_node_num();

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

#pragma xmp loop (i) on tx(i,*)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[4:2][4:1]=a[4:2][4:1];

  ierr=0;
#pragma xmp loop (i) on tx(i,*)
  for(i=4;i<5;i++){
    for(j=4;j<5;j++){
      printf("i=%d,j=%d,b=%f,myrank=%d\n",i,j,b[i][j],myrank);
    }
  }

  printf("max error=%d\n",ierr);

  return 0;

}

