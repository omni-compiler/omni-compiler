#include <stdio.h>
#include <math.h>
#include <xmp.h>

int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n-1,0:n-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with tx(i,j)

int main(){

  int i,j;
  int myrank;
  double err;

  myrank=xmp_all_node_num();

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
  b[3:4][3:4]=a[0:4][0:4];

  err=0.0;
#pragma xmp loop (i,j) on tx(i,j)
  for(i=3;i<7;i++){
    for(j=3;j<7;j++){
      printf("i=%d,j=%d,b=%f,myrank=%d\n",i,j,b[i][j],myrank);
      err=err+fabs(b[i][j]-(i-2+j-2));
    }
  }

  printf("max error=%f\n",err);

  return 0;

}

