#include <stdio.h>
#include <math.h>
#include <xmp.h>

int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

int main(){

  int i,j;
  int myrank;
  double err;

  myrank=xmp_all_node_num();

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

  printf("max error=%f,myrank=%d\n",err,myrank);

  return 0;

}

