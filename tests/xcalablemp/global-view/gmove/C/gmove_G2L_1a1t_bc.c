#include <stdio.h>
#include <math.h>
#include <xmp.h>

int n=8;
double a[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic(2)) onto p
#pragma xmp align a[i] with tx(i)

int main(){

  int i,j,ierr;
  int myrank;
  double b[n],err;

  myrank=xmp_all_node_num();

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    a[i]=i+1;
  }

  for(i=0;i<n;i++){
    b[i]=0;
  }

#pragma xmp gmove
  b[4:4]=a[0:4];

  err=0.0;
  for(i=4;i<8;i++){
    err=err+fabs(b[i]-(i-3));
  }

#pragma xmp reduction (MAX:err)
  if (myrank ==1){
    printf("max error=%f\n",err);
  }
  ierr=err;

  return ierr;

}

