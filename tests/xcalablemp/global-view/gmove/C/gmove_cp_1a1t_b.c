#include <stdio.h>
#include <math.h>
#include <xmp.h>

int n=8;
double a[n],b[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i] with tx(i)
#pragma xmp align b[i] with tx(i)

int main(){

  int i,j,ierr;
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
  b[1:4]=a[4:4];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=1;i<5;i++){
      err=err+fabs(b[i]-(i+4));
  }

#pragma xmp reduction (MAX:err)
  if (myrank ==1){
    printf("max error=%f\n",err);
  }
  ierr=err;

  return ierr;

}

