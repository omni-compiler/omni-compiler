#define NAMELEN 25
#define N 8
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_G2L_1a1t_bc(){

char name[NAMELEN]="gmove_G2L_1a1t_bc";
int n=N;
double a[n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic(2)) onto p
#pragma xmp align a[i] with tx(i)

 int i,/*j,*/ierr;
  double b[N],err;

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
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_G2L_1a1t_bc();

  return 0;

}
