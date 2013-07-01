#include <stdio.h>
#include <xmp.h>

int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

int main(){

  int i,j,k;
  int myrank;
  double err;

  myrank=xmp_all_node_num();

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+2;
      }
    }
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
#pragma xmp loop (k) on tx(k)
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
#pragma xmp loop (k) on tx(k)
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

  printf("max error=%f\n",err);

  return 0;

}

