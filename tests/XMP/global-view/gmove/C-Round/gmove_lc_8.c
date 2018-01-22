#define NAMELEN 25
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_lc_32a3t_b2(){

char name[NAMELEN]="gmove_lc_32a3t_b2";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0]=a[0][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<1;k++){
        err=err+fabs(b[i][j][k]-i-j-3);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_32a3t_b(){

char name[NAMELEN]="gmove_lc_32a3t_b";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0][0:n][0:n]=a[0:n][0][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<1;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-j-k-3);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_b(){

char name[NAMELEN]="gmove_lc_3a3t_b";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_bc(){

char name[NAMELEN]="gmove_lc_3a3t_bc";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_b_h(){

char name[NAMELEN]="gmove_lc_3a3t_b_h";
int n=9;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(block,block,block) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_c(){

char name[NAMELEN]="gmove_lc_3a3t_c";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic,cyclic) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_c_h(){

char name[NAMELEN]="gmove_lc_3a3t_c_h";
int n=9;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(cyclic,cyclic,cyclic) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with tx(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-a[i][j][k]);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_lc_3a3t_gb(){

char name[NAMELEN]="gmove_lc_3a3t_gb";
int n=8;
double a[n][n][n],b[n][n][n];
int mx[2]={2,6};
int my[2]={2,6};
#pragma xmp nodes p(2,2,2)
#pragma xmp template tx(0:n-1,0:n-1,0:n-1)
#pragma xmp template ty(0:n-1,0:n-1,0:n-1)
#pragma xmp distribute tx(gblock(mx),gblock(mx),gblock(mx)) onto p
#pragma xmp distribute ty(gblock(my),gblock(my),gblock(my)) onto p
#pragma xmp align a[i][j][k] with tx(i,j,k)
#pragma xmp align b[i][j][k] with ty(i,j,k)

  int i,j,k,ierr;
  double err ;

#pragma xmp loop (i,j,k) on tx(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        a[i][j][k]=i+j+k+3;
      }
    }
  }

#pragma xmp loop (i,j,k) on ty(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        b[i][j][k]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
#pragma xmp loop (i,j,k) on ty(i,j,k)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-i-j-k-3);
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

int main(){

  gmove_lc_32a3t_b2();
  gmove_lc_32a3t_b();
  gmove_lc_3a3t_b();
  gmove_lc_3a3t_bc();
  gmove_lc_3a3t_b_h();
  gmove_lc_3a3t_c();
  gmove_lc_3a3t_c_h();
  gmove_lc_3a3t_gb();

  return 0;

}
