#define NAMELEN 25
#include <stdio.h>
#include <math.h>

extern int chk_int(char name[], int ierr);

void gmove_ata_2a1t_b2(){

char name[NAMELEN]="gmove_ata_2a1t_b2";
int n=10;
double a[n][n],b[n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][i] with tx(i)

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-(i+j+2));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_ata_2a1t_b(){

char name[NAMELEN]="gmove_ata_2a1t_b";
int n=9;
double a[n][n],b[n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][i] with tx(i)

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-(i+j+2));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_ata_2a1t_b_c(){

char name[NAMELEN]="gmove_ata_2a1t_b_c";
int n=9;
double a[n][n],b[n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][i] with ty(i)

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
  for(i=0;i<n;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-(i+j+2));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name, ierr);

}

void gmove_ata_2a1t_b_h(){

char name[NAMELEN]="gmove_ata_2a1t_b_h";
int n=9;
double a[n][n],b[n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][i] with tx(i)

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-a[i][j]);
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_2a1t_c(){

char name[NAMELEN]="gmove_ata_2a1t_c";
int n=8;
double a[n][n],b[n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][i] with tx(i)

  int i,j,ierr;
  double err;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i][j]=i+j+2;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      b[i][j]=0;
    }
  }

#pragma xmp gmove
  b[0:n][0:n]=a[0:n][0:n];

  err=0.0;
  for(i=0;i<n;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n;j++){
      err=err+fabs(b[i][j]-(i+j+2));
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_3a1t_b2(){

char name[NAMELEN]="gmove_ata_3a1t_b2";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

  int i,j,k,ierr;
  double err;

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
  b[:][:][:]=a[:][:][:];

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

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_3a1t_b(){

char name[NAMELEN]="gmove_ata_3a1t_b";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

  int i,j,k,ierr;
  double err;

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

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_3a1t_b_c(){

char name[NAMELEN]="gmove_ata_3a1t_b_c";
int n=9;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[*][*][i] with tx(i)
#pragma xmp align b[*][i][*] with ty(i)

  int i0,i1,i2,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
#pragma xmp loop (i2) on tx(i2)
      for(i2=0;i2<n;i2++){
        a[i0][i1][i2]=i0+i1+i2+2;
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        b[i0][i1][i2]=0;
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n]=a[0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        err=err+fabs(b[i0][i1][i2]-(i0+i1+i2+2));
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_3a1t_b_h(){

char name[NAMELEN]="gmove_ata_3a1t_b_h";
int n=9;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

  int i,j,k,ierr;
  double err;

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

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_3a1t_c(){

char name[NAMELEN]="gmove_ata_3a1t_c";
int n=8;
double a[n][n][n],b[n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp distribute tx(cyclic) onto p
#pragma xmp align a[i][*][*] with tx(i)
#pragma xmp align b[*][*][i] with tx(i)

  int i,j,k,ierr;
  double err;

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
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
#pragma xmp loop (k) on tx(k)
      for(k=0;k<n;k++){
        err=err+fabs(b[i][j][k]-(i+j+k+2));
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_4a1t_b(){

char name[NAMELEN]="gmove_ata_4a1t_b";
int n=10;
double a[n][n][n][n],b[n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(block) onto p
#pragma xmp align a[*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*] with ty(i)

  int i0,i1,i2,i3,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
#pragma xmp loop (i3) on tx(i3)
        for(i3=0;i3<n;i3++){
          a[i0][i1][i2][i3]=i0+i1+i2+i3+2;
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          err=err+fabs(b[i0][i1][i2][i3]-(i0+i1+i2+i3+2));
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_4a1t_b_c(){

char name[NAMELEN]="gmove_ata_4a1t_b_c";
int n=9;
double a[n][n][n][n],b[n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*] with ty(i)

  int i0,i1,i2,i3,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
#pragma xmp loop (i3) on tx(i3)
        for(i3=0;i3<n;i3++){
          a[i0][i1][i2][i3]=i0+i1+i2+i3+2;
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          b[i0][i1][i2][i3]=0;
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          err=err+fabs(b[i0][i1][i2][i3]-(i0+i1+i2+i3+2));
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_5a1t_b(){

char name[NAMELEN]="gmove_ata_5a1t_b";
int n=10;
double a[n][n][n][n][n],b[n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(block) onto p
#pragma xmp align a[*][*][*][i][*] with tx(i)
#pragma xmp align b[*][i][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
#pragma xmp loop (i3) on tx(i3)
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            a[i0][i1][i2][i3][i4]=i0+i1+i2+i3+i4+2;
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            b[i0][i1][i2][i3][i4]=0;
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            err=err+fabs(b[i0][i1][i2][i3][i4]-(i0+i1+i2+i3+i4+2));
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_5a1t_b_c(){

char name[NAMELEN]="gmove_ata_5a1t_b_c";
int n=5;
double a[n][n][n][n][n],b[n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[*][*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
#pragma xmp loop (i4) on tx(i4)
          for(i4=0;i4<n;i4++){
            a[i0][i1][i2][i3][i4]=i0+i1+i2+i3+i4+2;
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            b[i0][i1][i2][i3][i4]=0;
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            err=err+fabs(b[i0][i1][i2][i3][i4]-(i0+i1+i2+i3+i4+2));
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_6a1t_b(){

char name[NAMELEN]="gmove_ata_6a1t_b";
int n=10;
double a[n][n][n][n][n][n],b[n][n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(block) onto p
#pragma xmp align a[*][*][*][*][i][*] with tx(i)
#pragma xmp align b[*][i][*][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,i5,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
#pragma xmp loop (i4) on tx(i4)
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              a[i0][i1][i2][i3][i4][i5]=i0+i1+i2+i3+i4+i5+2;
            }
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              b[i0][i1][i2][i3][i4][i5]=0;
            }
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              err=err+fabs(b[i0][i1][i2][i3][i4][i5]-(i0+i1+i2+i3+i4+i5+2));
            }
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_6a1t_b_c(){

char name[NAMELEN]="gmove_ata_6a1t_b_c";
int n=5;
double a[n][n][n][n][n][n],b[n][n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[*][*][*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,i5,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
#pragma xmp loop (i5) on tx(i5)
            for(i5=0;i5<n;i5++){
              a[i0][i1][i2][i3][i4][i5]=i0+i1+i2+i3+i4+i5+2;
            }
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              b[i0][i1][i2][i3][i4][i5]=0;
            }
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              err=err+fabs(b[i0][i1][i2][i3][i4][i5]-(i0+i1+i2+i3+i4+i5+2));
            }
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_7a1t_b(){

char name[NAMELEN]="gmove_ata_7a1t_b";
int n=10;
double a[n][n][n][n][n][n][n],b[n][n][n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(block) onto p
#pragma xmp align a[*][*][*][*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,i5,i6,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
#pragma xmp loop (i6) on tx(i6)
              for(i6=0;i6<n;i6++){
                a[i0][i1][i2][i3][i4][i5][i6]=i0+i1+i2+i3+i4+i5+i6+2;
              }
            }
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              for(i6=0;i6<n;i6++){
                b[i0][i1][i2][i3][i4][i5][i6]=0;
              }
            }
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              for(i6=0;i6<n;i6++){
                err=err+fabs(b[i0][i1][i2][i3][i4][i5][i6]-(i0+i1+i2+i3+i4+i5+i6+2));
              }
            }
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}

void gmove_ata_7a1t_b_c(){

char name[NAMELEN]="gmove_ata_7a1t_b_c";
int n=5;
double a[n][n][n][n][n][n][n],b[n][n][n][n][n][n][n];
#pragma xmp nodes p(2)
#pragma xmp template tx(0:n-1)
#pragma xmp template ty(0:n-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp align a[*][*][*][*][*][*][i] with tx(i)
#pragma xmp align b[*][i][*][*][*][*][*] with ty(i)

  int i0,i1,i2,i3,i4,i5,i6,ierr;
  double err;

  for(i0=0;i0<n;i0++){
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
#pragma xmp loop (i6) on tx(i6)
              for(i6=0;i6<n;i6++){
                a[i0][i1][i2][i3][i4][i5][i6]=i0+i1+i2+i3+i4+i5+i6+2;
              }
            }
          }
        }
      }
    }
  }

  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              for(i6=0;i6<n;i6++){
                b[i0][i1][i2][i3][i4][i5][i6]=0;
              }
            }
          }
        }
      }
    }
  }

#pragma xmp gmove
  b[0:n][0:n][0:n][0:n][0:n][0:n][0:n]=a[0:n][0:n][0:n][0:n][0:n][0:n][0:n];

  err=0.0;
  for(i0=0;i0<n;i0++){
#pragma xmp loop (i1) on ty(i1)
    for(i1=0;i1<n;i1++){
      for(i2=0;i2<n;i2++){
        for(i3=0;i3<n;i3++){
          for(i4=0;i4<n;i4++){
            for(i5=0;i5<n;i5++){
              for(i6=0;i6<n;i6++){
                err=err+fabs(b[i0][i1][i2][i3][i4][i5][i6]-(i0+i1+i2+i3+i4+i5+i6+2));
              }
            }
          }
        }
      }
    }
  }

#pragma xmp reduction (MAX:err)
  ierr=err;
  chk_int(name,ierr);

}


int main(){

  gmove_ata_2a1t_b2();
  gmove_ata_2a1t_b();
  gmove_ata_2a1t_b_c();
  gmove_ata_2a1t_b_h();
  gmove_ata_2a1t_c();
  gmove_ata_3a1t_b2();
  gmove_ata_3a1t_b();
  gmove_ata_3a1t_b_c();
  gmove_ata_3a1t_b_h();
  gmove_ata_3a1t_c();
  gmove_ata_4a1t_b();
  gmove_ata_4a1t_b_c();
  gmove_ata_5a1t_b();
  gmove_ata_5a1t_b_c();
  gmove_ata_6a1t_b();
  gmove_ata_6a1t_b_c();
  gmove_ata_7a1t_b();
  gmove_ata_7a1t_b_c();

  return 0;

}
