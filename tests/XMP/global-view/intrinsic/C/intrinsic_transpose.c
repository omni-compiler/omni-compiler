#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

extern int chk_int(int ierr);
extern int chk_int2(int ierr);
extern void xmp_transpose(void *dst_d, void *src_d, int opt);
#pragma xmp nodes p[4]

void test_tr_a2a_bb_c16_1(){

int m=21, n=25;
double _Complex a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bb_c8_1(){

int m=21, n=25;
float _Complex a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bb_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bb_r4_1(){

int m=21, n=25;
float a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bb_r8_1(){

int m=21, n=25;
double a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bc_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[m]
#pragma xmp template ty[n]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic] onto p
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[i][*] with ty[i]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_a2a_bc_i4_3(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[*][j] with tx[j][*]
#pragma xmp align b[*][j] with ty[j][*]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j][*]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on p[0:2][0]
{
  chk_int2(ierr);
}

}

void test_tr_a2a_bc_i4_4(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][*] with tx[*][i]
#pragma xmp align b[i][*] with ty[*][i]

  int i,j,ierr;
#pragma xmp loop on tx[*][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on p[0][0:2]
{
  chk_int2(ierr);
}

}

void test_tr_bca_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2]=p[0][0:2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}

void test_tr_bca_bc_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2]=p[0][0:2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][*] with ty[i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}

void test_tr_bca_bc_i4_3(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[*][j] with ty[j][*]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on q[0:2][0]
{
  chk_int2(ierr);
}

}

void test_tr_bca_bc_i4_4(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][*] with ty[*][i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on q[0][0:2]
{
  chk_int2(ierr);
}

}

void test_tr_cp0_bc_c16_1(){

int m=21, n=25;
double _Complex a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cp0_bc_c8_1(){

int m=21, n=25;
float _Complex a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cp0_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cp_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[m]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic] onto p
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cp_bc_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[4]
#pragma xmp template tx[n]
#pragma xmp template ty[n]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[i][*] with ty[i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cp_bc_i4_3(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][*] with tx[*][i]
#pragma xmp align b[*][j] with ty[j][*]

  int i,j,ierr;
#pragma xmp loop on tx[*][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j][*]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on p[0:2][0]
{
  chk_int2(ierr);
}

}

void test_tr_cp_bc_i4_4(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[*][j] with tx[j][*]
#pragma xmp align b[i][*] with ty[*][i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j][*]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[*][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

#pragma xmp task on p[0][0:2]
{
  chk_int2(ierr);
}

}

void test_tr_cp_bc_r4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[m]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic] onto p
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+fabs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cps_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[4]
#pragma xmp template tx[n][m]
#pragma xmp template ty[n]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[i][*] with ty[i]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cps_bc_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[4]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block][block] onto p
#pragma xmp distribute ty[cyclic] onto q
#pragma xmp align a[i][j] with tx[j][i]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
#pragma xmp loop on tx[j][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cps_bc_i4_3(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[4]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto q
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_cps_bc_i4_4(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[4]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[n]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[cyclic][cyclic] onto q
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_lc_bb_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[m]
#pragma xmp template ty[m]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[*][j] with ty[j]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
  for(i=0;i<n;i++){
#pragma xmp loop on ty[j]
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_lc_bb_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[*]
#pragma xmp template tx[n]
#pragma xmp template ty[n]
#pragma xmp distribute tx[block] onto p
#pragma xmp distribute ty[block] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[i][*] with ty[i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_proj_bc_i4_1(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2]=p[0][0:2]
#pragma xmp template tx[n]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block] onto q
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[*][j] with tx[j]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_proj_bc_i4_2(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2]=p[0][0:2]
#pragma xmp template tx[m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block] onto q
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][*] with tx[i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_proj_bc_i4_3(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto q
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[*][j] with tx[j][*]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
  for(i=0;i<m;i++){
#pragma xmp loop on tx[j][*]
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

void test_tr_proj_bc_i4_4(){

int m=21, n=25;
int a[m][n],b[n][m];
#pragma xmp nodes p[2][2]
#pragma xmp nodes q[2][2]
#pragma xmp template tx[n][m]
#pragma xmp template ty[m][n]
#pragma xmp distribute tx[block][block] onto q
#pragma xmp distribute ty[cyclic][cyclic] onto p
#pragma xmp align a[i][*] with tx[*][i]
#pragma xmp align b[i][j] with ty[j][i]

  int i,j,ierr;
#pragma xmp loop on tx[*][i]
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      a[i][j]=i*m+j+1;
    }
  }

#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      b[i][j]=0;
    }
  }

  xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 1);

  ierr=0;
#pragma xmp loop on ty[j][i]
  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      ierr=ierr+abs(b[i][j]-(j*m+i+1));
    }
  }

  chk_int(ierr);

}

#include "mpi.h"

int main(){

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_tr_a2a_bb_c16_1();
  test_tr_a2a_bb_c8_1();
#endif

  test_tr_a2a_bb_i4_1();
  test_tr_a2a_bb_r4_1();
  test_tr_a2a_bb_r8_1();
  test_tr_a2a_bc_i4_1();
  test_tr_a2a_bc_i4_2();
  test_tr_a2a_bc_i4_3();
  test_tr_a2a_bc_i4_4();
  test_tr_bca_bc_i4_1();
  test_tr_bca_bc_i4_2();
  test_tr_bca_bc_i4_3();
  test_tr_bca_bc_i4_4();

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_tr_cp0_bc_c16_1();
  test_tr_cp0_bc_c8_1();
#endif

  test_tr_cp0_bc_i4_1();
  test_tr_cp_bc_i4_1();
  test_tr_cp_bc_i4_2();
  test_tr_cp_bc_i4_3();
  test_tr_cp_bc_i4_4();
  test_tr_cp_bc_r4_1();
  test_tr_cps_bc_i4_1();
  test_tr_cps_bc_i4_2();
  test_tr_cps_bc_i4_3();
  test_tr_cps_bc_i4_4();
  test_tr_lc_bb_i4_1();
  test_tr_lc_bb_i4_2();
  test_tr_proj_bc_i4_1();
  test_tr_proj_bc_i4_2();
  test_tr_proj_bc_i4_3();
  test_tr_proj_bc_i4_4();

  return 0;

}
