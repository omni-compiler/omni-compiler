#include <stdio.h>
#include <math.h>

extern int chk_int(int ierr);
extern int chk_int2(int ierr);
extern void xmp_matmul(void *x_p, void *a_p, void *b_p);

#pragma xmp nodes p(4)

void test_mm_aaa_b_b_b_r8(){

int n1=21,n2=23, n3=25;
double a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(block,block) onto p
#pragma xmp distribute tz(block,block) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_bc_bc_bc_r8(){

int n1=21,n2=23, n3=25;
double a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(cyclic(2),cyclic(2)) onto p
#pragma xmp distribute ty(cyclic(2),cyclic(2)) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_bc_c16(){

int n1=21,n2=23, n3=25;
double _Complex a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_bc_c8(){

int n1=21,n2=23, n3=25;
float _Complex a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);
}


void test_mm_aaa_b_c_bc_r4(){

int n1=21,n2=23, n3=25;
float a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_bc_r8_ax(){

int n1=21,n2=23, n3=25;
double a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n2,0:n1)
#pragma xmp template ty(0:n3,0:n2)
#pragma xmp template tz(0:n3,0:n1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(j,i)
#pragma xmp align b[i][j] with ty(j,i)
#pragma xmp align x[i][j] with tz(j,i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(j,i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(j,i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(j,i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(j,i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_bc_r8(){

int n1=21,n2=23, n3=25;
double a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aaa_b_c_gb_r8(){

int n1=21,n2=23, n3=25;
int m1[2]={10,11},m2[2]={11,14};
double a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(gblock(m1),gblock(m2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_aani_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_aanj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_ania_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_anini_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_aninj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_anja_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_anjni_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_anjnj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1,0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block,block) onto p
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][j] with tx(i,j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i,j) on tx(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_niaa_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_niani_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_nianj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_ninia_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_ninini_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_nininj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_ninja_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_ninjni_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_ninjnj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n2-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[*][j] with tx(j)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tx(j)
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njaa_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njani_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_njanj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1,0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic,cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto q
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][j] with ty(i,j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i,j) on ty(i,j)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

#pragma xmp task on q
{
  chk_int2(ierr);
}

}


void test_mm_njnia_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njnini_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njninj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n3-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[*][j] with ty(j)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

  for(i=0;i<n2;i++){
#pragma xmp loop (j) on ty(j)
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njnja_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(2,2)
#pragma xmp nodes q(2)=p(1:2,1)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1,0:n3-1)
#pragma xmp distribute tx(block) onto q
#pragma xmp distribute ty(cyclic) onto q
#pragma xmp distribute tz(cyclic(2),cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][j] with tz(i,j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i,j) on tz(i,j)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njnjni_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n3-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[*][j] with tz(j)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

  for(i=0;i<n1;i++){
#pragma xmp loop (j) on tz(j)
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


void test_mm_njnjnj_b_c_bc_i4(){

int n1=21,n2=23, n3=25;
int a[n1][n2],b[n2][n3],x[n1][n3];
#pragma xmp nodes p(*)
#pragma xmp template tx(0:n1-1)
#pragma xmp template ty(0:n2-1)
#pragma xmp template tz(0:n1-1)
#pragma xmp distribute tx(block) onto p
#pragma xmp distribute ty(cyclic) onto p
#pragma xmp distribute tz(cyclic(2)) onto p
#pragma xmp align a[i][*] with tx(i)
#pragma xmp align b[i][*] with ty(i)
#pragma xmp align x[i][*] with tz(i)

 int i,j,/*k,*/ierr;
  double rn1,rn2,rn3,rn4,rn5,rn6,ra,rb;

#pragma xmp loop (i) on tx(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n2;j++){
      a[i][j]=i*n1+j+1;
    }
  }

#pragma xmp loop (i) on ty(i)
  for(i=0;i<n2;i++){
    for(j=0;j<n3;j++){
      b[i][j]=j*n3+i+1;
    }
  }

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      x[i][j]=0;
    }
  }

  xmp_matmul(xmp_desc_of(x),xmp_desc_of(a),xmp_desc_of(b));

  ierr=0;

  rn1=n1;
  rn2=n2;
  rn3=n3;
  rn4=rn2*(rn2+1)*(2*rn2+1)/6.0;
  rn5=rn2*(rn2+1)/2.0;

#pragma xmp loop (i) on tz(i)
  for(i=0;i<n1;i++){
    for(j=0;j<n3;j++){
      ra=i*rn1;
      rb=j*rn3;
      rn6=rn4+rn5*(ra+rb)+ra*rb*rn2;
      ierr=ierr+(x[i][j]-rn6);
    }
  }

  chk_int(ierr);

}


int main(){

  test_mm_aaa_b_b_b_r8();
  test_mm_aaa_bc_bc_bc_r8();
  test_mm_aaa_b_c_bc_c16();
  test_mm_aaa_b_c_bc_c8();
  test_mm_aaa_b_c_bc_i4();
  test_mm_aaa_b_c_bc_r4();
  test_mm_aaa_b_c_bc_r8_ax();
  test_mm_aaa_b_c_bc_r8();
  test_mm_aaa_b_c_gb_r8();
  test_mm_aani_b_c_bc_i4();
  test_mm_aanj_b_c_bc_i4();
  test_mm_ania_b_c_bc_i4();
  test_mm_anini_b_c_bc_i4();
  test_mm_aninj_b_c_bc_i4();
  test_mm_anja_b_c_bc_i4();
  test_mm_anjni_b_c_bc_i4();
  test_mm_anjnj_b_c_bc_i4();
  test_mm_niaa_b_c_bc_i4();
  test_mm_niani_b_c_bc_i4();
  test_mm_nianj_b_c_bc_i4();
  test_mm_ninia_b_c_bc_i4();
  test_mm_ninini_b_c_bc_i4();
  test_mm_nininj_b_c_bc_i4();
  test_mm_ninja_b_c_bc_i4();
  test_mm_ninjni_b_c_bc_i4();
  test_mm_ninjnj_b_c_bc_i4();
  test_mm_njaa_b_c_bc_i4();
  test_mm_njani_b_c_bc_i4();
  test_mm_njanj_b_c_bc_i4();
  test_mm_njnia_b_c_bc_i4();
  test_mm_njnini_b_c_bc_i4();
  test_mm_njninj_b_c_bc_i4();
  test_mm_njnja_b_c_bc_i4();
  test_mm_njnjni_b_c_bc_i4();
  test_mm_njnjnj_b_c_bc_i4();

  return 0;

}
