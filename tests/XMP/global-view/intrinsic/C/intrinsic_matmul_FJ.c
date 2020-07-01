#include <stdio.h>
#include <xmp.h>
#include <math.h>
#include <complex.h>
extern int chk_int(int ierr);
extern void xmp_matmul(void *x_p, void *a_p, void *b_p);

int test_matmul_001(){

#pragma xmp nodes q[8]
#pragma xmp nodes p[2][2]=q[0:4]
#pragma xmp nodes pp[4]=q[0:4]
#pragma xmp template ta[8][16]
#pragma xmp template tb[16][8]
#pragma xmp template tx[16][16]
#pragma xmp distribute ta[block][block] onto p
#pragma xmp distribute tb[cyclic][cyclic] onto p
#pragma xmp distribute tx[cyclic(3)][cyclic(2)] onto p

   int i, j, k, error;
   int a[16][8], b[8][16], x[16][16];
   int c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i][j]
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = i+8*j;
      }
   }
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         c[j][i] = i+8*j;
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         b[i][j] = -1*(j+16*i);
      }
   }
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         d[i][j] = -1*(j+16*i);
      }
   }

   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         y[j][i] = 0;
         for(k=0; k<8; k++){
            y[j][i] += c[j][k]*d[k][i];
         }
      }
   }

   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j] reduction(+: error)
   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]) error++;
      }
   }

   chk_int(error);
   return 0;

}

int test_matmul_002(){

#pragma xmp nodes q[8]
#pragma xmp nodes p[2][2]=q[0:4]
#pragma xmp nodes pp[4]=q[0:4]
#pragma xmp template ta[8][16]
#pragma xmp template tb[16][8]
#pragma xmp template tx[16][16]
#pragma xmp distribute ta[block][block] onto p
#pragma xmp distribute tb[cyclic][cyclic] onto p
#pragma xmp distribute tx[cyclic(3)][cyclic(2)] onto p

   int i, j, k, error;
   double a[16][8], b[8][16], x[16][16];
   double c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i][j]
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = i+8*j;
      }
   }
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         c[j][i] = i+8*j;
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         b[i][j] = -1*(j+16*i);
      }
   }
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         d[i][j] = -1*(j+16*i);
      }
   }

   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         y[j][i] = 0;
         for(k=0; k<8; k++){
            y[j][i] += c[j][k]*d[k][i];
         }
      }
   }

   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j] reduction(+: error)
   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         if(fabs(x[j][i]-y[j][i]) > 0.00000001) error++;
         if(x[j][i] != y[j][i]) error++;
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_003(){

#pragma xmp nodes q[8]
#pragma xmp nodes p[2][2]=q[0:4]
#pragma xmp nodes pp[4]=q[0:4]
#pragma xmp template ta[8][16]
#pragma xmp template tb[16][8]
#pragma xmp template tx[16][16]
#pragma xmp distribute ta[cyclic(3)][cyclic(2)] onto p
#pragma xmp distribute tb[block][block] onto p
#pragma xmp distribute tx[cyclic][cyclic] onto p

   int i, j, k, error;
   int a[16][8], b[8][16], x[16][16];
   int c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i][j]
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = i+8*j;
      }
   }
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         c[j][i] = i+8*j;
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         b[i][j] = -1*(j+16*i);
      }
   }
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         d[i][j] = -1*(j+16*i);
      }
   }

   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         y[j][i] = 0;
         for(k=0; k<8; k++){
            y[j][i] += c[j][k]*d[k][i];
         }
      }
   }

   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j] reduction(+: error)
   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]) error++;
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_004(){

#pragma xmp nodes q[8]
#pragma xmp nodes p[2][2]=q[0:4]
#pragma xmp nodes pp[4]=q[0:4]
#pragma xmp template ta[8][20]
#pragma xmp template tb[16][8]
#pragma xmp template tx[16][20]
#pragma xmp distribute ta[block][block] onto p
#pragma xmp distribute tb[block][block] onto p
#pragma xmp distribute tx[block][block] onto p

   int i, j, k, error;
   int a[20][8], b[8][16], x[20][16];
   int c[20][8], d[8][16], y[20][16];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i][j]
   for(j=0; j<20; j++){
      for(i=0; i<8; i++){
         a[j][i] = i+8*j;
      }
   }
   for(j=0; j<20; j++){
      for(i=0; i<8; i++){
         c[j][i] = i+8*j;
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         b[i][j] = -1*(j+16*i);
      }
   }
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         d[i][j] = -1*(j+16*i);
      }
   }

   for(j=0; j<20; j++){
      for(i=0; i<16; i++){
         y[j][i] = 0;
         for(k=0; k<8; k++){
            y[j][i] += c[j][k]*d[k][i];
         }
      }
   }

   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j] reduction(+: error)
   for(j=0; j<20; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_005(){

#pragma xmp nodes q[8]
#pragma xmp nodes p[2][2]=q[0:4]
#pragma xmp nodes pp[4]=q[0:4]
#pragma xmp template ta[8][20]
#pragma xmp template tb[16][8]
#pragma xmp template tx[16][20]
#pragma xmp distribute ta[block][block] onto p
#pragma xmp distribute tb[block][block] onto p
#pragma xmp distribute tx[block][block] onto p

   int i, j, k, error;
   int a[20][8], b[8][16], x[20][16];
   int c[20][8], d[8][16], y[20][16];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[j][i]
#pragma xmp shadow a[1][1]
#pragma xmp shadow b[1][1]
#pragma xmp shadow x[1][1]

#pragma xmp loop on ta[i][j]
   for(j=0; j<20; j++){
      for(i=0; i<8; i++){
         a[j][i] = i+8*j;
      }
   }
   for(j=0; j<20; j++){
      for(i=0; i<8; i++){
         c[j][i] = i+8*j;
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         b[i][j] = -1*(j+16*i);
      }
   }
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         d[i][j] = -1*(j+16*i);
      }
   }

   for(j=0; j<20; j++){
      for(i=0; i<16; i++){
         y[j][i] = 0;
         for(k=0; k<8; k++){
            y[j][i] += c[j][k]*d[k][i];
         }
      }
   }

   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j] reduction(+: error)
   for(j=0; j<20; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_006(){

#define M 45
#define N 54
#define L 23

#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[8]
#pragma xmp template ta[L][M]
#pragma xmp template tb[L][L][N]
#pragma xmp template tx[N]
#pragma xmp distribute ta[block][block] onto pa
#pragma xmp distribute tb[cyclic][cyclic][cyclic] onto pb
#pragma xmp distribute tx[cyclic(3)] onto px

   int i, j, k, error;
   short a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[j][*]
#pragma xmp align b[*][j] with tb[*][*][j]
#pragma xmp align x[*][j] with tx[j]

#pragma xmp loop on ta[j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = (i*L+j)%11+1;
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = (i*L+j)%11+1;
      }
   }

#pragma xmp loop on tb[*][*][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = (i*N+j)%13+1;
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = (i*N+j)%13+1;
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_007(){

#define M 45
#define N 54
#define L 23

int m1[2]={10,13};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[4]=p[2:4]
#pragma xmp template ta[L][L][M]
#pragma xmp template tb[L][L][N]
#pragma xmp template tx[N]
#pragma xmp distribute ta[block][gblock(m1)][block] onto pa
#pragma xmp distribute tb[cyclic][cyclic][cyclic] onto pb
#pragma xmp distribute tx[cyclic(3)] onto px

   int i, j, k, error;
   int a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[*][j][*]
#pragma xmp align b[*][j] with tb[*][*][j]
#pragma xmp align x[*][j] with tx[j]

#pragma xmp loop on ta[*][j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = (i*L+j)%11+1;
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = (i*L+j)%11+1;
      }
   }

#pragma xmp loop on tb[*][*][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = (i*N+j)%13+1;
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = (i*N+j)%13+1;
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_008(){

#define M 45
#define N 54
#define L 23

int m1[2]={10,13};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[6]=p[1:6]
#pragma xmp nodes px[8]
#pragma xmp template ta[L][L][M]
#pragma xmp template tb[L]
#pragma xmp template tx[N]
#pragma xmp distribute ta[block][gblock(m1)][block] onto pa
#pragma xmp distribute tb[block] onto pb
#pragma xmp distribute tx[cyclic(3)] onto px

   int i, j, k, error;
   long long a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[*][j][*]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[*][j] with tx[j]

#pragma xmp loop on ta[*][j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = (i*L+j)%11+1;
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = (i*L+j)%11+1;
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = (i*N+j)%13+1;
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = (i*N+j)%13+1;
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_009(){

#define M 45
#define N 54
#define L 23

int m1[2]={10,13};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[8]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][L][M]
#pragma xmp template tb[L]
#pragma xmp template tx[N][1:L][M]
#pragma xmp distribute ta[block][gblock(m1)][block] onto pa
#pragma xmp distribute tb[block] onto pb
#pragma xmp distribute tx[block][block][block] onto px

   int i, j, k, error;
   float a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[*][j][*]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[i][j] with tx[j][*][i]

#pragma xmp loop on ta[*][j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = (i*L+j)%11+1;
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = (i*L+j)%11+1;
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = (i*N+j)%13+1;
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = (i*N+j)%13+1;
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][*][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_010(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={10,13};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2]=p[4:4]
#pragma xmp nodes pb[8]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][M]
#pragma xmp template tb[L]
#pragma xmp template tx[N][1:L][M]
#pragma xmp distribute ta[cyclic][block] onto pa
#pragma xmp distribute tb[block] onto pb
#pragma xmp distribute tx[block][block][block] onto px

   int i, j, k, error;
   double a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[i][j] with tx[j][*][i]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = (i*L+j)%11+1;
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = (i*L+j)%11+1;
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = (i*N+j)%13+1;
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = (i*N+j)%13+1;
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][*][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_011(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={10,13};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][M]
#pragma xmp template tb[N][L][L]
#pragma xmp template tx[N][1:L][M]
#pragma xmp distribute ta[cyclic][block] onto pa
#pragma xmp distribute tb[cyclic(2)][block][block] onto pb
#pragma xmp distribute tx[block][block][block] onto px

   int i, j, k, error;
   float _Complex a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i][*]
#pragma xmp align x[i][j] with tx[j][*][i]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }

#pragma xmp loop on tb[j][i][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][*][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_012(){

#define M 45
#define N 54
#define L 23

int m1[2]={30,24};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2][2]=p[0:4]
#pragma xmp template ta[L][M]
#pragma xmp template tb[N][L][L]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[cyclic][block] onto pa
#pragma xmp distribute tb[cyclic(2)][block][block] onto pb
#pragma xmp distribute tx[gblock(m1)][block] onto px

   int i, j, k, error;
   short a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[j][i][*]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[j][i][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_013(){

#define M 45
#define N 54
#define L 23

int m1[2]={30,24};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4]=p[0:4]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2][2]=p[4:4]
#pragma xmp template ta[M]
#pragma xmp template tb[N][L][L]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[cyclic] onto pa
#pragma xmp distribute tb[cyclic(2)][block][block] onto pb
#pragma xmp distribute tx[gblock(m1)][block] onto px

   int i, j, k, error;
   int a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][*] with ta[i]
#pragma xmp align b[i][j] with tb[j][i][*]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[j][i][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_014(){

#define M 45
#define N 54
#define L 23

int m1[2]={30,24};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[8]
#pragma xmp nodes pb[2][2]=p[2:4]
#pragma xmp nodes px[2][4]
#pragma xmp template ta[M]
#pragma xmp template tb[L][N]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[cyclic] onto pa
#pragma xmp distribute tb[cyclic][block] onto pb
#pragma xmp distribute tx[gblock(m1)][block] onto px

   int i, j, k, error;
   long long a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][*] with ta[i]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_015(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={30,24};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[8]
#pragma xmp nodes pb[4][2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[M]
#pragma xmp template tb[L][N]
#pragma xmp template tx[1:100][M][N]
#pragma xmp distribute ta[cyclic] onto pa
#pragma xmp distribute tb[cyclic][block] onto pb
#pragma xmp distribute tx[cyclic][cyclic][cyclic] onto px

   int i, j, k, error;
   float a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][*] with ta[i]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[*][i][j]

#pragma xmp loop on ta[i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][i][j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_016(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={30,24};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2]=p[4:4]
#pragma xmp nodes pb[4][2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][M]
#pragma xmp template tb[L][N]
#pragma xmp template tx[1:100][M][N]
#pragma xmp distribute ta[cyclic(3)][cyclic] onto pa
#pragma xmp distribute tb[cyclic][block] onto pb
#pragma xmp distribute tx[cyclic][cyclic][cyclic] onto px

   int i, j, k, error;
   double a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[*][i][j]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][i][j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_017(){

#define M 45
#define N 54
#define L 23

int m1[2]={14,40};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][M]
#pragma xmp template tb[L][N][1:100]
#pragma xmp template tx[1:100][M][N]
#pragma xmp distribute ta[cyclic(3)][cyclic] onto pa
#pragma xmp distribute tb[cyclic][gblock(m1)][block] onto pb
#pragma xmp distribute tx[cyclic][cyclic][cyclic] onto px

   int i, j, k, error;
   double _Complex a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[i][j][*]
#pragma xmp align x[i][j] with tx[*][i][j]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }

#pragma xmp loop on tb[i][j][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][i][j] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_018(){

#define M 45
#define N 54
#define L 23

int m1[2]={40,14};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2]=pb[1][0][:]
#pragma xmp template ta[L][M]
#pragma xmp template tb[L][N][1:100]
#pragma xmp template tx[M]
#pragma xmp distribute ta[cyclic(3)][cyclic] onto pa
#pragma xmp distribute tb[cyclic][gblock(m1)][block] onto pb
#pragma xmp distribute tx[cyclic(7)] onto px

   int i, j, k, error;
   short a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[j][i]
#pragma xmp align b[i][j] with tb[i][j][*]
#pragma xmp align x[i][*] with tx[i]

#pragma xmp loop on ta[j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_019(){

#define M 45
#define N 54
#define L 23

int m1[2]={40,14};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[8]
#pragma xmp template ta[M][1:100][L]
#pragma xmp template tb[L][N][1:100]
#pragma xmp template tx[M]
#pragma xmp distribute ta[cyclic(3)][cyclic][block] onto pa
#pragma xmp distribute tb[cyclic][gblock(m1)][block] onto pb
#pragma xmp distribute tx[cyclic(13)] onto px

   int i, j, k, error;
   int a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][*][j]
#pragma xmp align b[i][j] with tb[i][j][*]
#pragma xmp align x[i][*] with tx[i]

#pragma xmp loop on ta[i][*][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j][*]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_020(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={40,14};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][2]=pa[:][0][:]
#pragma xmp nodes px[8]
#pragma xmp template ta[M][1:100][L]
#pragma xmp template tb[N][L]
#pragma xmp template tx[M]
#pragma xmp distribute ta[cyclic(3)][cyclic][block] onto pa
#pragma xmp distribute tb[cyclic][cyclic(3)] onto pb
#pragma xmp distribute tx[cyclic(13)] onto px

   int i, j, k, error;
   long long a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][*][j]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][*] with tx[i]

#pragma xmp loop on ta[i][*][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_021(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={40,14};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][4]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[M][1:100][L]
#pragma xmp template tb[N][L]
#pragma xmp template tx[1:100][N][M]
#pragma xmp distribute ta[cyclic(3)][cyclic][block] onto pa
#pragma xmp distribute tb[cyclic][cyclic(13)] onto pb
#pragma xmp distribute tx[cyclic][cyclic(3)][cyclic(2)] onto px

   int i, j, k, error;
   float a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][*][j]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[*][j][i]

#pragma xmp loop on ta[i][*][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_022(){

#define M 45
#define N 54
#define L 23

int m1[2]={20,3};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][4]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[M][L]
#pragma xmp template tb[N][L]
#pragma xmp template tx[1:100][N][M]
#pragma xmp distribute ta[cyclic(14)][gblock(m1)] onto pa
#pragma xmp distribute tb[cyclic][cyclic(3)] onto pb
#pragma xmp distribute tx[cyclic][cyclic(3)][cyclic(2)] onto px

   int i, j, k, error;
   double a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][j]
#pragma xmp align b[i][j] with tb[j][i]
#pragma xmp align x[i][j] with tx[*][j][i]

#pragma xmp loop on ta[i][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[j][i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_023(){

#define M 45
#define N 54
#define L 23

int m1[2]={20,3};
int m2[2]={13,10};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2]=p[2:2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[M][L]
#pragma xmp template tb[L]
#pragma xmp template tx[1:100][N][M]
#pragma xmp distribute ta[cyclic(14)][gblock(m1)] onto pa
#pragma xmp distribute tb[gblock(m2)] onto pb
#pragma xmp distribute tx[cyclic][cyclic(3)][cyclic(2)] onto px

   int i, j, k, error;
   float _Complex a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][j]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[i][j] with tx[*][j][i]

#pragma xmp loop on ta[i][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[*][j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_024(){

#define M 45
#define N 54
#define L 23

int m1[2]={20,3};
int m2[2]={13,10};
int m3[4]={7,9,11,18};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2]=p[3:2]
#pragma xmp nodes px[2][4]
#pragma xmp template ta[M][L]
#pragma xmp template tb[L]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[cyclic(14)][gblock(m1)] onto pa
#pragma xmp distribute tb[gblock(m2)] onto pb
#pragma xmp distribute tx[block][gblock(m3)] onto px

   int i, j, k, error;
   short a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[i][j]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[i][j]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_025(){

#define M 45
#define N 54
#define L 23

int m1[2]={15,30};
int m2[2]={13,10};
int m3[4]={7,9,11,18};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2]=p[1:2]
#pragma xmp nodes px[2][4]
#pragma xmp template ta[1:100][L][M]
#pragma xmp template tb[L]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[block][cyclic][gblock(m1)] onto pa
#pragma xmp distribute tb[gblock(m2)] onto pb
#pragma xmp distribute tx[block][gblock(m3)] onto px

   int i, j, k, error;
   int a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[*][j][i]
#pragma xmp align b[i][*] with tb[i]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[*][j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_026(){

#define M 45
#define N 54
#define L 23

int m1[2]={15,30};
int m2[2]={3,20};
int m3[4]={7,9,11,18};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][4]
#pragma xmp nodes px[2][4]
#pragma xmp template ta[1:100][L][M]
#pragma xmp template tb[L][N]
#pragma xmp template tx[N][M]
#pragma xmp distribute ta[block][cyclic][gblock(m1)] onto pa
#pragma xmp distribute tb[gblock(m2)][cyclic(7)] onto pb
#pragma xmp distribute tx[block][gblock(m3)] onto px

   int i, j, k, error;
   long long a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[*][j][i]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[j][i]

#pragma xmp loop on ta[*][j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[j][i] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_027(){

#define M 45
#define N 54
#define L 23

int m1[2]={25,20};
int m2[2]={20,3};
int m3[2]={24,30};
int m4[2]={20,25};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[2][2][2]
#pragma xmp nodes pb[2][4]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[1:100][L][M]
#pragma xmp template tb[L][N]
#pragma xmp template tx[M][N][1:100]
#pragma xmp distribute ta[block][cyclic][gblock(m1)] onto pa
#pragma xmp distribute tb[gblock(m2)][cyclic(7)] onto pb
#pragma xmp distribute tx[gblock(m4)][gblock(m3)][block] onto px

   int i, j, k, error;
   float a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[i][j] with ta[*][j][i]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[i][j][*]

#pragma xmp loop on ta[*][j][i]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j][*] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_028(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={25,20};
int m2[2]={20,3};
int m3[2]={24,30};
int m4[2]={20,25};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][4]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][1:100]
#pragma xmp template tb[L][N]
#pragma xmp template tx[M][N][1:100]
#pragma xmp distribute ta[block][cyclic] onto pa
#pragma xmp distribute tb[gblock(m2)][cyclic(7)] onto pb
#pragma xmp distribute tx[gblock(m4)][gblock(m3)][block] onto px

   int i, j, k, error;
   double a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[j][*]
#pragma xmp align b[i][j] with tb[i][j]
#pragma xmp align x[i][j] with tx[i][j][*]

#pragma xmp loop on ta[j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1);
      }
   }

#pragma xmp loop on tb[i][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j][*] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_029(){

#define M 45
#define N 54
#define L 23

  //int m1[2]={25,20};
  //int m2[2]={20,3};
int m3[2]={24,30};
int m4[2]={20,25};
#pragma xmp nodes p[8]
#pragma xmp nodes pa[4][2]
#pragma xmp nodes pb[2][2][2]
#pragma xmp nodes px[2][2][2]
#pragma xmp template ta[L][1:100]
#pragma xmp template tb[1:100][1:100][N]
#pragma xmp template tx[M][N][1:100]
#pragma xmp distribute ta[block][cyclic] onto pa
#pragma xmp distribute tb[cyclic][cyclic][cyclic] onto pb
#pragma xmp distribute tx[gblock(m4)][gblock(m3)][block] onto px

   int i, j, k, error;
   double _Complex a[M][L], b[L][N], x[M][N], aa[M][L], bb[L][N], xx[M][N];
#pragma xmp align a[*][j] with ta[j][*]
#pragma xmp align b[*][j] with tb[*][*][j]
#pragma xmp align x[i][j] with tx[i][j][*]

#pragma xmp loop on ta[j][*]
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         a[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }
   for(i=0; i<M; i++){
      for(j=0; j<L; j++){
         aa[i][j] = ((i*L+j)%11+1)+I*((i*L+j)%7+1);
      }
   }

#pragma xmp loop on tb[*][*][j]
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         b[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }
   for(i=0; i<L; i++){
      for(j=0; j<N; j++){
         bb[i][j] = ((i*N+j)%13+1)+I*((i*N+j)%5+1);
      }
   }

   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         xx[i][j] = 0;
         for(k=0; k<L; k++){
            xx[i][j] += aa[i][k]*bb[k][j];
         }
      }
   }
   xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b));

   error = 0;
#pragma xmp loop on tx[i][j][*] reduction(+: error)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         if(x[i][j] != xx[i][j]){
            error++;
         }
      }
   }

   chk_int(error);
   return 0;
}

#include "mpi.h"

int main(int argc, char **argv){

  test_matmul_001();
  test_matmul_002();
  test_matmul_003();
  test_matmul_004();
  test_matmul_005();
  test_matmul_006();
  test_matmul_007();
  test_matmul_008();
  test_matmul_009();
  test_matmul_010();

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_matmul_011();
#endif

  test_matmul_012();
  test_matmul_013();
  test_matmul_014();
  test_matmul_015();
  test_matmul_016();

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_matmul_017();
#endif

  test_matmul_018();
  test_matmul_019();
  test_matmul_020();
  test_matmul_021();
  test_matmul_022();

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_matmul_023();
#endif

  test_matmul_024();
  test_matmul_025();
  test_matmul_026();
  test_matmul_027();
  test_matmul_028();

#if ((MPI_VERSION >= 3) || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2))
  test_matmul_029();
#endif

  return 0;

}
