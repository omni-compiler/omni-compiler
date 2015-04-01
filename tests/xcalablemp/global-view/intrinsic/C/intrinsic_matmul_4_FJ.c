#include <stdio.h>
#include <xmp.h>
#include <math.h>
#include <complex.h>

extern int chk_int(int ierr);
extern void xmp_matmul(void *x_p, void *a_p, void *b_p);
int test_matmul_001(){

#pragma xmp nodes p(2,2)
#pragma xmp nodes pp(4)
#pragma xmp template ta(0:15,0:7)
#pragma xmp template tb(0:7,0:15)
#pragma xmp template tx(0:15,0:15)
#pragma xmp distribute ta(block,block) onto p
#pragma xmp distribute tb(cyclic,cyclic) onto p
#pragma xmp distribute tx(cyclic(2),cyclic(3)) onto p

   int i, j, k, error;
   int a[16][8], b[8][16], x[16][16];
   int c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j)
#pragma xmp align x[i][j] with tx(i,j)

#pragma xmp loop (j,i) on ta(j,i)
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

#pragma xmp loop (i,j) on tb(i,j)
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
#pragma xmp loop (j,i) on tx(j,i) reduction(+: error)
   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]) error++;
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_002(){

#pragma xmp nodes p(2,2)
#pragma xmp nodes pp(4)
#pragma xmp template ta(0:15,0:7)
#pragma xmp template tb(0:7,0:15)
#pragma xmp template tx(0:15,0:15)
#pragma xmp distribute ta(block,block) onto p
#pragma xmp distribute tb(cyclic,cyclic) onto p
#pragma xmp distribute tx(cyclic(2),cyclic(3)) onto p

   int i, j, k, error;
   double a[16][8], b[8][16], x[16][16];
   double c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j)
#pragma xmp align x[i][j] with tx(i,j)

#pragma xmp loop (j,i) on ta(j,i)
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

#pragma xmp loop (i,j) on tb(i,j)
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
#pragma xmp loop (j,i) on tx(j,i) reduction(+: error)
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

#pragma xmp nodes p(2,2)
#pragma xmp nodes pp(4)
#pragma xmp template ta(0:15,0:7)
#pragma xmp template tb(0:7,0:15)
#pragma xmp template tx(0:15,0:15)
#pragma xmp distribute ta(cyclic(2),cyclic(3)) onto p
#pragma xmp distribute tb(block,block) onto p
#pragma xmp distribute tx(cyclic,cyclic) onto p

   int i, j, k, error;
   int a[16][8], b[8][16], x[16][16];
   int c[16][8], d[8][16], y[16][16];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j)
#pragma xmp align x[i][j] with tx(i,j)

#pragma xmp loop (j,i) on ta(j,i)
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

#pragma xmp loop (i,j) on tb(i,j)
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
#pragma xmp loop (j,i) on tx(j,i) reduction(+: error)
   for(j=0; j<16; j++){
      for(i=0; i<16; i++){
         if(x[j][i] != y[j][i]) error++;
      }
   }

   chk_int(error);
   return 0;
}

int test_matmul_004(){

#pragma xmp nodes p(2,2)
#pragma xmp nodes pp(4)
#pragma xmp template ta(0:19,0:7)
#pragma xmp template tb(0:7,0:15)
#pragma xmp template tx(0:19,0:15)
#pragma xmp distribute ta(block,block) onto p
#pragma xmp distribute tb(block,block) onto p
#pragma xmp distribute tx(block,block) onto p

   int i, j, k, error;
   int a[20][8], b[8][16], x[20][16];
   int c[20][8], d[8][16], y[20][16];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j)
#pragma xmp align x[i][j] with tx(i,j)

#pragma xmp loop (j,i) on ta(j,i)
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

#pragma xmp loop (i,j) on tb(i,j)
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
#pragma xmp loop (j,i) on tx(j,i) reduction(+: error)
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

#pragma xmp nodes p(2,2)
#pragma xmp nodes pp(4)
#pragma xmp template ta(0:19,0:7)
#pragma xmp template tb(0:7,0:15)
#pragma xmp template tx(0:19,0:15)
#pragma xmp distribute ta(block,block) onto p
#pragma xmp distribute tb(block,block) onto p
#pragma xmp distribute tx(block,block) onto p

   int i, j, k, error;
   int a[20][8], b[8][16], x[20][16];
   int c[20][8], d[8][16], y[20][16];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j)
#pragma xmp align x[i][j] with tx(i,j)
#pragma xmp shadow a[1][1]
#pragma xmp shadow b[1][1]
#pragma xmp shadow x[1][1]

#pragma xmp loop (j,i) on ta(j,i)
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

#pragma xmp loop (i,j) on tb(i,j)
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
#pragma xmp loop (j,i) on tx(j,i) reduction(+: error)
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

int main(int argc, char **argv){

  test_matmul_001();
  test_matmul_002();
  test_matmul_003();
  test_matmul_004();
  test_matmul_005();

  return 0;

}
