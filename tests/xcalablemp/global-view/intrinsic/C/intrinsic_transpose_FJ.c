#include <stdio.h>
#include <xmp.h>
#include <complex.h>

extern int chk_int(int ierr);
extern void xmp_transpose(void *dst_d, void *src_d, int opt);
#pragma xmp nodes p(8)

int test_transpose_001(){

#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(4,2)
#pragma xmp template ta(0:7,0:15)
#pragma xmp template tb(0:15,0:7)
#pragma xmp distribute ta(block,block) onto pa
#pragma xmp distribute tb(block,block) onto pb

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][i] with ta(i,j)
#pragma xmp align b[j][i] with tb(i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j,i) on tb(j,i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);
}

int test_transpose_002(){

#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(4,2)
#pragma xmp template ta(0:7,0:15)
#pragma xmp template tb(0:15,0:7)
#pragma xmp distribute ta(cyclic,cyclic) onto pa
#pragma xmp distribute tb(cyclic,cyclic) onto pb

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][i] with ta(i,j)
#pragma xmp align b[j][i] with tb(i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j,i) on tb(j,i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_003(){

#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(4,2)
#pragma xmp template ta(0:7,0:15)
#pragma xmp template tb(0:15,0:7)
#pragma xmp distribute ta(cyclic(2),cyclic(3)) onto pa
#pragma xmp distribute tb(cyclic(4),cyclic) onto pb

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][i] with ta(i,j)
#pragma xmp align b[j][i] with tb(i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j,i) on tb(j,i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_004(){

int b1[2]={2,6};
int b2[4]={2,2,4,8};
int b3[4]={8,4,2,2};
int b4[2]={5,3};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(4,2)
#pragma xmp template ta(0:7,0:15)
#pragma xmp template tb(0:15,0:7)
#pragma xmp distribute ta(gblock(b1),gblock(b2)) onto pa
#pragma xmp distribute tb(gblock(b3),gblock(b4)) onto pb

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][i] with ta(i,j)
#pragma xmp align b[j][i] with tb(i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j,i) on tb(j,i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_005(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:15)
#pragma xmp distribute t(block) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][*] with t(j)
#pragma xmp align b[*][i] with t(i)

#pragma xmp loop (j) on t(j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on t(j) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_006(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:7)
#pragma xmp distribute t(block) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[*][j] with t(j)
#pragma xmp align b[i][*] with t(i)

#pragma xmp loop (i) on t(i)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on t(i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_007(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:15)
#pragma xmp distribute t(cyclic) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][*] with t(j)
#pragma xmp align b[*][i] with t(i)

#pragma xmp loop (j) on t(j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on t(j) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_008(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:7)
#pragma xmp distribute t(cyclic) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[*][j] with t(j)
#pragma xmp align b[i][*] with t(i)

#pragma xmp loop (i) on t(i)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on t(i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_009(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:15)
#pragma xmp template tb(0:15)
#pragma xmp distribute t(cyclic(3)) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][*] with t(j)
#pragma xmp align b[*][j] with t(j)

#pragma xmp loop (j) on t(j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on t(j)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}


int test_transpose_010(){

#pragma xmp nodes p(8)
#pragma xmp template t(0:7)
#pragma xmp distribute t(cyclic(2)) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[*][j] with t(j)
#pragma xmp align b[i][*] with t(i)

#pragma xmp loop (i) on t(i)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on t(i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_011(){

static int m[8]={1,2,1,2,1,2,3,4};
#pragma xmp nodes p(8)
#pragma xmp template t(0:15)
#pragma xmp distribute t(gblock(m)) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[j][*] with t(j)
#pragma xmp align b[*][i] with t(i)

#pragma xmp loop (j) on t(j)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on t(j) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_012(){

static int m[8]={0,1,0,1,1,1,2,2};
#pragma xmp nodes p(8)
#pragma xmp template t(0:7)
#pragma xmp distribute t(gblock(m)) onto p

   int i, j, error;
   int a[16][8], b[8][16];
#pragma xmp align a[*][j] with t(j)
#pragma xmp align b[i][*] with t(i)

#pragma xmp loop (i) on t(i)
   for(j=0; j<16; j++){
      for(i=0; i<8; i++){
         a[j][i] = -1*(j*8+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on t(i) reduction(+: error)
   for(i=0; i<8; i++){
      for(j=0; j<16; j++){
         if(b[i][j] != -1*(j*8+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_013(){

#define M 23
#define N 31
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:M-1)
#pragma xmp distribute ta(block,block) onto pa
#pragma xmp distribute tb(cyclic,cyclic,cyclic) onto pb

   int i, j, error;
   short a[M][N], b[N][M];
#pragma xmp align a[*][i] with ta(*,i)
#pragma xmp align b[*][i] with tb(i,*,*)

#pragma xmp loop (i) on ta(*,i)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on tb(j,*,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_014(){

#define M 23
#define N 31
#pragma xmp nodes p(8)
#pragma xmp nodes pa(8)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:M-1)
#pragma xmp distribute ta(cyclic(3)) onto pa
#pragma xmp distribute tb(cyclic,cyclic,cyclic) onto pb

   int i, j, error;
   int a[M][N], b[N][M];
#pragma xmp align a[*][i] with ta(i)
#pragma xmp align b[*][i] with tb(i,*,*)

#pragma xmp loop (i) on ta(i)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on tb(j,*,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);
}

int test_transpose_015(){
#define M 23
#define N 31
int m1[2]={11,12};

#pragma xmp nodes p(8)
#pragma xmp nodes pa(8)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:M-1)
#pragma xmp distribute ta(cyclic(3)) onto pa
#pragma xmp distribute tb(cyclic,gblock(m1),cyclic) onto pb

   int i, j, error;
   long long a[M][N], b[N][M];
#pragma xmp align a[*][i] with ta(i)
#pragma xmp align b[*][i] with tb(*,i,*)

#pragma xmp loop (i) on ta(i)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on tb(*,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_016(){
#define M 23
#define N 31
int m1[2]={11,12};

#pragma xmp nodes p(8)
#pragma xmp nodes pa(4)=p(3:6)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:M-1)
#pragma xmp distribute ta(block) onto pa
#pragma xmp distribute tb(cyclic,gblock(m1),cyclic) onto pb

   int i, j, error;
   float a[M][N], b[N][M];
#pragma xmp align a[j][*] with ta(j)
#pragma xmp align b[*][i] with tb(*,i,*)

#pragma xmp loop (j) on ta(j)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (j) on tb(*,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_017(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4)=p(3:6)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(block) onto pa
#pragma xmp distribute tb(block,block,block) onto pb

   int i, j, error;
   double a[M][N], b[N][M];
#pragma xmp align a[j][*] with ta(j)
#pragma xmp align b[j][i] with tb(j,*,i)

#pragma xmp loop (j) on ta(j)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,*,j) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);
}

int test_transpose_018(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(block,cyclic) onto pa
#pragma xmp distribute tb(block,block,block) onto pb

   int i, j, error;
   float _Complex a[M][N], b[N][M];
#pragma xmp align a[j][i] with ta(j,i)
#pragma xmp align b[j][i] with tb(j,*,i)

#pragma xmp loop (j,i) on ta(j,i)
   for(j=0; j<M; j++){
      for(i=0; i<N; i++){
         a[j][i] = -1*(j*N+i)+I*(j*N+i);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,*,j) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)+I*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_019(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1,0:N-1,0:M-1)
#pragma xmp distribute ta(block,cyclic) onto pa
#pragma xmp distribute tb(block,block,cyclic(3)) onto pb

   int i, j, error;
   short a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(*,i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,i,j) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_020(){

#define M 23
#define N 31
int m1[2]={1,30};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1,0:N-1,0:M-1)
#pragma xmp distribute ta(block,gblock(m1)) onto pa
#pragma xmp distribute tb(block,block,cyclic(3)) onto pb

   int i, j, error;
   int a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(*,i,j)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,i,j) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_021(){

#define M 23
#define N 31
int m1[2]={1,30};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(8)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1)
#pragma xmp distribute ta(block,gblock(m1)) onto pa
#pragma xmp distribute tb(cyclic) onto pb

   int i, j, error;
   long long a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][*] with tb(i)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on tb(i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_022(){

#define M 23
#define N 31
  //int m1[2]={1,30};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(8)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:N-1)
#pragma xmp distribute ta(cyclic,block) onto pa
#pragma xmp distribute tb(cyclic) onto pb

   int i, j, error;
   float a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][*] with tb(i)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on tb(i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_023(){

#define M 23
#define N 31
  //int m1[2]={1,30};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:M-1,0:N-1,0:N-1)
#pragma xmp distribute ta(cyclic,block) onto pa
#pragma xmp distribute tb(cyclic,cyclic,cyclic) onto pb

   int i, j, error;
   double a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][j] with tb(j,i,*)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(j,i,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_024(){

#define M 23
#define N 31
  //int m1[2]={1,30};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:M-1,0:N-1,0:N-1)
#pragma xmp distribute ta(cyclic,cyclic(15)) onto pa
#pragma xmp distribute tb(cyclic,cyclic,cyclic) onto pb

   int i, j, error;
   double _Complex a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(j,i,*)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j)+I*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(j,i,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)+I*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_025(){

#define M 23
#define N 31
int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(cyclic,cyclic(7)) onto pa
#pragma xmp distribute tb(cyclic,cyclic,gblock(m1)) onto pb

   int i, j, error;
   short a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(*,j,i)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,j,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_026(){

#define M 23
#define N 31
int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4)=p(2:5)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(cyclic(2)) onto pa
#pragma xmp distribute tb(cyclic,cyclic,gblock(m1)) onto pb

   int i, j, error;
   int a[M][N], b[N][M];
#pragma xmp align a[i][*] with ta(i)
#pragma xmp align b[i][j] with tb(*,j,i)

#pragma xmp loop (i) on ta(i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,j,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_027(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4)=p(2:5)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(cyclic(2)) onto pa
#pragma xmp distribute tb(block,block,cyclic) onto pb

   int i, j, error;
   long long a[M][N], b[N][M];
#pragma xmp align a[i][*] with ta(i)
#pragma xmp align b[i][j] with tb(j,*,i)

#pragma xmp loop (i) on ta(i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(j,*,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_028(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(cyclic(4),cyclic) onto pa
#pragma xmp distribute tb(block,block,cyclic) onto pb

   int i, j, error;
   float a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(j,*,i)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(j,*,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_029(){

#define M 23
#define N 31
  //int m1[2]={11,20};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(cyclic(4),cyclic) onto pa
#pragma xmp distribute tb(cyclic(2),cyclic(3),cyclic(4)) onto pb

   int i, j, error;
   double a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j,*)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_030(){

#define M 23
#define N 31
int m1[2]={20,11};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(gblock(m1),cyclic(9)) onto pa
#pragma xmp distribute tb(cyclic(4),cyclic(2),cyclic(3)) onto pb

   int i, j, error;
   float _Complex a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][j] with tb(i,j,*)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j)+I*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)+I*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_031(){

#define M 23
#define N 31
int m1[2]={20,11};
int m2[8]={3,2,1,3,2,1,9,10};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(8)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:N-1)
#pragma xmp distribute ta(gblock(m1),cyclic(9)) onto pa
#pragma xmp distribute tb(gblock(m2)) onto pb

   int i, j, error;
   short a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][*] with tb(i)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on tb(i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_032(){

#define M 23
#define N 31
int m1[2]={12,11};
int m2[8]={10,3,2,1,3,2,1,9};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(8)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1)
#pragma xmp distribute ta(gblock(m1),block) onto pa
#pragma xmp distribute tb(gblock(m2)) onto pb

   int i, j, error;
   int a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][*] with tb(i)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i) on tb(i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_033(){

#define M 23
#define N 31
int m1[2]={12,11};
int m2[2]={10,21};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(2,4)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1,0:N-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(gblock(m1),block) onto pa
#pragma xmp distribute tb(gblock(m2),cyclic,block) onto pb

   int i, j, error;
   long long a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(i,j)
#pragma xmp align b[i][j] with tb(i,j,*)

#pragma xmp loop (i,j) on ta(i,j)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_034(){

#define M 23
#define N 31
int m1[2]={3,20};
int m2[2]={10,21};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:N-1,0:M-1,0:M-1)
#pragma xmp distribute ta(cyclic(5),gblock(m1)) onto pa
#pragma xmp distribute tb(gblock(m2),cyclic,block) onto pb

   int i, j, error;
   float a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][j] with tb(i,j,*)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(i,j,*) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_035(){

#define M 23
#define N 31
int m1[2]={3,20};
int m2[2]={10,21};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(4,2)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:N-1,0:M-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(cyclic(5),gblock(m1)) onto pa
#pragma xmp distribute tb(block,gblock(m1),gblock(m2)) onto pb

   int i, j, error;
   double a[M][N], b[N][M];
#pragma xmp align a[i][j] with ta(j,i)
#pragma xmp align b[i][j] with tb(*,j,i)

#pragma xmp loop (i,j) on ta(j,i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,j,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int test_transpose_036(){

#define M 23
#define N 31
int m1[2]={20,3};
int m2[2]={3,28};
#pragma xmp nodes p(8)
#pragma xmp nodes pa(8)
#pragma xmp nodes pb(2,2,2)
#pragma xmp template ta(0:M-1)
#pragma xmp template tb(0:M-1,0:M-1,0:N-1)
#pragma xmp distribute ta(block) onto pa
#pragma xmp distribute tb(block,gblock(m1),gblock(m2)) onto pb

   int i, j, error;
   double _Complex a[M][N], b[N][M];
#pragma xmp align a[i][*] with ta(i)
#pragma xmp align b[i][j] with tb(*,j,i)

#pragma xmp loop (i) on ta(i)
   for(i=0; i<M; i++){
      for(j=0; j<N; j++){
         a[i][j] = -1*(i*N+j)+I*(i*N+j);
      }
   }

   xmp_transpose(xmp_desc_of(b), xmp_desc_of(a), 0);

   error = 0;
#pragma xmp loop (i,j) on tb(*,j,i) reduction(+: error)
   for(i=0; i<N; i++){
      for(j=0; j<M; j++){
         if(b[i][j] != -1*(j*N+i)+I*(j*N+i)){
            error++;
         }
      }
   }

   return chk_int(error);

}

int main(){

  test_transpose_001();
  test_transpose_002();
  test_transpose_003();
  test_transpose_004();
  test_transpose_005();
  test_transpose_006();
  test_transpose_007();
  test_transpose_008();
  test_transpose_009();
  test_transpose_010();
  test_transpose_011();
  test_transpose_012();
  test_transpose_013();
  test_transpose_014();
  test_transpose_015();
  test_transpose_016();
  test_transpose_017();
  test_transpose_018();
  test_transpose_019();
  test_transpose_020();
  test_transpose_021();
  test_transpose_022();
  test_transpose_023();
  test_transpose_024();
  test_transpose_025();
  test_transpose_026();
  test_transpose_027();
  test_transpose_028();
  test_transpose_029();
  test_transpose_030();
  test_transpose_031();
  test_transpose_032();
  test_transpose_033();
  test_transpose_034();
  test_transpose_035();
  test_transpose_036();

  return 0;

}
