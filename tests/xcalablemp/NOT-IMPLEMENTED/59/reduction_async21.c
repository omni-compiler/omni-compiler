/*testp101.c*/
/*loop指示文とreflect指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#pragma xmp nodes p(4,*)
static const int N=1000;
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(block,block) onto p
#pragma xmp distribute t3(block,block) onto p
int a1[N][N],a2[N][N];
double b1[N][N],b2[N][N];
float c1[N][N],c2[N][N];
#pragma xmp align a1[i][j] with t1(j,i)
#pragma xmp align a2[i][j] with t1(j,i)
#pragma xmp align b1[i][j] with t2(j,i)
#pragma xmp align b2[i][j] with t2(j,i)
#pragma xmp align c1[i][j] with t3(j,i)
#pragma xmp align c2[i][j] with t3(j,i)
#pragma xmp shadow a2[1][1]
#pragma xmp shadow b2[2][2]
#pragma xmp shadow c2[3][3]
int i,j,ii,jj;
char *result;
int main(void){
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a1[i][j] = 0;
         a2[i][j] = j*N+i;
      }
   }
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b1[i][j] = 0.0;
         b2[i][j] = (double)(j*N+i);
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c1[i][j] = 0.0;
         c2[i][j] = (float)(j*N+i);
      }   
   }
 
   result = "OK";
#pragma xmp loop (j,i) on t1(j,i)
   for(i=1;i<N-1;i++){
      for(j=1;j<N-1;j++){
#pragma xmp reflect (a2) async(1)
         if(a1[i][j] != 0) result = "NG1";
#pragma xmp wait_async(1)
         for(ii=i-1;ii<i+2;ii++){
            for(jj=j-1;jj<j+2;jj++){
               a1[i][j] = a1[i][j]+a2[ii][jj];
            }
         }
         a1[i][j] = a1[i][j]/9;
      }
   }
#pragma xmp loop (j,i) on t2(j,i)
   for(i=2;i<N-2;i++){
      for(j=2;j<N-2;j++){
#pragma xmp reflect (b2) async(1)
         if(b1[i][j] != 0) result = "NG2";
#pragma xmp wait_async(1)
         for(ii=i-2;ii<i+3;ii++){
            for(jj=j-2;jj<j+3;jj++){
               b1[i][j] = b1[i][j]+b2[ii][jj];
            }
         }
         b1[i][j] = b1[i][j]/25.0;
      }
   }

#pragma xmp loop (j,i) on t3(j,i)
   for(i=3;i<N-3;i++){
      for(j=3;j<N-3;j++){
#pragma xmp reflect (c2) async(1)
         if(c1[i][j] != 0) result = "NG3";
#pragma xmp wait_async(1)
         for(ii=i-3;ii<i+4;ii++){
            for(jj=j-3;jj<j+4;jj++){
               c1[i][j] = c1[i][j]+c2[ii][jj];
            }
         }
         c1[i][j] = c1[i][j]/49.0;
      }
   }

#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(i==0||i==N-1||j==0||j==N-1){
            if(a1[i][j] != 0) result = "NG4";
         }else{
            if(a1[i][j] != j*N+i) result = "NG5";
         }
      }
   }

#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(i<=1||i>=N-2||j<=1||j>=N-2){
            if(b1[i][j] != 0.0) result = "NG6";
         }else{
            if(abs(b1[i][j]-(double)(j*N+i))) result = "NG7";
         }
      }
   }

#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(i<=2||i>=N-3||j<=2||j>=N-3){
            if(c1[i][j] != 0.0) result = "NG8";
         }else{
            if(abs(c1[i][j]-(float)(j*N+i))) result = "NG9";
         }
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp101.c",result); 

   return 0;
}
      
         
      
   
