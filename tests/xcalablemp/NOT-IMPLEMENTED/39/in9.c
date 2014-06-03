/*testp104.c*/
/*loop指示文とgmove指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>      
#pragma xmp nodes p(4,*)
static const int N=1000;
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp template t4(0:N-1,0:N-1)
#pragma xmp template t5(0:N-1,0:N-1)
#pragma xmp template t6(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(cyclic,block) onto p
#pragma xmp distribute t3(block,cyclic) onto p
#pragma xmp distribute t4(cyclic,cyclic) onto p
#pragma xmp distribute t5(cyclic(5),block) onto p
#pragma xmp distribute t6(block,cyclic(7)) onto p
int a1[N][N],a2[N][N];
double b1[N][N],b2[N][N];
float c1[N][N],c2[N][N];
#pragma xmp align a1[i][j] with t1(j,i)
#pragma xmp align a2[i][j] with t2(j,i)
#pragma xmp align b1[i][j] with t3(j,i)
#pragma xmp align b2[i][j] with t4(j,i)
#pragma xmp align c1[i][j] with t5(j,i)
#pragma xmp align c2[i][j] with t6(j,i)
int i,j;
char *result;
int main(void){
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a1[i][j] = 0;
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b1[i][j] = 0.0;
      }
   }
#pragma xmp loop (j,i) on t5(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c1[i][j] = 0.0;
      }
   }
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a2[i][j] = j*N+i;
      }
   }
#pragma xmp loop (j,i) on t4(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b2[i][j] = (double)(j*N+i);
      }
   }
#pragma xmp loop (j,i) on t6(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c2[i][j] = (float)(j*N+i);
      }
   }

#pragma xmp loop (j,i) on t2(j,:) 
   for(j=0;j<N;j++){
#pragma xmp gmove out
      a1[0:N/2:2][j] = a2[0:N/2:2][j];
#pragma xmp gmove out
      a1[1:N/2:2][j] = a2[1:N/2:2][j];
   }

#pragma xmp loop (j,i) on t4(j,:) 
   for(j=0;j<N;j++){
#pragma xmp gmove out
      b1[0:N/2:2][j] = b2[0:N/2:2][j];
#pragma xmp gmove out
      b1[1:N/2:2][j] = b2[1:N/2:2][j];
   }

#pragma xmp loop (j,i) on t5(j,:) 
   for(j=0;j<N;j++){
#pragma xmp gmove in
      c1[0:N/2:2][j] = c2[0:N/2:2][j];
#pragma xmp gmove in 
      c1[1:N/2:2][j] = c2[1:N/2:2][j];
   }

   result = "OK";
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a1[i][j] != j*N+i) result = "NG1";
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(b1[i][j]-(double)(j*N+i)) > 0.00000001) result = "NG2";
      }
   }
#pragma xmp loop (j,i) on t5(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(c1[i][j]-(float)(j*N+i)) > 0.001) result = "NG3";
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp104.c",result);
   return 0;
}    
         
      
   
   
