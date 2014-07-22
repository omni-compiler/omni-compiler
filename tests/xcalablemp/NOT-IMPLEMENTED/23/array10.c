/*testp100.c*/
/*loop指示文とarray指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h> 
#include<stdlib.h>     
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t1(cyclic(3),cyclic(7)) onto p
#pragma xmp distribute t2(block,cyclic) onto p
#pragma xmp distribute t3(cyclic,block) onto p
int a1[N][N],a2[N][N];
double b1[N][N],b2[N][N];
float c1[N][N],c2[N][N];
#pragma xmp align a1[i][j] with t1(j,i)
#pragma xmp align a2[i][j] with t1(j,i)
#pragma xmp align b1[i][j] with t2(j,i)
#pragma xmp align b2[i][j] with t2(j,i)
#pragma xmp align c1[i][j] with t3(j,i)
#pragma xmp align c2[i][j] with t3(j,i)
int i,j;
char *result;
int main(void){
   result = "OK";
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a2[i][j] = j*N+i;
      }
   }
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b2[i][j] = (double)(j*N+i);
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c2[i][j] = (float)(j*N+i);
      }
   }
#pragma xmp loop (i) on t1(:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t1(i,j)
      a1[:i] = a2[:j]+1;
   }

#pragma xmp loop (i) on t2(:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t2(i,j)
      b1[:i] = b2[:j]+2;
   }
#pragma xmp loop (i) on t3(:,i)
   for(i=0;i<N;i++){
#pragma xmp array on t3(i,j)
      c1[:i] = c2[:j]+3;
   }

   result = "OK";
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a1[i][j] != j*N+i+1) result = "NG";
      }
   }
   
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(b1[i][j]-(double)(j*N+i+2))>0.00000001) result = "NG";
      }
   }

#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(c1[i][j]-(double)(j*N+i+3))>0.001) result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp100.c",result);
   return 0;
}    
         
      
   
