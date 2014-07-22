/*testp105.c*/
/*loop指示文とbarrier指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
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
#pragma xmp distribute t3(cyclic,cyclic) onto p
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
#pragma xmp barrier
#pragma xmp gmove out
      a1[i][:] = a2[i][:];
   }

#pragma xmp loop (j,i) on t4(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b2[i][j] = (double)(j*N+i);
      }
#pragma xmp barrier
#pragma xmp gmove out
      b1[i][:] = b2[i][:];
   }

#pragma xmp loop (j,i) on t6(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c2[i][j] = (float)j*N+i;
      }
#pragma xmp barrier
#pragma xmp gmove out
      c1[i][:] = c2[i][:];
   }

   result = "OK";
#pragma xmp loop on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a1[i][j] != j*N+i) result = "NG1";
      }
   }

#pragma xmp loop on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(b1[i][j]-(double)(j*N+i))>0.00000001) result = "NG2";
      }
   }

#pragma xmp loop on t5(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(abs(c1[i][j]-(float)(j*N+i))>0.001) result = "NG3";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp105.c",result);
   return 0;
}
      
         
     
   

