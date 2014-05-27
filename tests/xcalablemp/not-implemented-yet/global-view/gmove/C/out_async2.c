/*testp023.c*/
/*gmove指示文とasync節のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(4,*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp template t4(0:N-1,0:N-1)
#pragma xmp template t5(0:N-1,0:N-1)
#pragma xmp template t6(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(block,cyclic) onto p
#pragma xmp distribute t3(cyclic,block) onto p
#pragma xmp distribute t4(cyclic,cyclic) onto p
#pragma xmp distribute t5(cyclic(2),cyclic(3)) onto p
#pragma xmp distribute t6(cyclic(4),cyclic(5)) onto p
int a1[N][N],a2[N][N],sa;
double b1[N][N],b2[N][N],sb;
float c1[N][N],c2[N][N],sc;
#pragma xmp align a1[i][j] with t1(j,i)
#pragma xmp align b1[i][j] with t2(j,i)
#pragma xmp align c1[i][j] with t3(j,i)
#pragma xmp align a2[i][j] with t4(j,i)
#pragma xmp align b2[i][j] with t5(j,i)
#pragma xmp align c2[i][j] with t6(j,i)
int procs;
int i,j;
char *result;
int main(void){

#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a1[i][j] = j*N+i;
      }
   }

#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b1[i][j] = (double)j*N+i;
      }
   }

#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c1[i][j] = (float)j*N+i;
      }
   }

#pragma xmp loop (j,i) on t4(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a2[i][j] = 1;
      }
   }

#pragma xmp loop on t5(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b2[i][j] = 1.0;
      }
   }

#pragma xmp loop on t6(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         c2[i][j] = 1.0;
      }
   }

   procs = xmp_num_nodes();
   for(j=1;j<procs+1;j++){
      if(j == xmp_node_num()){
#pragma xmp gmove out async(1)
         a1[:][:] = a2[:][:];
#pragma xmp gmove out async(2)
         b1[:][:] = b2[:][:];
#pragma xmp gmove out async(3)
         c1[:][:] = c2[:][:];
      }
   }

#pragma xmp wait_async(1)
   if(xmp_node_num() == 1){
      a1[0][0] = a1[0][0] + 10;
   }
#pragma xmp wait_async(2)
   if(xmp_node_num() == 1){
      b1[0][0] = b1[0][0] + 30.0;
   }
#pragma xmp wait_async(3)
   if(xmp_node_num() == 1){
      c1[0][0] = c1[0][0] + 20.0;
   }

   sa = 0;
   sb = 0.0;
   sc = 0.0;
#pragma xmp loop (j,i) on t1(j,i) reduction(+:sa)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sa = sa+a1[i][j];
      }
   }
#pragma xmp loop (j,i) on t2(j,i) reduction(+:sb)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sb = sb+b1[i][j];
      }
   }
#pragma xmp loop (j,i) on t3(j,i) reduction(+:sc)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sc = sc+c1[i][j];
      }
   }

   result = "OK";
   if((sa != 1000010)||abs(sb-1000030.0) > 0.000000001||abs(sc-1000020.0) > 0.0001){
      result="NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp023.c",result); 
   return 0;
}
      
         
      
   
















