/*testp110.c*/
/*loop指示文とpost/wait指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>      
#pragma xmp nodes p(4,4)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(cyclic,cyclic) onto p
int a[N][N];
int ii,jj;
#pragma xmp align a[i][j] with t(i)
int i,j;
char *result; 
int main(void){
   if(xmp_num_nodes() != 16){
      printf("%d\n","You have to run this progmam by 16 nodes.");
   }   
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = i;
      }
   }
   
   ii = (xmp_node_num()%4)+1;
   jj = xmp_node_num()/4+1;
#pragma xmp barrier
#pragma xmp loop (i) on t(:,i)
   for(i=2;i<N+1;i++){
      if(i != 2){
#pragma xmp wait(p(jj,(i-1)%4))
      }
#pragma xmp gmove in
      a[:][i] = a[:][i-1];
#pragma xmp loop (i) on t(j,i)
      for(i=0;i<N;i++){
         a[i][j] = a[i][j] + i*N;
      }
      if(i!=N){
#pragma xmp post(p(jj,(i+1)%4)),1
      }
   }

   result = "OK";
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a[i][j] != j*N+i) result = "NG";
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp110.c",result);
   return 0;
}    
         
      
   
