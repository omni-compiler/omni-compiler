#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>     
#pragma xmp nodes p(4,4)
#pragma xmp template t(4,:)
#pragma xmp distribute t(cyclic,block) onto p
int i,j,N,s;
int **a;
#pragma xmp align a[i][j] with t(j,i)
char *result;
int main(void){
   if(xmp_num_nodes()!=16){
      printf("%s\n","You have to run this program by 16 nodes.");
   }
   N=0;
   for(i=1;N<1001;i++)
      N+=1; 
#pragma xmp template_fix(block) t(N,N)
   for(i=0;i<N;i++)
      a[i]=(int *)malloc(sizeof(int) * N);

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j]=xmp_node_num();
      }
   }
   s=0;
#pragma xmp loop (j,i) on t(j,i) reduction(+:s)
   for(i=1;i<N+1;i++){
      for(j=1;j<N+1;j++){
         s = s+a[i][j];
      }
   }
   result="OK";
   if(s != 45000000){
      result = "NG";
   }
   printf("%d %s %s\n",xmp_all_node_num(),"testp005.c",result);
   for(i=0;i<10000;i++)
      free(a[i]); 
   return 0;
}
      
         
      
   
   
