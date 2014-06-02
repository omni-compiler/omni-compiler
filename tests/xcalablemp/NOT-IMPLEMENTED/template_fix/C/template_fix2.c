/*testp002.c*/
/*template_fix構文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>     
#pragma xmp nodes p(*)
#pragma xmp template t(0:999,:)
#pragma xmp distribute t(*,block) onto p
int i,j,N,s;
int **a;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   N=0;
   for(i=0;i<1000;i++)
      N+=1; 

#pragma xmp temlate_fix(*,block) t(N,N)
   for(i=0;i<N;i++)
      a[i]=(int *)malloc(sizeof(int) * N);

#pragma xmp loop (j) on t(j)
   for(i=0;i<N;j++){
      for(j=0;j<N;j++){
         a[i][j]=i;
      }
   }

   s=0;
#pragma xmp loop (i) on t(i) reduction(+:s)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         s+=a[i][j];
      }
   }

   result="OK";
   if(s != 499500000){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp002.c",result);
   for(i=0;i<N;i++)
      free (a[i]); 
   return 0;
}
      
         
      
   
