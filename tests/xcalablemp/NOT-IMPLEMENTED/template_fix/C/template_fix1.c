#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>     
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(block) onto p
int i,N,s;
int *a;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   N=0;
   for(i=1;N<1001;i++)
      N+=1;
 
#pragma xmp temlate_fix(block) t(N)
   a=(int *)malloc(sizeof(int) * N);

#pragma xmp loop (i) on t(i)
   for(i=0;i<N;i++){
      a[i]=i;
   }

   s=0;
#pragma xmp loop (i) on t(i) reduction(+:s)
   for(i=0;i<N;i++){
      s+=a[i];
   }

   result="OK";
   if(s != 499500){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_all_node_num(),"testp001.c",result);
   free(a); 
   return 0;
}
      
         
      
   
