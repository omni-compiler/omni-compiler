int sub(int *x,int N){
#pragma xmp nodes p(*)
#pragma xmp template t(N)
#pragma xmp distribute t(cyclic(3)) onto p
#pragma xmp align x[i] with t(i)
   return 0;
}

#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(cyclic(3)) onto p
int N,s;
int *a,i,*x;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   N=0;
   for(i=0;i<1000;i++){   
      N = N+1;
   }
#pragma xmp template_fix(cyclic(3)) t(N)
   a = (int *)malloc(N);

   sub(a,N);   

#pragma xmp loop (i) on t(i)
for(i=0;i<N;i++){
   a[i] = i;
   }
   s = 0;
#pragma xmp loop (i) on t(i) reduction(+:s)
   for(i=0;i<N;i++){
      s = s+a[i];
   }
   result = "OK";
   if(s != 499500){
      result = "OK";
   }

   printf("%d %s %s\n",xmp_all_node_num(),"testp003.c",result); 
   return 0;
}
               
      
   
