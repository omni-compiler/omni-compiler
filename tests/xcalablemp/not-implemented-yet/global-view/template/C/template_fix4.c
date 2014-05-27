#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(gblock(*)) onto p
int i,j,s,procs,remain;
int *a,*m;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){
   procs=xmp_num_nodes();
   m=(int *)malloc(sizeof(int) * procs);
   remain=N;
   for(i=1;i<procs;i++){
      m[i]=remain/2;
      remain=remain-m[i];
   }
   m[procs]=remain;
#pragma xmp template_fix(block(m)) t(N)
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
   printf("%d %s %s\n",xmp_node_num(),"testp004.c",result);
   free (a);
   free (m); 
}
      
         
      
   
