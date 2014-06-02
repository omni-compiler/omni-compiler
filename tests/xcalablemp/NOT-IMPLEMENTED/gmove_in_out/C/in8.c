/*testp099.c*/
/*loop指示文とpost/wait指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>      
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
int a[N],aa;
int procs;
#pragma xmp align a[i] with t(i)
int i;
char *result; 
int main(void){
   
   procs = xmp_num_nodes();
   if(xmp_node_num() == 1){
      a[1] = 1;
   }
#pragma xmp barrier
#pragma xmp loop on t1(i)
   for(i=2;i<N;i++){
      if(i!=2){
#pragma xmp wait(p((i-1)%procs),1)
      }
#pragma xmp gmove in
      aa = a[i-1];
      a[i] = aa+1;
      if(i!=N){
#pragma xmp post(p((i+1)%procs),1)
      }
   }
   result = "OK";
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      if(a[i] != i) result = "NG";
   }
   printf("%d %s %s\n",xmp_node_num(),"testp099.c",result);
   return 0;
}    
         
      
   
