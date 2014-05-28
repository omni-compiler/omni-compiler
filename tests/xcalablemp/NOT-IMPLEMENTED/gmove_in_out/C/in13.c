/* barrier Construction allnode  rank 1 */
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#include<unistd.h>
#pragma xmp nodes p(*)
static const int N=1000; 
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int a[N],procs,w;
int i,j;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   procs = xmp_num_nodes();
   if((N%procs)==0){
      w = N/procs;
   }else{
      w = N/procs+1;
   }
    
   for(j=1;j<procs;j++){
      if(xmp_node_num() == 1){
         for(i=0;i<w;i++){
            a[i] = j*w+i;
         }
      }
   
#pragma xmp barrier on t(:)
#pragma xmp task on t(j+1)
      {
#pragma xmp gmove in 
         a[j*w:w] = a[0:w];
      }
   }

   if(xmp_node_num() == 1){
      for(i=0;i<w;i++){
         a[i] = i;
      }
   }

   result = "OK";
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      if(a[i] != i) result = "NG";
   }
   printf("%d %s %s\n",xmp_node_num(),"testp117.c",result);
   return 0;
}
      
         
     
   

