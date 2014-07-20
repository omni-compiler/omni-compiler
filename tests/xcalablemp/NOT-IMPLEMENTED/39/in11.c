/*testp115.c*/
/*barrier指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#include<unistd.h>
static const int N=1000;
#pragma xmp nodes p(*) 
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int a[N],procs,w;
int i,j;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   procs = xmp_num_nodes();
    
   for(j=1;j<procs;j++){
      if((N%procs)==0){
         w = N/procs;
      }else{
         w = N/procs+1;
         w = (w < (N-w*(j-1)))? w: N-w*(j-1);
      }
      if(xmp_node_num() == 1){
         for(i=0;i<w;i++){
            a[i] = j*w+i;
         }
      }
   
#pragma xmp barrier on p(:)
#pragma xmp task on p(j+1)
      {
#pragma xmp gmove in 
         a[j*w:w] = a[0:w];
      }
   }
   if(xmp_node_num() == 1){
      for(i=0;i<w;i++){
         a[i] = i+1;
      }
   }
   result = "OK";
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      if(a[i] != i) result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp115.c",result);
   return 0;
}
      
         
     
   

