/*task指示文,post指示文,wait指示文の組み合わせ*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<unistd.h>      
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int procs ,half_procs,w;
int a[N],sa;
int i;
#pragma xmp align a[i] with t(i)
char *result;
int main(void){

   if(xmp_num_nodes()!=4){
      printf("%s\n","You have to run this program by 4 nodes.");
   }

#pragma xmp tasks
   {
#pragma xmp task on p(1:2)
      {
#pragma xmp task on p(1:2)
#pragma xmp loop on t(i)
         for(i=0;i<N/2;i++){
            a[i] = i+N/2;
         }
#pragma xmp post (p(3),1)
#pragma xmp wait (p(3),2)
#pragma xmp loop on t(i)
         for(i=0;i<N/2;i++){
            a[i] = i;
         }
      }
#pragma xmp task on p(3:4)
      {
#pragma xmp wait (p(1),1)
#pragma xmp gmove in
         a[N/2:N/2] = a[0:N/2];
#pragma xmp post (p(1),2)
      }
   }
   sa=0;
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      sa+=a[i];
   }
#pragma xmp reduction(+:sa)
   if(sa == N*(N+1)/2){
      result = "OK";
   }else{
      result = "NG";
   }
   
   printf("%d %s %s\n",xmp_node_num(),"testp076.c",result);
   return 0;
}    
         
      
   
   
