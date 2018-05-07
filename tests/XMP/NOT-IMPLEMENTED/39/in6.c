/*testp088.c*/
/*task指示文,post指示文,wait指示文*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<unistd.h>      
static const int N=1000;
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(cyclic,block) onto p
int a[N][N],sa;
int i,j;
#pragma xmp align a[i][j] with t(j,i)
char *result;
int main(void){

   if(xmp_num_nodes()!=16){
      printf("%s\n","You have to run this program by 16 nodes.");
   }

#pragma xmp tasks
   {
#pragma xmp task on p(:,1:2)
      {
#pragma xmp loop (j,i) on t(j,i)
         for(i=0;i<N/2;i++){
            for(j=0;j<N;j++){
               a[i][j] = 2;
            }
         }
#pragma xmp post (p(1,3),1)
#pragma xmp wait (p(1,3),2)
#pragma xmp loop (j,i) on t(j,i)
         for(i=0;i<N/2;i++){
            for(i=0;i<N;i++){
               a[i][j] = 1;
            }
         }
      }
#pragma xmp task on p(:,3:4)
      {
#pragma xmp wait (p(1,1),1)
#pragma xmp gmove in
         a[:][N/2:N/2] = a[:][0:N/2];
#pragma xmp post (p(1),2)
      }
   }
   sa=0;
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sa = sa + a[i][j];
      }
   }
#pragma xmp reduction(+:sa)
   if(sa == 1500000){
      result = "OK";
   }else{
      result = "NG";
   }
   
   printf("%d %s %s\n",xmp_node_num(),"testp088.c",result);
   return 0;
}    
         
      
   
   
