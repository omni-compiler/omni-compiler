/* barrier Construction allnode  rank 1 */
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#include<unistd.h>
#pragma xmp nodes p(4,*)
static const int N=1000; 
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
#pragma xmp align a[i][j] with t(j,i)
int a[N][N],procs,w1,w2,p1,p2,pi,pj;
int i,j,k,w;
char *result;
int main(void){

   procs = xmp_num_nodes();
   p1 = 4;
   p2 = procs/4;
   w1 = 250;
   if((N%p2)==0){
      w2 = N/p2;
   }else{
      w2 = N/p2 + 1;
   }
    
   for(k=0;k<procs-1;k++){
      pi = k%4;
      pj = k/4;
      if(xmp_node_num() == 1){
         for(i=0;i<w2;i++){
            for(j=0;j<w1;j++){
               a[i][j] = (j+pj*w2)+pi*w1+i;
            }
         }
      }

#pragma xmp barrier on t(:,:)
#pragma xmp task on p(pi+1,pj+1)
      {
#pragma xmp gmove in 
         a[pi*w1:w1][pj*w2:w2] = a[0:w1][0:w2];
      }
   }
   
   if(xmp_node_num() == 1){
      for(i=0;i<w2;i++){
         for(i=0;i<w1;i++){
            a[i][j] = j*N+i;
         }
      }
   }

   result = "OK";
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if(a[i][j] != j*N+i) result = "NG";
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp121.c",result);
   return 0;
}
      
         
     
   

