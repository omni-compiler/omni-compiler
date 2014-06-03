/*testp172.c */
/*loop指示文とreduction節(^)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
int random_array[1000000],ans_val,val;
int a[N][N],sa;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(cyclic,cyclic) onto p
int i,j,k,l;
#pragma xmp align a[i][j] with t(j,i)
char *result;
int main(void){
   result="OK";
   for(k=114;k<10001;k=k+113){
      random_array[0] = k;
      for(i=1;i<N*N;i++){
         random_array[i]=(13*random_array[i-1]+17)%9973; 
      }
#pragma xmp loop (j,i) on t(j,i)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            l = j*N+i;
            a[i][j] = random_array[l];
         }
      }
      ans_val = -1;

      for(i=0;i<N*N;i++){
         ans_val = ans_val^random_array[i];
      }

      sa = -1;
#pragma xmp loop (j,i) on t(j,i) reduction(^:sa)
      for(i=0;i<N;i++){
         for(j=0;j<N;j++){
            sa = sa^a[i][j];    
         }
      }   
      if(sa != ans_val){
         result = "NG";
      }
   }
   printf("%d %s %s\n",xmp_node_num(),"testp172.c",result); 
   return 0;
}
      
         
      
   












