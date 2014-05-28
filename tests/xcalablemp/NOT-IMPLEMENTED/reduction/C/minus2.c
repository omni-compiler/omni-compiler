/*testp052.c*/
/*reduction指示文(-)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(4,*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int a[N][N],sa;
double b[N][N],sb;
float c[N][N],sc;
int i,j;
char *result;
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
int main(void){

   if(xmp_num_nodes() < 16){
      printf("%s\n","You have to run this program by more than 16 nodes.");
   }
  
   sa=0;
   sb=0.0;
   sc=0.0;

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = 1;
         b[i][j] = 0.5;
         c[i][j] = 0.25;
      }
   }
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sa = sa-a[i][j];
      }
   }
#pragma xmp reduction (-:sa) 

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sb = sb-b[i][j];
      }
   }
#pragma xmp reduction (-:sb) 

#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sc = sc-c[i][j];
      }
   }
#pragma xmp reduction (-:sc) 
  
   result ="OK";
   if(sa != -(N*N)||abs(sb+((double)N*N*0.5))||abs(sc+((float)N*N*0.25))){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp052.c",result); 
   return 0;
}
      
         
      
   

   
