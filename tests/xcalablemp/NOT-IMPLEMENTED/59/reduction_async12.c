/*testp049.c*/
/*reduction指示文(*)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(4,*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t1(block,block) onto p
#pragma xmp distribute t2(block,cyclic) onto p
#pragma xmp distribute t3(cyclic,cyclic) onto p
int a[N][N],sa;
double b[N][N],sb;
float c[N][N],sc;
int i,j,m;
char *result;
#pragma xmp align a[i][j] with t1(j,i)
#pragma xmp align b[i][j] with t2(j,i)
#pragma xmp align c[i][j] with t3(j,i)
int main(void){

   if(xmp_num_nodes() < 4){
      printf("%s\n","You have to run this program by more than 4 nodes.");
   }
  
   sa=1;
   sb=1.0;
   sc=1.0;

#pragma xmp loop (j,i) on  t1(i,j)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i==j)&&(i%100==0)){
            a[i][j] = 2;
         }else{
            a[i][j] = 1;
         }
      }
   }

#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         m = j*N+i;
         if(m%4 == 0){
            b[i][j] = 1.0;
         }else if(m%4 == 1){
            b[i][j] = 2.0;
         }else if(m%4 == 2){
            b[i][j] = 4.0;
         }else{
            b[i][j] = 0.125;
         }
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         m = j*N+i;
         if(m%4 == 0){
            b[i][j] = 0.5;
         }else if(m%4 == 1){
            b[i][j] = 2.0;
         }else if(m%4 == 2){
            b[i][j] = 4.0;
         }else{
            b[i][j] = 0.25;
         }
      }
   }
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sa = sa*a[i][j];
      }
   }
#pragma xmp reduction (*:sa) async(1)

#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sb = sb*b[i][j];
      }
   }
#pragma xmp reduction (*:sb) async(2)

#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         sc = sc*c[i][j];
      }
   }
#pragma xmp reduction (*:sc) async(3)
#pragma xmp wait_async(1)
   sa = sa/2;
#pragma xmp wait_async(2)
   sb = sb/2.0;
#pragma xmp wait_async(3)
   sc = sc/4.0;
  
   result ="OK";
   if(sa != 512||abs(sb-0.5)>0.000001||abs(sc-0.25)>0.0001){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp049.c",result); 
   return 0;
}
      
         
      
   

   
