/*testp082.c*/
/*task指示文とgmove指示文の組み合わせテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
static const int N=1000;
#pragma xmp nodes p(4,4)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp distribute t1(block,gblock((/200,700,50,50/))) onto p
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp distribute t2(block,gblock((/700,200,50,50/))) onto p
#pragma xmp template t3(0:N-1,0:N-1)
#pragma xmp distribute t3(gblock((/250,250,250,250/)),gblock((/50,700,200,50/))) onto p
int a1[N][N],a2[N][N];
double b1[N][N],b2[N][N];
float c1[N][N],c2[N][N];
int i,j;
#pragma xmp align a1[i][j] with t1(j,i)
#pragma xmp align a2[i][j] with t2(j,i)
#pragma xmp align b1[i][j] with t3(j,i)
#pragma xmp align b2[i][j] with t4(j,i)
#pragma xmp align c1[i][j] with t5(j,i)
#pragma xmp align c2[i][j] with t6(j,i)
char *result;
int main(void){

   if(xmp_num_nodes() != 16){
      printf("%s","You have to run this program by 16 nodes.");
   }
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a1[i][j] = xmp_node_num();
         b2[i][j] = -1.0;
      }
   }
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b2[i][j] = -1.0;
         c2[i][j] = (float)xmp_node_num();
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         b1[i][j] = (double)xmp_node_num();
         c2[i][j] = -1.0;
      }
   }
#pragma xmp task on p(1,1)
   {
#pragma xmp gmove in 
      b2[0:250][0:200] = b1[500:250][750:200];
   }
#pragma xmp task on p(2,2)
   {
#pragma xmp gmove in 
      a2[250:250][700:200] = a1[750:250][0:200];
   }
#pragma xmp task on p(3,3)
   {
#pragma xmp gmove in 
      c2[500:250][750:200] = c1[250:250][700:200];
   }

   result = "OK";
#pragma xmp loop (j,i) on t2(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i >= 250)&&(i <= 500)&&(j >= 701)&&(j <= 900)){
            if(a2[i][j] != 13){
               result = "NG";
            }
         }else{
            if(a2[i][j] != -1){
               result = "NG";
            }
         }
      }
   }
#pragma xmp loop (j,i) on t1(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i >= 0)&&(i <= 199)&&(j >= 1)&&(j <= 249)){
            if(b2[i][j] != 4.0){
               result = "NG";
            }
         }else{
            if(b2[i][j] != -1.0){
               result = "NG";
            }
         }
      }
   }
#pragma xmp loop (j,i) on t3(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         if((i >= 500)&&(i <= 749)&&(j >= 750)&&(j <= 949)){
            if(c2[i][j] != 6.0){
               result = "NG";
            }
         }else{
            if(c2[i][j] != -1.0){
               result = "NG";
            }
         }
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp081.c",result);
   return 0;
}    
         

