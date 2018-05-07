/*testp070.c*/
/*task指示文とgmove指示文の組み合わせテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t1(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t1(*,*,gblock((/200,700,50,50/))) onto p
#pragma xmp template t2(0:N-1)
#pragma xmp distribute t2(*,*,gblock((/700,200,50,50/))) onto p
#pragma xmp template t3(0:N-1)
#pragma xmp distribute t3(*,*,gblock((/50,700,200,50/))) onto p
int a1[N],a2[N];
double b1[N],b2[N];
float c1[N],c2[N];
int i;
#pragma xmp align a1[i] with t1(*,*,i)
#pragma xmp align a2[i] with t2(*,*,i)
#pragma xmp align b1[i] with t3(*,*,i)
#pragma xmp align b2[i] with t1(*,*,i)
#pragma xmp align c1[i] with t2(*,*,i)
#pragma xmp align c2[i] with t3(*,*,i)
char *result;
int main(void){

   if(xmp_num_nodes() != 4){
      printf("%s","You have to run this program by 4 nodes.");
   }

#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      a1[i] = xmp_node_num();
      b2[i] = -1.0;
   }
#pragma xmp loop on t2(i)
   for(i=0;i<N;i++){
      a2[i] = -1;
      c1[i] = (float)xmp_node_num();
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      b1[i] = (double)xmp_node_num();
      c2[i] = -1.0;
   }
#pragma xmp task on p(1)
   {
#pragma xmp gmove in 
      b2[0:200] = b1[750:200];
   }

#pragma xmp task on p(2)
   {
#pragma xmp gmove in 
      a2[700:200] = a1[0:200];
   }

#pragma xmp task on p(3)
   {
#pragma xmp gmove in 
      c2[750:200] = c1[700:200];
   }
#pragma xmp barrier

   result = "OK";
#pragma xmp loop on t2(:,:,i)
   for(i=0;i<N;i++){
      if((i >= 700)&&(i <= 899)){
         if(a2[i] != 1){
            result = "NG";
         }
      }else{
         if(a2[i] != -1){
            result = "NG";
         }
      }
   }

#pragma xmp loop on t1(:,:,i)
   for(i=0;i<N;i++){
      if((i >= 0)&&(i <= 199)){
         if(b2[i] != 3.0){
            result = "NG";
         }
      }else{
         if(b2[i] != -1.0){
            result = "NG";
         }
      }
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      if((i >= 750)&&(i <= 949)){
         if(a2[i] != 2.0){
            result = "NG";
         }
      }else{
         if(a2[i] != -1.0){
            result = "NG";
         }
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp070.c",result);
   return 0;
}    
         
