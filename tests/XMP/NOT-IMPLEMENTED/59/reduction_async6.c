/*testp037.c */
/*reduction指示文(^)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
int random_array[1000],ans_val;
int a[N],sa;
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
#pragma xmp align a[i] with t(i)
char *result;
int i,k;
int main(void){

   result = "OK";
   for(k=114;k<10001;k=k+113){
      for(i=1;i<N;i++){
         random_array[i]=(random_array[i-1]*random_array[i-1])%100000000;
         random_array[i]=(random_array[i]-((random_array[i]%100)/100))%10000;
      }
   
#pragma xmp loop on t(i)
      for(i=0;i<N;i++){
         a[i] = random_array[i];
      }

      sa = 0;
#pragma xmp loop on t(i)
      for(i=0;i<N;i++){
         sa = sa^a[i];
      }
#pragma xmp reduction(^:sa) async(1)
       
      ans_val = 0;
      for(i=0;i<N;i++){
         ans_val = ans_val^random_array[i];
      }   
   
#pragma xmp wait_async(1)
   
      if(sa != ans_val){
         result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp037.c",result); 
   return 0;
}
      
         
      
   
