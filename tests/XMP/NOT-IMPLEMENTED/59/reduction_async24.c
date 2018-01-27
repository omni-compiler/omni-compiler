/*testp206.c*/
/*組込み手続きのテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
double time,tick,max_tick;
double xmp_wtime(),xmp_wtick();
int async_val;
char *result;     
int i;
int main(void){
   result = "OK";
   time = xmp_wtime();
   tick = xmp_wtick();
   max_tick = tick;

#pragma xmp reduction(max: tick) async(1)

   async_val = xmp_test_async(1);
   if(!async_val){
      async_val = xmp_test_async(1);
   }

#pragma xmp wait_async(1)
 
   if(tick != max_tick) result = "NG";

   printf("%d %s %s\n",xmp_node_num(),"testp206.c",result); 
   return 0;
}
      
         
      
   
   
