/*testp033.c*/
/*reduction指示文(&)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
#pragma xmp nodes p(*)
int procs,id;
int mask1,val1;
int mask2,val2;
char *result;
int i,w;
int main(void){

   if(xmp_num_nodes() > 31){
      printf("%s\n","You have to run this program by less than 32 nodes.");
   }

   procs = xmp_num_nodes();
   id = xmp_num_nodes()-1;
   result = "OK";
   w=1;
   for(i=0;i<procs;i++){
      w*2;
   }
   for(i=0;i<w;i=i+2){
      mask1 = 1 << id;
      val1 =!(i & mask1);
#pragma xmp reduction(&:val1) async(1)
      mask2 = 1 << id;
      val2 =!((i+1) & mask2);
#pragma xmp reduction(&:val2) async(2)
#pragma xmp wait_async(1)
      if((!(val1)) != i){
         result = "NG";
      }
#pragma xmp wait_async(2)
      if((!(val2)) != i+1){
         result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp033.c",result); 
   return 0;
}
      
         
      
   
