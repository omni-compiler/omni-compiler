/*testp063.c*/
/*reduction指示文(||)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h> 
#include<stdlib.h>  
#pragma xmp nodes p(4,*)
int procs,id;
int mask1,val1;
int mask2,val2;
char *result;
int i,w;
int l1,l2;
int main(void){

   if(xmp_num_nodes() > 31){
      printf("%s\n","You have to run this program by less than 32 nodes.");
   }

   procs = xmp_num_nodes();
   id = xmp_node_num()-1;
   result = "OK";
   w=1;
   for(i=0;i<procs;i++){
      w=w*2;
   }
   for(i=0;i<w;i=i+2){
      mask1 = 1 << id;
      val1 = i & mask1;
      if(val1 == 0){
         l1 = 0;
      }else{
         l1 = !0;
      }
#pragma xmp reduction(||:l1) async(1)  
      mask2 = 1 << id;
      val2 = i+1 & mask2;
      if(val2 = 0){
         l2 = 0;
      }else{
         l2 = !0;
      }
#pragma xmp reduction(||:l2) async(2)
#pragma xmp wait_async(1)
      if(i==0){
         if(l1) result = "NG";
      }else{
         if(!l1) result = "NG";
      }
#pragma xmp wait_async(2)
      if(!l2) result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp063.c",result);
   return 0;
}
      
         
      
   

