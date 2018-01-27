/*testp95.c*/
/*loop指示文とreduction指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>   
#pragma xmp nodes p(4)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t(*,*,gblock((/333,555,111,1))) onto p
int a1[N],a2[N],aa;
#pragma xmp align a1[i] with t(*,*,i)
#pragma xmp align a2[i] with t(*,*,i)
int i;
char *result;
int main(void){
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      a1[i] = 0;
      a2[i] = 0;
   }
   result = "OK";
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      aa = a2[i];
#pragma xmp reduction(||: aa) async(1)
      if(a1[i] != 0) result = "NG";
#pragma xmp wait_async(1)
      aa = aa-1;
      a1[i] = aa;
   }
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++)
      if(a1[i] != i-1) result = "NG";

   printf("%d %s %s\n",xmp_node_num(),"testp095.c",result); 
   return 0;
}
      
         
      
   
