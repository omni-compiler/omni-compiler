/*testp065.c*/
/*tasks指示文およびtask指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>      
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1,0:N-1,0:N-1)
#pragma xmp distribute t(*,*,block) onto p
int a[N],s,lb,ub,procs ,w,ans;
int i;
char *result;
#pragma xmp align a[i] with t(*,*,i)
#pragma xmp shadow a[1]
int main(void){

   s=0;
#pragma xmp loop on t(:,:,i)
   for(i=0;i<N;i++){
      a[i]=i;
   }

   procs = xmp_num_nodes();

   if(N%procs == 0){
      w = N/procs;
   }else{
      w = N/procs+1;
   }

   lb = (xmp_node_num()-1)*w;
   ub = xmp_node_num()*w;

#pragma xmp tasks
   {
#pragma xmp task on p(1)
      {
         for(i=lb;i<ub+1;i++){
            s = s+a[i];
         }
      }
#pragma xmp task on p(2)
      {
         if(procs > 1){
            for(i=lb;i<ub+1;i++){
               s = s+a[i];
            }
         }
      }
#pragma xmp task on p(4)
      {
         if(procs > 3){
            for(i=lb;i<ub+1;i++){
               s = s+a[i];
            }
         }
      }
   }
#pragma xmp reduction (+:s)
   if(procs < 2){
      ans = w*(w+1)/2;
   }else if(procs < 4){
      ans = w*(w*2+1);
   }else{
      ans = w*(w*2+1) + w*2*(w*4+1) - w*3*(w*3+1)/2;
   }

   if(ans == s){
      result = "OK";
   }else{
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp065.c",result);
   return 0;
}    

         
      
   
