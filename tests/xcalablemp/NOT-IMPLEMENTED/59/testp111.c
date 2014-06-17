/*testp111.c*/
/*reflect指示文,1次元分散,width,asyncあり*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>   
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int a[N];
double b[N];
float c[N];
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)
#pragma xmp shadow a[1]
#pragma xmp shadow b[2]
#pragma xmp shadow c[3]
int w,i,ii,k;
char *result;
int main(void){
   if(N%xmp_num_nodes() == 0){
      w = N/xmp_num_nodes();
   }else{
      w = N/xmp_num_nodes()+1;
   }
   
   result = "OK";
   k = 1;
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      a[i] = k;
      b[i] = (double)k;
      c[i] = (float)k;
   }
#pragma xmp reflect (a) async(1)
#pragma xmp reflect (b) async(2)
#pragma xmp reflect (c) async(3)
#pragma xmp wait_async(1)
#pragma xmp loop on t(i)
   for(i=2;i<N;i++){
      for(ii=i-1;ii<i+2;ii++){
         if(a[ii] != k) result = "NG1";
      }
   }
#pragma xmp wait_async(2)
#pragma xmp loop on t(i)
   for(i=3;i<N-1;i++){
      for(ii=i-2;ii<i+3;ii++){
         if(b[ii] != (double)k) result = "NG2";
      }
   }
#pragma xmp wait_async(3)
#pragma xmp loop on t(i)
   for(i=4;i<N-2;i++){
      for(ii=i-3;ii<i+4;ii++){
         if(c[ii] != (float)k) result = "NG3";
      }
   }
   k = 2;
#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      a[i] = k;
      b[i] = (double)k;
      c[i] = (float)k;
   }
#pragma xmp reflect (a) async(1)
#pragma xmp reflect (b) width(2) async(2)
#pragma xmp reflect (c) width(2) async(3)
#pragma xmp wait_async(1)

#pragma xmp loop on t(i)
   for(i=2;i<N;i++){
      for(ii=i-1;ii<i+2;ii++){
         if(a[ii] != k) result = "NG4";
      }
   }
#pragma xmp wait_async(2)
#pragma xmp loop on t(i)
   for(i=3;i<N-1;i++){
      for(ii=i-2;ii<i+3;ii++){
         if(b[ii] != (double)k) result = "NG5";
      }
   }
#pragma xmp wait_async(3)
#pragma xmp loop on t(i)
   for(i=4;i<N-2;i++){
      for(ii=i-3;ii<i+4;ii++){
         if(c[ii] != (float)k) result = "NG6";
      }
      if(i==4||i==w+1){
         if(c[i-3] != 1.0) result = "NG7";
      }
      if(i==N-3||i==w*xmp_node_num()){
         if(c[i+3]!=1.0) result = "NG8";
      }
   } 
   printf("%d %s %s\n",xmp_node_num(),"testp111.c",result); 

   return 0;
}
