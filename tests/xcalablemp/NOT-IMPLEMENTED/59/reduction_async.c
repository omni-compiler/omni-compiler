/*testp027.c*/
/*reduction構文(+)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int a[N],sa;
double b[N],sb;
float c[N],sc;
int procs,w,remain;
int *w1;
int i,j;
char *result;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)
int main(void){

   if(xmp_num_nodes() < 4){
      printf("%s\n","You have to run this program by more than 4 nodes.");
   }
  
   sa=0;
   sb=0.0;
   sc=0.0;

#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      a[i]=1;
      b[i]=0.5;
      c[i]=0.01;
   }

#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      sa = sa+a[i];
   }
#pragma xmp reduction (+:sa) on p(1:2) async(1)

#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      sb = sb+b[i];
   }

#pragma xmp wait_async(1)
   sa = sa+1000;

#pragma xmp reduction(+:sb) on p(2:3) async(1)

#pragma xmp loop on t(i)
   for(i=0;i<N;i++){
      sc = sc+c[i];
   }

#pragma xmp wait_async(1)
   sb = sb+30.0;

#pragma xmp reduction(+:sc) on p(3:4) async(1)
#pragma xmp wait_async(1) 
   sc = sc+25.0;

   procs = xmp_num_nodes();
   if((N%procs) == 0){
      w = N/procs;
   }else{
      w = N/procs+1;
   }
   w1 = (int *)malloc( procs );
   remain = N;
   for(i=1;i<procs;i++){
      if(w < remain){
         w1[i] = w;
      }else{
         w1[i] = remain;
      }
      remain = remain - w1[i];
   }
   w1[procs] = remain;
  
   result = "OK";
   if(xmp_node_num() == 1){
      if(sa != (w1[1]+w1[2]+1000)||abs(sb-((double)w1[1]*0.5+30.0)) > 0.000001||abs(sc-((float)w1[1]*0.01+25.0)) > 0.000001){
         result = "NG";
      }
   }else if(xmp_node_num() == 2){
      if(sa != (w1[1]+w1[2]+1000)||abs(sb-((double)(w1[2]+w1[3])*0.5+30.0)) > 0.000001||abs(sc-((float)w1[2]*0.01+25.0)) > 0.000001){
         result = "NG";
      }
   }else if(xmp_node_num() == 3){
      if(sa != (w1[3]+1000)||abs(sb-((double)(w1[2]+w1[3])*0.5+30.0)) > 0.000001||abs(sc-((float)(w1[3]+w1[4])*0.01+25.0)) > 0.000001){
         result = "NG";
      }
   }else if(xmp_node_num() == 4){
      if(sa != w1[4]+1000||abs(sb-((double)w1[4]*0.5+30.0)) > 0.000001||abs(sc-((float)(w1[3]+w1[4])*0.01+25.0)) > 0.000001){
         result = "NG";
      }
   }else{
      i=xmp_node_num();
      if(sa != w1[i]+1000||abs(sb-((double)w1[i]*0.5+30.0)) > 0.000001||abs(sc-((float)w1[i]*0.01+25.0)) > 0.000001){
         result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp027.c",result); 
   return 0;
}
      
         
      
   

   
