/*testp084.c*/
/*task指示文とredeuction指示文の組み合わせ*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h> 
#include<stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(4,4)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(gblock((/50,50,50,850/)),block) onto p
int procs;
int a[N][N],sa,ans;
double b[N][N],sb;
float c[N][N],sc;
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
int i,j;
char *result;
int main(void){

   if(xmp_num_nodes() != 16){
      printf("%s","You have to run this program by 16 nodes.");
   }

   sa=0;
   sb=0.0;
   sc=0.0;

#pragma xmp loop (j,i) on t(j)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = 1;
         b[i][j] = 2.0;
         c[i][j] = 3.0;
      }
   }

   procs = xmp_num_nodes();

#pragma xmp task on p(1,1)
   {
#pragma xmp loop on t(j)
      for(i=0;i<250;i++){
         for(j=0;j<50;j++){
            sa =sa+a[i][j];
            sb =sb+b[i][j];
            sc =sc+c[i][j];
         }
      }
   }
#pragma xmp task on p(2:4,2:4)
   {
      if(procs == 16){
#pragma xmp loop (j,i) on t(j)
         for(i=750;i<N;i++){
            for(j=150;j<N;j++){
               sa =sa+a[i][j];
               sb =sb+b[i][j];
               sc =sc+c[i][j];
            }
         }
      }
#pragma xmp reduction(+:sa,sb,sc) async(1)
#pragma xmp wait_async(1)
      sa = sa+1;
      sb = sb+1.0;
      sc = sc+1.0;
   }

   result = "OK";
   if(xmp_node_num() == 1){
      ans = 12500;
      if(sa != ans){
         result = "NG";
      }
      if(abs(sb-2.0*(double)ans)>0.000000001){
         result = "NG";
      }
      if(abs(sc-3*(float)ans)>0.000001){
         result = "NG";
      }
   }else if(xmp_node_num()==6||xmp_node_num()==7||xmp_node_num()==8||xmp_node_num()==9||xmp_node_num()==10||xmp_node_num()==11||xmp_node_num()==12||xmp_node_num()==13||xmp_node_num()==14||xmp_node_num()==15||xmp_node_num()==16 ){
      ans = 212501;
      if(sa != ans){
         result = "NG";
      }
      if(abs(sb-2.0*(double)ans)>0.000000001){
         result = "NG";
      }
      if(abs(sc-3*(float)ans)>0.000001){
         result = "NG";
      }  
   }else{
      if((sa!=0)||(sb!=0.0)||(sc!=0.0)){
         result = "NG";
      }
   }

   printf("%d %s %s\n",xmp_node_num(),"testp084.c",result);
   return 0;
}    
         
      
   
   
   
