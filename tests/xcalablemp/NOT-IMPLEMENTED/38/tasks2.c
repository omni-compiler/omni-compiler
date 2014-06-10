/*testp077.c*/
/*tasks指示文およびtask指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>      
#pragma xmp nodes p(4,*)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int a[N][N],sa,ansa,lb,ub,procs,procs2;
double b[N][N],sb,ansb;
float c[N][N],sc,ansc;
int i,j;
char *result;
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
#pragma xmp shadow a[1][1]
#pragma xmp shadow b[2][2]
#pragma xmp shadow c[3][3]
int main(void){

   sa = 0;
   sb = 0.0;
   sc = 0.0;

#pragma xmp loop on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j]=2;
         b[i][j]=1.5;
         c[i][j]=0.5;
      }
   }

   procs = xmp_num_nodes();
   procs2 = procs/4;

#pragma xmp tasks
   {
#pragma xmp task on p(1,1)
      {
#pragma xmp loop (j,i) on t(j,i)
         for(i=0;i<N/procs2;i++){
            for(j=0;j<N/4;j++){
               sa = sa+a[i][j];
               sb = sb+b[i][j];
               sc = sc+c[i][j];
            }
         }
      }
#pragma xmp task on p(1,2)
      {
#pragma xmp loop (j,i) on t(j,i)
         for(i=N/procs2;i<N/procs2*2;i++){
            for(j=0;j<N/4;j++){
               sa = sa+a[i][j];
               sb = sb+b[i][j];
               sc = sc+c[i][j];
            }
         }
      }
#pragma xmp task on p(2,1)
      {
#pragma xmp loop (j,i) on t(j,i)
         for(i=0;i<N/procs2;i++){
            for(j=N/4;j<N/2;j++){
               sa = sa+a[i][j];
               sb = sb+b[i][j];
               sc = sc+c[i][j];
            }
         }
      }
#pragma xmp task on p(2,2)
      {
#pragma xmp loop (j,i) on t(j,i)
         for(i=N/procs2+1;i<N/procs2*2;i++){
            for(j=N/4;j<N/2;j++){
               sa = sa+a[i][j];
               sb = sb+b[i][j];
               sc = sc+c[i][j];
            }
         }
      }
   }
#pragma xmp reduction (+:sa,sb,sc)
   result = "OK";
   ansa = (N*N/procs2)*2;
   if(sa != ansa){
      result = "NG";
   }
   ansb = (double)((N*N/procs2)*1.5);
   if(abs(sb-ansb) > 0.0000001){
      result = "NG";
   }
   ansc = (float)(N*N/procs2)*0.5;
   if(abs(sc-ansc) > 0.00001){
      result = "NG";
   }
   
   printf("%d %s %s\n",xmp_node_num(),"testp077.c",result);
   return 0;
}    

         
      
   
