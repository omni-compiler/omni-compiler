/*testp086.c*/
/*task指示文とbcast指示文の組み合わせ*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>      
#pragma xmp nodes p(4,4)
static const int N=1000;
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,block) onto p
int procs,w;
int a[N][N],sa,ansa;
double b[N][N],sb,ansb;
float c[N][N],sc,ansc;
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp align b[i][j] with t(j,i)
#pragma xmp align c[i][j] with t(j,i)
char *result;
int i,j;
int main(void){
   if(xmp_num_nodes() < 16){
      printf("%s\n","You have to run this program by 16 nodes.");
   }
#pragma xmp loop (j,i) on t(j,i)
   for(i=0;i<N;i++){
      for(j=0;j<N;j++){
         a[i][j] = j/10*N+i/10;
         b[i][j] = (double)(j*N+i);
         c[i][j] = (float)(j*N+i);
      }
   }

   sa = 0;
   sb = 0.0;
   sc = 0.0;
   procs = xmp_num_nodes();
   if(N%procs == 0){
      w = N/procs;
   }else{
      w = N/procs+1;
   }

#pragma xmp task on p(2:3,2:3)
   {
      if(procs == 16){
#pragma xmp loop (j,i) on t(j,i)
         for(i=w;i<3*w;i++){
            for(j=w;j<3+w;j++){
               sa = sa+a[i][j];
               sb = sb+b[i][j];
               sc = sc+c[i][j];
            }
         }
      }
#pragma xmp bcast (sa) from p(2,3) async(1)
#pragma xmp bcast (sb) from p(2,2) async(2)
#pragma xmp bcast (sc) from p(3,2) async(3)
#pragma xmp wait_async(1)
#pragma xmp wait_async(2)
#pragma xmp wait_async(3)
   }
 
   result = "OK";
   if(xmp_node_num()==6||xmp_node_num()==7||xmp_node_num()==10||xmp_node_num()==11){
      ansa = 0;
      ansb = 0.0;
      ansc = 0.0;
      for(i=2*w;i<3*w;i++){
         for(j=w;j<2*w;j++){
            ansa = ansa+j/10*N+i/10;
         }
      }
      for(i=w;i<2*w;i++){
         for(j=w;j<2*w;j++){
            ansb = ansb+(double)(j*N+i);
         }
      }
      for(i=w;i<2*w;i++){
         for(j=2*w;j<3*w;j++){
            ansc = ansc+(float)(j*N+i);
         }
      }
      if(sa!=ansa||sb!=ansb||sb!=ansc){
         result = "NG";
      }
   }else{
      if(sa!=0||sb!=0.0||sb!=0.0){
         result = "NG";
      }
   }  
     
   printf("%d %s %s\n",xmp_node_num(),"testp086.c",result);
   return 0;
}    
         
      
   
