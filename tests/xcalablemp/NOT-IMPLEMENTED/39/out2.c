/*testp018.c*/
/*gmove指示文のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp distribute t1(*,block) onto p
#pragma xmp distribute t2(cyclic,*) onto p
int a1[N],a2[N],sa;
double b1[N],b2[N],sb;
float c1[N],c2[N],sc;
#pragma xmp align a1[i] with t1(*,i)
#pragma xmp align b1[i] with t1(*,i)
#pragma xmp align c1[i] with t1(*,i)
#pragma xmp align a2[i] with t2(i,*)
#pragma xmp align b2[i] with t2(i,*)
#pragma xmp align c2[i] with t2(i,*)
int procs;
int i,j;
char *result;
int main(void){

#pragma xmp loop on t1(:,i)
   for(i=0;i<N;i++){
      a1[i] = i;
      b1[i] = (double)i;
      c1[i] = (float)i;
   }

#pragma xmp loop on t2(i,:)
   for(i=0;i<N;i++){
      a2[i] = 1;
      b2[i] = 1.0;
      c2[i] = 1.0;
   }

   procs = xmp_num_nodes();
   for(j=1;j<procs+1;j++){
      if(j == xmp_node_num()){
#pragma xmp gmove out 
         a1[:] = a2[:];
#pragma xmp gmove out 
         c1[:] = c2[:];
#pragma xmp gmove out 
         b1[:] = b2[:];
      }
   }

   sa = 0;
   sb = 0.0;
   sc = 0.0;
#pragma xmp loop on t1(:,i) reduction(+:sa,sb,sc)
   for(i=0;i<N;i++){
      sa = sa+a1[i];
      sb = sb+b1[i];
      sc = sc+c1[i];
   }

   result = "OK";
   if(sa != 1000||abs(sb-1000.0) > 0.000000001|| abs(sc-1000.0) > 0.0001){
      result = "NG";
   }

   printf("%d %s %s\n",xmp_node_num(),"testp018.c",result); 
   return 0;
}
      
         
      
   
















