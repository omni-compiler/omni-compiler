/*testp154.c*/
/*loop指示文とreduction節(-)のテスト*/
#include<mpi.h>
#include<xmp.h>
#include<stdio.h>  
#include<stdlib.h> 
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic) onto p
int a[N],sa;
double b[N],sb;
float c[N],sc;
int i,*w;
int ans,procs;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)
char *result;
int main(void){
#pragma xmp loop (i) on t(i)
   for(i=0;i<N;i++){
      a[i]=xmp_node_num();
      b[i]=(double)xmp_node_num();
      c[i]=(float)xmp_node_num();
   }
   sa = 0;
   sb = 0.0;
   sc = 0.0;

#pragma xmp loop on t(i) reduction(-:sa)
   for(i=0;i<N;i++)
      sa+=a[i];

#pragma xmp loop on t(i) reduction(-:sb)
   for(i=0;i<N;i++)
      sb+=b[i];

#pragma xmp loop on t(i) reduction(-:sc)
   for(i=0;i<N;i++)
      sc+=c[i];

   procs = xmp_num_nodes();
   w = (int *)malloc(procs);
   if(N%procs == 0){
      for(i=1;i<procs+1;i++){
         w[i] = N/procs;
      }
   }else{
      for(i=1;i<procs+1;i++){
         if(i <= N%procs){
            w[i]=N/procs+1;
         }else{
            w[i]=N/procs;
         }
      }
   }
   ans=0;
   for(i=1;i<procs+1;i++){
      ans = ans + i*w[i];
   }
   result="OK";
   if((sa != ans)||abs(sb-(double)ans) >= 0.0000001 ||abs(sc-(float)ans) >= 0.0001){
      result = "NG";  
   }
   printf("%d %s %s\n",xmp_node_num(),"testp154.c",result);
   free(w); 
   return 0;
}
