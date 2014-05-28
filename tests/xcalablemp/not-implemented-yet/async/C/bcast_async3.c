/*testp097.c*/
/*loop指示文とbcast指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>    
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t1(0:N-1)
#pragma xmp template t2(0:N-1)
#pragma xmp template t3(0:N-1)
#pragma xmp distribute t1(cyclic(3)) onto p
#pragma xmp distribute t2(block) onto p
#pragma xmp distribute t3(cyclic) onto p
int a[N],aa;
double b[N],bb;
float c[N],cc;
#pragma xmp align a[i] with t1(i)
#pragma xmp align b[i] with t2(i)
#pragma xmp align c[i] with t3(i)
int i;
char *result;
int main(void){
#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      a[i] = -1;
   }
#pragma xmp loop on t2(i)
   for(i=0;i<N;i++){
      b[i] = -2.0;
   }

#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      c[i] = -3.0;
   }

   result = "OK";
#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      aa = i;
#pragma xmp bcast (aa) async(1)
      if(a[i] != -1) result = "NG1";
#pragma xmp wait_async(1)
      a[i] = aa;
   }
#pragma xmp loop on t2(i)
   for(i=0;i<N;i++){
      bb = (double)i;
#pragma xmp bcast (bb) async(1)
      if(b[i] != -2.0) result = "NG2";
#pragma xmp wait_async(2)
      b[i] = bb;
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      cc = i;
#pragma xmp bcast (cc) async(3)
      if(c[i] != -3.0) result = "NG3";
#pragma xmp wait_async(3)
      c[i] = cc;
   }
#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      if(a[i] != i) result = "NG4";
   }
#pragma xmp loop on t2(i)
   for(i=0;i<N;i++){
      if(abs(b[i]-(double)i) > 0.00000001) result = "NG5";
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      if(abs(c[i]-(float)i) > 0.0001) result = "NG6";
   }
   printf("%d %s %s\n",xmp_node_num(),"testp097.c",result); 
   return 0;
}
      
          
      
   
            
