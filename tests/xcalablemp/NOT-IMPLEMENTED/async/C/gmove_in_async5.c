/*testp092.c*/
/*loop指示文とgmove指示文のテスト*/
#include<xmp.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>      
#pragma xmp nodes p(*)
static const int N=1000;
#pragma xmp template t1(0:N-1)
#pragma xmp template t2(0:N-1)
#pragma xmp template t3(0:N-1)
#pragma xmp template t4(0:N-1)
#pragma xmp template t5(0:N-1)
#pragma xmp template t6(0:N-1)
#pragma xmp distribute t1(block) onto p
#pragma xmp distribute t2(cyclic) onto p
#pragma xmp distribute t3(cyclic(2)) onto p
#pragma xmp distribute t4(cyclic(3)) onto p
#pragma xmp distribute t5(cyclic(5)) onto p
#pragma xmp distribute t6(cyclic(7)) onto p
int a1[N],a2[N];
double b1[N],b2[N];
float c1[N],c2[N];
#pragma xmp align a1[i] with t1(i)
#pragma xmp align a2[i] with t2(i)
#pragma xmp align b1[i] with t3(i)
#pragma xmp align b2[i] with t4(i)
#pragma xmp align c1[i] with t5(i)
#pragma xmp align c2[i] with t6(i)
int i;
char *result;
int main(void){
#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      a1[i] = 0;
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      b1[i] = 0.0;
   }
#pragma xmp loop on t5(i)
   for(i=0;i<N;i++){
      c1[i] = 0.0;
   }
#pragma xmp loop on t2(i)
   for(i=0;i<N;i++){
      a2[i] = i;
   }
#pragma xmp loop on t4(i)
   for(i=0;i<N;i++){
      b2[i] = (double)i;
   }
#pragma xmp loop on t6(i)
   for(i=0;i<N;i++){
      c2[i] = (float)i;
   }

#pragma xmp loop on t1(i) 
   for(i=0;i<N;i++){
#pragma xmp gmove in async(1)
      a1[i:1] = a2[i:1];
#pragma xmp wait_async(1)
   }

#pragma xmp loop on t3(i) 
   for(i=0;i<N;i++){
#pragma xmp gmove in async(1)
      b1[i:1] = b2[i:1];
#pragma xmp wait_async(1)
   }

#pragma xmp loop on t6(i) 
   for(i=0;i<N;i++){
#pragma xmp gmove out async(1)
      c1[i:1] = c2[i:1];
#pragma xmp wait_async(1)
   }
   
   result = "OK";
#pragma xmp loop on t1(i)
   for(i=0;i<N;i++){
      if(a1[i] != i) result = "NG1";
   }
#pragma xmp loop on t3(i)
   for(i=0;i<N;i++){
      if(abs(b1[i]-(double)i) > 0.00000001) result = "NG2";
   }
#pragma xmp loop on t5(i)
   for(i=0;i<N;i++){
      if(abs(c1[i]-(float)i) > 0.001) result = "NG3";
   }
   printf("%d %s %s\n",xmp_node_num(),"testp092.c",result);
   return 0;
}    
         
      
   
