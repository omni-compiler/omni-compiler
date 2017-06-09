#include <xmp.h>
#include <stdio.h> 
#include <stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(4)
#pragma xmp template t(0:N-1)
int g[4] = {100,200,600,100};
#pragma xmp distribute t(gblock(g)) onto p
int a[N],   sa=0,ans,i,j,result=0;
double b[N],sb=0.0;
float c[N], sc=0.0;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)

int main(void)
{
#pragma xmp loop on t(j)
  for(j=0;j<N;j++){
    a[j] = 1;
    b[j] = 2.0;
    c[j] = 3.0;
  }

#pragma xmp loop on t(j)
  for(j=0;j<N;j++){
    sa += a[j];
    sb += b[j];
    sc += c[j];
  }
   
  ans = g[xmp_node_num()-1];
  if(sa != ans){
    result = -1;
  }
  if(abs(sb-2.0*(double)ans)>0.000000001){
    result = -1;
  }
  if(abs(sc-3*(float)ans)>0.000001){
    result = -1;
  }

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
   {
     if(result == 0){
       printf("PASS\n");
     } else{
       printf("ERROR\n");
       exit(1);
     }
   }

   return 0;
}    
