#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t1(0:N-1,0:N-1)
#pragma xmp template t2(0:N-1,0:N-1)
#pragma xmp distribute t1(*,block) onto p
#pragma xmp distribute t2(block,*) onto p
int    a1[N],a2[N],sa=0;
double b1[N],b2[N],sb=0.0;
float  c1[N],c2[N],sc=0.0;
#pragma xmp align a1[i] with t1(*,i)
#pragma xmp align b1[i] with t1(*,i)
#pragma xmp align c1[i] with t1(*,i)
#pragma xmp align a2[i] with t2(i,*)
#pragma xmp align b2[i] with t2(i,*)
#pragma xmp align c2[i] with t2(i,*)
int i,j,result=0;
int main(void)
{
#pragma xmp loop on t1(*,i)
  for(i=0;i<N;i++){
    a1[i] = i;
    b1[i] = (double)i;
    c1[i] = (float)i;
  }

#pragma xmp loop on t2(i,*)
  for(i=0;i<N;i++){
    a2[i] = 1;
    b2[i] = 1.0;
    c2[i] = 1.0;
  }
  
#pragma xmp gmove
  a1[:] = a2[:];

#pragma xmp gmove 
  c1[:] = c2[:];

#pragma xmp gmove 
  b1[:] = b2[:];
   
#pragma xmp loop on t1(:,i) reduction(+:sa,sb,sc)
   for(i=0;i<N;i++){
     sa += a1[i];
     sb += b1[i];
     sc += c1[i];
   }
   
   if(sa != 1000||abs(sb-1000.0) > 0.000000001||abs(sc-1000.0) > 0.0001)
     result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
   {
     if(result == 0){
       printf("PASS\n");
     }
     else{
       fprintf(stderr, "ERROR\n");
       exit(1);
     }
   }

   return 0;
}
