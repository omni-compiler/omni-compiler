int sub(int *x,int N){
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(cyclic(3)) onto p
#pragma xmp align x[i] with t(i)
   return 0;
}

#include<xmp.h>
#include<stdio.h>
#include<stdlib.h>
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(cyclic(3)) onto p
int N,s,*a,i,*x,result=0;
#pragma xmp align a[i] with t(i)

int main(void)
{
  N = 1000;
#pragma xmp template_fix(cyclic(3)) t(0:N-1)
  a = (int *)malloc(N);
  sub(a,N);   

#pragma xmp loop (i) on t(i)
  for(i=0;i<N;i++)
    a[i] = i;

   s = 0;
#pragma xmp loop (i) on t(i) reduction(+:s)
   for(i=0;i<N;i++)
      s = s+a[i];

   if(s != 499500)
     result = -1;

   return 0;
}
               
      
   
