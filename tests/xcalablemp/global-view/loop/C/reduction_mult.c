#include <xmp.h>
#include <stdio.h>  
#include <stdlib.h> 
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
int a[N],sa=1;
double b[N],sb=1.0;
float c[N],sc=1.0;
int i,result=0;
#pragma xmp align a[i] with t(i)
#pragma xmp align b[i] with t(i)
#pragma xmp align c[i] with t(i)

int main(void)
{
#pragma xmp loop (i) on t(i)
  for(i=0;i<N;i++){
    if((i+1)%100 == 0){
      a[i] = 2;
      b[i] = 2.0;
      c[i] = 4.0;
    }else if((i+1)%100 == 50){
      a[i] = 1.0;
      b[i] = 0.5;
      c[i] = 0.25;
    }else{
      a[i] = 1;
      b[i] = 1.0;
      c[i] = 1.0;  
    }
  }

#pragma xmp loop on t(i) reduction(*:sa)
  for(i=0;i<N;i++)
    sa = sa*a[i];
  
#pragma xmp loop on t(i) reduction(*:sb)
  for(i=0;i<N;i++)
    sb = sb*b[i];
  
#pragma xmp loop on t(i) reduction(*:sc)
  for(i=0;i<N;i++)
    sc = sc*c[i];
  
  if((sa != 1024)||abs(sb-(double)1.0) >= 0.0000001 ||abs(sc-(float)1.0) >= 0.0001){
    result = -1;
  }
  
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
