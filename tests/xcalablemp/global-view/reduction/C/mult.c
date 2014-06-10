#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>   
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(0:N-1,0:N-1)
#pragma xmp distribute t(block,*) onto p 
int a[1000],   sa=1,  ansa=1024, i, result=0;
double b[1000],sb=1.0,ansb=1024.0;
float c[1000], sc=1.0,ansc=1024.0;
#pragma xmp align a[i] with t(i,*)
#pragma xmp align b[i] with t(i,*)
#pragma xmp align c[i] with t(i,*)

int main(void)
{
#pragma xmp loop on t(i,:)
  for(i=0;i<N;i++){
    if(i%100 == 0){
      a[i]=2;
      b[i]=2.0;
      c[i]=2.0;
    }else{
      a[i]=1;
      b[i]=1.0;
      c[i]=1.0;
    }
  }

#pragma xmp loop on t(i,:)
  for(i=0;i<N;i++)
    sa = sa*a[i];
#pragma xmp reduction(*:sa)
 
#pragma xmp loop on t(i,:)
  for(i=0;i<N;i++)
    sb = sb*b[i];
#pragma xmp reduction(*:sb) 

#pragma xmp loop on t(i,:)
  for(i=0;i<N;i++)
     sc = sc*c[i];
#pragma xmp reduction(*:sc)

  if(sa != ansa||abs(sb-ansb) > 0.000001||abs(sc-ansc) > 0.000001)
    result = -1; // error;
#pragma xmp reduction(+:result)

#pragma xmp task on p(1)
   {
     if(result == 0){
       printf("PASS\n");
     } else{
       fprintf(stderr, "ERROR\n");
       exit(1);
     }
   }
   return 0;
}

