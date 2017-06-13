#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(cyclic(3)) onto p
int N,s=0,*a,i,*x,result=0;
#pragma xmp align a[i] with t(i)

int main(void)
{
  N = 1000;
#pragma xmp template_fix(cyclic(3)) t(0:N-1)
  a = (int *)xmp_malloc(xmp_desc_of(a), N);

#pragma xmp loop on t(i)
  for(i=0;i<N;i++)
    a[i] = i;

#pragma xmp loop on t(i) reduction(+:s)
  for(i=0;i<N;i++)
    s = s + a[i];

  if(s != 499500)
    result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
  {
    if(result == 0 ){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR\n");
      exit(1);
    }
  }
  
  return 0;
}
